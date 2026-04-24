import math

from lib.models.tbsi_track import build_tbsi_track
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class TBSITrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TBSITrack, self).__init__(params)
        network = build_tbsi_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        # 多模板缓存：列表，元素为Preprocessor输出的NestedTensor（包含RGB+IR拼接张量）
        self.template_list = []
        # 模板更新策略参数
        self.max_templates = getattr(self.cfg.TEST, 'MAX_TEMPLATES', 3)
        self.update_interval = getattr(self.cfg.TEST, 'UPDATE_INTERVAL', 10)
        self.conf_threshold = getattr(self.cfg.TEST, 'CONF_THRESHOLD', 0.0)

        print("max templates is: ", self.max_templates)
        print("Update interval is: ", self.update_interval)
        print("conf threshold is: ", self.conf_threshold)

    def initialize(self, image, info: dict):
        # forward the template once
        #从原始图像(image)中，根据初始给定的目标框(info['init_bbox'])，裁剪出一块区域作为模板
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        # 使用预处理器(preprocessor)将裁剪出的图像块(z_patch_arr)转换成神经网络需要的张量(tensor)格式
        # 这个过程通常包括：转为PyTorch张量、归一化、调整维度顺序等
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        #no_grad: 表示在计算过程中不进行梯度计算，从而节省内存和计算资源
        with torch.no_grad():
            self.z_dict1 = template
            # 初始化模板列表，索引0保留为初始模板
            self.template_list = [template]
            # 如允许多个模板，预置第二个模板为初始模板的副本
            if self.max_templates >= 2:
                self.template_list.append(template)

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def _maybe_update_templates(self, image, pred_score_map):
        """根据分数与时间间隔动态更新模板列表，保留第一个初始模板。
        - 时间间隔: 每update_interval帧最多更新一次
        - 置信度: 分数图最大值超过阈值才更新
        - 截断: 最多保留max_templates个模板
        """
        if self.update_interval <= 0:
            return
        if (self.frame_id % self.update_interval) != 0:
            return
        if pred_score_map is None:
            return
        max_conf = float(pred_score_map.max().item())
        if max_conf < self.conf_threshold:
            return

        # 1. 根据最新的预测框 self.state，从当前帧图像中重新裁剪一个标准尺寸的模板
        # 注意：这里的 output_sz 必须是 self.params.template_size
        new_patch_arr, _, new_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)

        # 2. 对这个新的图像块进行预处理，得到符合要求的张量
        new_template = self.preprocessor.process(new_patch_arr, new_amask_arr)
       
        # 插入到列表末尾并截断，保留模板0为初始模板
        if len(self.template_list) < self.max_templates:
            self.template_list.append(new_template)
        else:
            # 丢弃最旧的非初始模板
            self.template_list = [self.template_list[0]] + self.template_list[2:] + [new_template]

    def _split_rgb_ir(self, nested_tensor: 'NestedTensor'):
        tensors = nested_tensor.tensors
        return tensors[:,:3,:,:], tensors[:,3:,:,:]

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            # 组织多模板输入：将模板列表拆分为RGB/IR两个列表
            z_list_rgb = []
            z_list_ir = []
            for nt in self.template_list:
                z_rgb, z_ir = self._split_rgb_ir(nt)
                z_list_rgb.append(z_rgb)
                z_list_ir.append(z_ir)

            x_rgb, x_ir = self._split_rgb_ir(search)
            # 运行网络
            out_dict = self.network.forward(
                template=[z_list_rgb, z_list_ir], search=[x_rgb, x_ir], ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # 多模板：根据策略更新模板列表
        self._maybe_update_templates(image, pred_score_map)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return TBSITrack
