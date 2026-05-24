# Purify-then-Fuse: Prototype-Guided Feature Learning for Robust RGB-T Visual Tracking

This repository contains the official implementation of the RGB-T object tracking framework, **ProtoTrack**, submitted to *[The Visual Computer](https://link.springer.com/journal/371)*.

> **Note:** This code is directly related to our manuscript currently submitted to *The Visual Computer*. If you find this code or our Purify-then-Fuse framework useful for your research, we strongly encourage you to contact us and cite our relevant manuscript .



## Highlights

- **Purify-then-Fuse Paradigm:** A novel framework that explicitly purifies modal features before cross-modal fusion, preventing noise propagation from degraded modalities.
- **Prototype Reconstruction Module (PRM):** Decomposes and reconstructs target appearance features via learnable prototypes, suppressing intra-modal background noise while preserving discriminative details.
- **Deep Cross-modal Fusion (DCF):** A bi-directional self-guided attention mechanism that progressively integrates purified RGB and TIR features, exploiting complementary information across modalities.



## Environment Installation

```
conda create -n prototrack python=3.8
conda activate prototrack
bash install.sh
```



## Project Paths Setup

Run the following command to set paths for this project:

```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files:

```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```



## Data Preparation

ut the tracking datasets in `./data`. Based on the evaluation scripts, our framework supports `LasHeR`, `RGBT210`, and `RGBT234`. Download datasets:

| Dataset | Download Link                                                |
| ------- | ------------------------------------------------------------ |
| LasHeR  | [BaiduNetdisk](https://pan.baidu.com/s/1SCkEmOoP8LcAhdzXSX_YXQ?pwd=kek1) |
| RGBT210 | [BaiduNetdisk](https://pan.baidu.com/s/1mT8zP-e-ILDu0GfrfaKK-w?pwd=9df2) |
| RGBT234 | [BaiduNetdisk](https://pan.baidu.com/s/1ncwZXmy-ygoI0vOU37HhWA?pwd=ev6s) |

It should look like:

```
${PROJECT_ROOT}
  -- data
      -- lasher
          |-- trainingset
          |-- testingset
          |-- trainingsetList.txt
          |-- testingsetList.txt
      -- rgbt210
          ...
      -- rgbt234
          ...
```



## Training

Download Pretrained weights (e.g., [ImageNet or SOT](https://pan.baidu.com/s/17W9qq_WFweByg0VN72DjDA?pwd=3p8f)) and put them under `$PROJECT_ROOT$/pretrained_models`.

To train the ProtoTrack model, run:

```
python tracking/train.py --script prototrack --config vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot --save_dir ./output/vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot --mode multiple --nproc_per_node 2
```

You can replace `--config` with other desired model configs located under `experiments/prototrack/`.



## Evaluation

Put the checkpoint into `$PROJECT_ROOT$/output/prototrack/vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot/...` or modify the checkpoint path in the testing code.

To evaluate the model on the testing set (e.g., LasHeR), run:

```
python tracking/test.py prototrack vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot --dataset_name lasher_test --threads 6 --num_gpus 1
```

To analyze the results and compute metrics, run:

```
python tracking/analysis_results.py --tracker_name prototrack --tracker_param vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot --dataset_name lasher_test
```

*(Note: You can also evaluate on RGBT210 or RGBT234 by modifying `--dataset_name rgbt210` or `--dataset_name rgbt234`)*



## Results on LasHeR testing set

| Model      | Backbone | Pretraining | Precision | NormPrec | Success | Checkpoint                                                   | Raw Result                                                   |
| ---------- | -------- | ----------- | --------- | -------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ptototrack | ViT-Base | SOT         | 74.2      | 70.2     | 59.3    | [download](https://pan.baidu.com/s/1bodOHxBQjiSw46Dp3B9_zw?pwd=26tc) | [download](https://pan.baidu.com/s/1xcwWwX9v3XKRCD-uiysw6Q?pwd=m1hk) |



## Acknowledgments

Our project is developed upon [OSTrack](https://github.com/botaoye/OSTrack). Thanks for their brilliant contributions to the community!

