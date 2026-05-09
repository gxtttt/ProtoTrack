# ProtoTrack for RGB-T Tracking

This repository contains the official implementation of the RGB-T object tracking framework, **ProtoTrack**.

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

Put the tracking datasets in `./data`. Based on the evaluation scripts, our framework supports `LasHeR`, `RGBT210`, and `RGBT234`. It should look like:

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

Download Pretrained weights (e.g., ImageNet or SOT) and put them under `$PROJECT_ROOT$/pretrained_models`.

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

## Acknowledgments

Our project is developed upon 

[OSTrack\]: https://github.com/botaoye/OSTrack

. Thanks for their brilliant contributions to the community!
