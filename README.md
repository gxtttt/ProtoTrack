# Refine-then-Fuse: Prototype-Guided Feature Refinement for Robust RGB-T Visual Tracking

This repository contains the official PyTorch implementation of **ProtoTrack**, the RGB-T visual tracking framework described in our manuscript submitted to *The Visual Computer*.

ProtoTrack follows a **Refine-then-Fuse** strategy. The Prototype Reconstruction Module (PRM) refines contextualized RGB and thermal-infrared token representations through decoupled prototype addressing and reconstruction before cross-modal interaction. The Deep Cross-modal Fusion (DCF) module then integrates the refined branch representations through self-guided cross-modal attention and deep aggregation.

> The revised terminology deliberately uses **refinement** rather than claiming that all modality noise is removed. The experiments support improved target-related representation and tracking performance, but they should not be interpreted as a universal guarantee of feature purification.

## Highlights

- **Refine-then-Fuse design.** PRM is placed before cross-modal interaction so that each modality branch is refined before fusion.
- **Prototype Reconstruction Module.** PRM uses decoupled assignment and reconstruction parameters to read token representations from learnable prototype memories.
- **Deep Cross-modal Fusion.** DCF performs self-guided cross-modal interaction and deep aggregation between the RGB and TIR branches.
- **Controlled validation.** The revised study includes parameter-matched bottleneck comparisons, alternative memory addressing, processing-order ablations, DCF design ablations, response-map comparisons, and prototype-assignment analysis.
- **Competitive performance.** ProtoTrack obtains leading results on LasHeR and remains competitive on RGBT210 and RGBT234.

## Main Results

| Benchmark | Metric | ProtoTrack |
|---|---:|---:|
| LasHeR | Precision Rate (PR) | 74.2 |
| LasHeR | Normalized Precision Rate (NPR) | 70.2 |
| LasHeR | Success Rate (SR) | 59.3 |
| RGBT210 | Maximum Precision Rate (MPR) | 88.3 |
| RGBT210 | Maximum Success Rate (MSR) | 64.3 |
| RGBT234 | Maximum Precision Rate (MPR) | 88.8 |
| RGBT234 | Maximum Success Rate (MSR) | 65.4 |

Results are reported in percent. ProtoTrack runs at **36.292 FPS** with **153.950 M parameters** and **89.625 G MACs** under the paper's profiling protocol on a single RTX 2080 Ti.

## Repository Structure

```text
ProtoTrack/
├── experiments/prototrack/   # experiment configurations
├── lib/                      # model, training, and evaluation code
├── tracking/                 # training, testing, analysis, and profiling entry points
├── environment.yml           # reproducible Conda environment
├── install.sh                # original installation script
├── REPRODUCIBILITY.md        # full experimental protocol
├── CITATION.cff              # citation metadata
├── THIRD_PARTY_NOTICES.md    # upstream attribution
└── LICENSE                   # MIT license
```

## Environment Installation

### Recommended: Conda environment file

```bash
conda env create -f environment.yml
conda activate prototrack
```

### Alternative: original installation script

```bash
conda create -n prototrack python=3.8
conda activate prototrack
bash install.sh
```

The released implementation follows the PyTorch/OSTrack software stack. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for hardware, training, inference, and evaluation details.

## Project Paths Setup

Run:

```bash
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir ./data \
  --save_dir ./output
```

The generated paths can also be edited manually:

```text
lib/train/admin/local.py       # training paths
lib/test/evaluation/local.py   # evaluation paths
```

## Data Preparation

Place the datasets under `./data`.

| Dataset | Existing mirror |
|---|---|
| LasHeR | [Baidu Netdisk](https://pan.baidu.com/s/1SCkEmOoP8LcAhdzXSX_YXQ?pwd=kek1) |
| RGBT210 | [Baidu Netdisk](https://pan.baidu.com/s/1mT8zP-e-ILDu0GfrfaKK-w?pwd=9df2) |
| RGBT234 | [Baidu Netdisk](https://pan.baidu.com/s/1ncwZXmy-ygoI0vOU37HhWA?pwd=ev6s) |

Expected layout:

```text
${PROJECT_ROOT}
└── data
    ├── lasher
    │   ├── trainingset
    │   ├── testingset
    │   ├── trainingsetList.txt
    │   └── testingsetList.txt
    ├── rgbt210
    └── rgbt234
```

Dataset use remains subject to the licenses and terms of the respective benchmark providers.

## Pretrained Weights

Download the RGB/SOT pretrained weights and place them under `pretrained_models/`:

- [Pretrained weights (existing Baidu mirror)](https://pan.baidu.com/s/17W9qq_WFweByg0VN72DjDA?pwd=3p8f)

## Training

The paper configuration uses:

- ViT-Base backbone initialized from an RGB tracker;
- LasHeR training set;
- global batch size 32;
- 20 training epochs;
- 60,000 sampled image pairs per epoch;
- AdamW optimizer;
- learning rate `1e-5` for the ViT backbone and `1e-4` for other parameters;
- learning-rate decay by a factor of 10 after epoch 10;
- search size `256 x 256` and template size `128 x 128`;
- PRM and DCF inserted after Transformer blocks 4, 7, and 10;
- 128 prototypes in the complete model.

Run:

```bash
python tracking/train.py \
  --script prototrack \
  --config vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot \
  --save_dir ./output/vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot \
  --mode multiple \
  --nproc_per_node 2
```

Other configurations are available under `experiments/prototrack/`.

## Evaluation

Place the checkpoint under the configured output directory, or update the checkpoint path in the testing code.

```bash
python tracking/test.py \
  prototrack \
  vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot \
  --dataset_name lasher_test \
  --threads 6 \
  --num_gpus 1
```

Analyze the results:

```bash
python tracking/analysis_results.py \
  --tracker_name prototrack \
  --tracker_param vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot \
  --dataset_name lasher_test
```

For the other benchmarks, replace `--dataset_name lasher_test` with `rgbt210` or `rgbt234` as supported by the evaluation code.

## Dynamic Template Update

During inference, the dynamic template is checked every **25 frames**. When the maximum target-classification score is greater than **0.7**, the RGB and TIR target regions cropped using the current predicted bounding box directly replace the previous dynamic-template pair. No weighted averaging is used.

## Profiling

Use the repository's profiling entry point to measure parameters, MACs, and speed under a fixed hardware and software environment. Report the GPU model, batch size, warm-up iterations, timed iterations, and whether preprocessing is included.

```bash
python tracking/profile_model.py \
  --script prototrack \
  --config vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot
```

The manuscript's speed value was measured on a single RTX 2080 Ti. Cross-paper FPS values should not be compared without checking the corresponding hardware and evaluation protocol.

## Checkpoint and Raw Results

| Model | Backbone | Training pretraining | LasHeR PR | LasHeR NPR | LasHeR SR | Checkpoint | Raw results |
|---|---|---|---:|---:|---:|---|---|
| ProtoTrack | ViT-Base | SOT | 74.2 | 70.2 | 59.3 | [Download](https://pan.baidu.com/s/1bodOHxBQjiSw46Dp3B9_zw?pwd=26tc) | [Download](https://pan.baidu.com/s/1xcwWwX9v3XKRCD-uiysw6Q?pwd=m1hk) |

An internationally accessible mirror and a DOI-backed archival release should be added when available. Until then, this README does not claim that such a mirror or DOI already exists.

## Reproducibility

The full protocol, result provenance, and remaining release checklist are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The reported benchmark numbers correspond to the released checkpoint and raw tracking outputs. The exact random seed of the historical training run was not retained in the manuscript record; therefore, this repository does not retroactively assign an unverified seed to that checkpoint. Future runs should explicitly set and log Python, NumPy, and PyTorch seeds.

## Citation

If this work is useful for your research, please cite the manuscript using the current revised title:

```bibtex
@article{gao2026prototrack,
  title   = {Refine-then-Fuse: Prototype Reconstruction for Robust RGB-T Visual Tracking},
  author  = {Gao, Xiaoting and Zhang, Boquan and Wang, Jingjie and Zeng, Bi and Zhang, Zhongxuan and Hu, Huiting},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Please update the bibliographic record after the article receives final volume, issue, page, and DOI information.

## License

This repository is released under the [MIT License](LICENSE). Portions of the codebase are derived from OSTrack and retain the corresponding upstream MIT attribution; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgments

ProtoTrack is developed upon [OSTrack](https://github.com/botaoye/OSTrack). We thank the OSTrack authors and the broader RGB-T tracking community for their open-source contributions.
