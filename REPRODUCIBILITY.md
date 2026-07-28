# ProtoTrack Reproducibility Protocol

This document records the experimental settings described in the revised
manuscript and distinguishes verified release information from items that still
need archival completion.

## 1. Code version

For a revision or camera-ready release:

1. create a GitHub release from the exact submitted code;
2. record the release tag and commit SHA in this file;
3. archive the release on Zenodo or an equivalent DOI-providing service;
4. add the DOI to this file, `README.md`, and `CITATION.cff`.

Current status:

- Repository: https://github.com/gxtttt/ProtoTrack
- Release tag for the revised submission: **to be created by the repository owner**
- Archival DOI: **not yet available**

No DOI or international mirror is claimed until it has actually been created.

## 2. Hardware

- Training: two NVIDIA RTX 3090 GPUs.
- Paper speed test: one NVIDIA RTX 2080 Ti GPU.

When reporting new efficiency measurements, also record:

- CUDA and driver versions;
- batch size;
- warm-up iterations;
- timed iterations;
- whether image loading and preprocessing are included;
- peak allocated GPU memory if memory consumption is reported.

## 3. Software environment

Recommended creation command:

```bash
conda env create -f environment.yml
conda activate prototrack
```

The environment file mirrors the repository's original Python 3.8 and
PyTorch/CUDA installation stack. Because GPU drivers and system libraries may
differ, record the output of the following commands with every archived run:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
conda env export --no-builds > environment.lock.yml
```

The generated `environment.lock.yml` should be included in the archival release
for the exact machine used to reproduce the tables.

## 4. Data

- Training dataset: LasHeR training set.
- Evaluation datasets: LasHeR testing set, RGBT210, and RGBT234.
- The trained LasHeR model is evaluated directly on all three benchmarks.

Expected data root:

```text
data/
├── lasher/
│   ├── trainingset/
│   ├── testingset/
│   ├── trainingsetList.txt
│   └── testingsetList.txt
├── rgbt210/
└── rgbt234/
```

The repository does not redistribute benchmark licenses. Users must obtain and
use each dataset under its provider's terms.

## 5. Model and training configuration

- Backbone: ViT-Base.
- Initialization: pretrained weights from an RGB/SOT tracker.
- Search-region size: 256 x 256.
- Template size: 128 x 128.
- Global batch size: 32.
- Epochs: 20.
- Sampled image pairs per epoch: 60,000.
- Optimizer: AdamW.
- Backbone learning rate: 1e-5.
- Other-parameter learning rate: 1e-4.
- Learning-rate decay: factor 0.1 after epoch 10.
- PRM/DCF insertion blocks: 4, 7, and 10.
- Prototype number in the complete model: 128.

Training command:

```bash
python tracking/train.py \
  --script prototrack \
  --config vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot \
  --save_dir ./output/vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot \
  --mode multiple \
  --nproc_per_node 2
```

## 6. Inference and dynamic template update

- Inputs: initial static template, dynamic template, and current search region.
- Update check interval: every 25 frames.
- Confidence threshold: maximum target-classification score greater than 0.7.
- Update content: RGB and TIR target crops obtained from the current predicted
  bounding box.
- Memory management: the new pair directly replaces the previous dynamic pair.
- Weighted averaging: not used.

## 7. Evaluation commands

LasHeR:

```bash
python tracking/test.py \
  prototrack \
  vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot \
  --dataset_name lasher_test \
  --threads 6 \
  --num_gpus 1

python tracking/analysis_results.py \
  --tracker_name prototrack \
  --tracker_param vitb_256_prototrack_mutitle_template_32x1_1e4_lasher_20ep_sot \
  --dataset_name lasher_test
```

Replace the dataset name with `rgbt210` or `rgbt234` for the corresponding
benchmark, subject to the evaluation script's supported names.

## 8. Result provenance

- Competitor accuracy values in the main comparison table are quoted from their
  corresponding publications.
- Efficiency values for TBSI, ViPT, BAT, and GMMT are quoted from BTMTrack,
  where those methods were evaluated on an RTX 2080 Ti.
- AINet, MambaVT, FMTrack, and ProtoTrack were locally evaluated on the same GPU
  model for the revised efficiency comparison.

Reported ProtoTrack values:

| Benchmark | Metrics (%) |
|---|---|
| LasHeR | PR 74.2, NPR 70.2, SR 59.3 |
| RGBT210 | MPR 88.3, MSR 64.3 |
| RGBT234 | MPR 88.8, MSR 65.4 |

Paper efficiency record:

- Parameters: 153.950 M.
- MACs: 89.625 G.
- FPS: 36.292 on one RTX 2080 Ti.

## 9. Randomness and seeds

The exact random seed used for the historical checkpoint that produced the
reported numbers was not retained in the manuscript record. A seed must not be
invented retrospectively and attributed to that checkpoint.

For every future training run, set and log Python, NumPy, and PyTorch seeds at
the start of the training process. A typical initialization is:

```python
import os
import random

import numpy as np
import torch

seed = int(os.environ.get("PROTOTRACK_SEED", "42"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

The displayed value `42` is a recommended default for future runs, not a claim
about the historical released checkpoint. Record the actual seed in the run
configuration, training log, checkpoint metadata, and release notes.

## 10. Release-completion checklist

Before responding that the reproducibility package is complete, verify that the
repository contains:

- [x] installation instructions;
- [x] an environment file;
- [x] exact training and evaluation commands;
- [x] pretrained-weight instructions;
- [x] checkpoint and raw-result links;
- [x] training hyperparameters;
- [x] dynamic-template update details;
- [x] software license and upstream attribution;
- [x] citation metadata;
- [ ] an internationally accessible checkpoint/raw-result mirror;
- [ ] a tagged archival release and DOI;
- [ ] the exact historical seed, if recoverable from logs or metadata;
- [ ] scripts or commands reproducing every manuscript table and figure;
- [ ] peak GPU-memory reporting under the paper's profiling protocol.
