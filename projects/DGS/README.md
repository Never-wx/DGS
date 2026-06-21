# DGS: Dynamic Group Subspace for Incremental Object Detection

> Official implementation of **"Boosting Vision-Language Models Towards Cross-Domain Incremental Object Detection"** (CVPR 2026 Highlight).

**[[Paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Boosting_Vision-Language_Models_Towards_Cross-Domain_Incremental_Object_Detection_CVPR_2026_paper.html)** 


## Abstract

Incremental Object Detection (IOD) aims to equip detectors with the ability to handle dynamic environments and emerging object categories, and the rise of vision-language models has substantially advanced this goal. However, existing studies often oversimplify real-world scenarios by assuming the incremental tasks come from a single general domain. To better investigate vision-language models under IOD, it is necessary to explore more generalized scenarios that encompass both novel categories and domains. To this end, we propose **Cross-Domain Incremental Object Detection (CDIOD)**, a new benchmark that assesses the ability to continuously adapt to diverse object detection tasks across domains. CDIOD reveals that existing methods struggle to balance between adaptivity and stability under substantial domain shifts. To tackle this challenge, we propose **Dynamic Group Subspace (DGS)**, a novel framework that dynamically groups tasks by distribution to promote knowledge sharing and prevent task collisions; progressively consolidates adapters to build shared subspaces and control parameter growth; and implements a dynamic training pipeline to maintain a proper stability-adaptivity balance. DGS enables vision-language models to effectively handle task streams of various distribution shifts. Extensive experiments across three benchmarks demonstrate that DGS achieves SOTA performance, highlighting its robustness in diverse incremental learning scenarios.

## Project Structure

```
projects/DGS/
├── dgs/                          # Core Python package
│   ├── detectors/                # Incremental detectors (GroundingDINO_DGS)
│   ├── domain_predictor/         # Dynamic task grouping & routing module
│   ├── heads/                    # Detection heads with distillation losses
│   ├── layers/                   # PEFT modules (LoRA, Adapter, MoE)
│   │   ├── modules/             # LoRA, GroupLoRA, Adapter, RepLinear
│   │   └── moe_layers/          # MoE-LoRA, AdaptiveExpandMoE, Routers
│   ├── hooks/                    # Training hooks (merge, domain predictor, weight transform)
│   ├── losses/                   # Distillation losses (text/query relation, query distillation)
│   ├── dataset/                  # CocoIncDataset (incremental class split)
│   ├── evaluation/               # Incremental & multi-dataset evaluators
│   └── utils/
├── configs/                      # Experiment configurations
│   ├── _base_/                  # Base configs for different benchmarks
│   ├── CDIOD/                   # Cross-Domain Incremental OD configs & scripts
│   ├── coco/                    # COCO incremental detection configs & scripts
│   └── IVLOD/                   # ODinW13 benchmark configs & scripts
```

The outer repository follows the standard [MMDetection](https://github.com/open-mmlab/mmdetection) layout:

```
DGS-main/
├── mmdet/                        # MMDetection core library
├── tools/                        # Training/testing entry points (train.py, test.py, dist_*.sh)
├── configs/                      # Standard mmdet model configs
└── projects/DGS/                 # This project
```

## Installation

This repo is based on [MMDetection 3.3](https://github.com/open-mmlab/mmdetection). Please follow the installation of MMDetection [GETTING_STARTED.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) and make sure you can run it successfully.

### Setup

```bash
# 1. Install MMDetection dependencies
pip install -U openmim
mim install mmengine mmcv mmdet
cd DGS-main
pip install -e .

```

## Checkpoints
Download official checkpoints from [gdino-swin-t](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)

## Data Preparation

All annotations follow COCO JSON format. Each dataset subset contains an `annotations_without_background.json` file within its split directory. Organize datasets under `data/` as follows:

### Conventional IOD (coco)

```
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json              # full 0-79 validation set
    ├── 40+40/                              # 2-step split (40 base + 40 novel)
    │   ├── instances_train2017_0-39.json
    │   ├── instances_train2017_40-79.json
    │   └── instances_val2017_0-39.json
    └── 40+10_4/                            # 5-step split (40 base + 10×4 novel)
        ├── instances_train2017_40-49.json
        ├── instances_train2017_50-59.json
        ├── instances_train2017_60-69.json
        ├── instances_train2017_70-79.json
        ├── instances_val2017_0-49.json
        ├── instances_val2017_0-59.json
        └── instances_val2017_0-69.json
```

### Cross-Domain IOD (CDIOD)

```
data/CDIOD/
├── DIOR/
│   ├── train/
│   │   ├── *.jpg
│   │   ├── annotations_without_background_Task2-{1,2}.json
│   │   └── annotations_without_background_Task4-{1,2,3,4}.json
│   └── valid/
│       ├── *.jpg
│       └── annotations_without_background.json
├── PascalVOC/
│   ├── train/
│   │   ├── *.jpg
│   │   ├── annotations_without_background_Task2-{1,2}.json
│   │   └── annotations_without_background_Task4-{1,2,3,4}.json
│   └── valid/
│       ├── *.jpg
│       └── annotations_without_background.json
└── RUOD/
    ├── train/
    │   ├── *.jpg
    │   └── annotations_without_background_Task2-{1,2}.json
    └── valid/
        ├── *.jpg
        └── annotations_without_background.json
```

### IVLOD (ODinW13)

Each of the 13 sub-datasets follows the same structure:

```
data/ODinW13/
├── AerialMaritimeDrone/
├── Aquarium/
├── CottontailRabbits/
├── EgoHands/
├── NorthAmericaMushroom/
├── Packages/
├── PascalVOC/
├── pistols/
├── pothole/
├── Raccoon/
├── ShellfishOpenImages/
├── thermalDogsAndPeople/
└── VehiclesOpenImages/
```

> **Download**:
> - **CDIOD & COCO split**: Available via [Google Drive](https://drive.google.com/drive/folders/1JdZHhIoB7REwMNiPnzVT_v0XJ-7txaaa?usp=drive_link) / [Quark](https://pan.quark.cn/s/e152a3aaf4cb). Download and place under `data/CDIOD/` and `data/coco/annotations/` respectively.
> - **ODinW13**: Follow the dataset preparation in [IVLOD](https://github.com/JarintotionDin/ZiRaGroundingDINO).
>
> Create symlinks to the actual data locations if needed, e.g., `ln -s /path/to/your/CDIOD data/CDIOD`.

## Usage

Training and testing are driven by pre-built shell scripts located in each config subdirectory. All scripts automatically detect available GPUs and support distributed training.

Two types of training scripts are provided:
- **`run_training_dataset_shuffle.sh`** — For single-config methods (e.g., Sequential Finetuning, ZIRA, and other baselines).
- **`run_training_dataset_shuffle_distn.sh`** — For two-stage methods like DGS that involve distillation (automatically switches between `stage1` and `stage2` configs based on domain predictor grouping results).

### CDIOD Benchmark

```bash
cd DGS-main

# Step 1: Feature extraction (required before DGS training)
python projects/DGS/configs/CDIOD/extract_cdiod_features.py

# Step 2: Train DGS 
# Usage: ./run_training_dataset_shuffle_distn.sh [method] [exp_name] [seed_id] [num_tasks] [start_phase]
bash projects/DGS/configs/CDIOD/run_training_dataset_shuffle_distn.sh dgs dgs 0 10 0

# Train baseline methods (single-config, e.g., seq, zira)
# Usage: ./run_training_dataset_shuffle.sh [method] [exp_name] [seed_id] [num_tasks] [start_phase]
bash projects/DGS/configs/CDIOD/run_training_dataset_shuffle.sh seq seq_exp 0 10 0

# Testing all phases
# Usage: ./run_testing_all.sh [method] [exp_name] [checkpoint] [start_phase]
bash projects/DGS/configs/CDIOD/run_testing_all.sh dgs dgs /path/to/checkpoint.pth 9
```

### COCO Benchmark

```bash
# Step 1: Feature extraction (required before DGS training)
python projects/DGS/configs/coco/extract_coco_features.py

# Step 2: Incremental training (40+10×4 or 40+40 splits)
# Usage: ./run_coco_training.sh [exp_name] [num_tasks] [start_phase]
bash projects/DGS/configs/coco/run_coco_training.sh dgs 5 0
```

### ODinW13 (IVLOD) Benchmark

```bash
# Step 1: Feature extraction (required before DGS training)
python projects/DGS/configs/IVLOD/extract_odinw_features.py
python projects/DGS/configs/IVLOD/extract_coco_ood_features.py  # for OOD evaluation

# Step 2: Two-stage training (Stage 1 + Stage 2 distillation)
# Usage: ./IVLOD_distn.sh [mode] [method] [exp_name] [start_step]
bash projects/DGS/configs/IVLOD/IVLOD_distn.sh fullval dgs dgs 0

# Testing
# Usage: ./IVLOD_test.sh [method] [exp_name] [checkpoint] [start_step] [split]
bash projects/DGS/configs/IVLOD/IVLOD_test.sh dgs dgs /path/to/checkpoint.pth 0 valid

# Zero-shot COCO evaluation
bash projects/DGS/configs/IVLOD/IVLOD_zcoco.sh
```

## Benchmarks

| Benchmark | Tasks | Incremental Setting | Description |
|-----------|-------|---------------------|-------------|
| COCO | 2/5 | Class-incremental | Sequential class splits on COCO-2017 (40+40 or 40+10×4) |
| CDIOD | 5/10 | Cross-domain class-incremental | Sequential category learning across DIOR, PascalVOC, and RUOD with domain shifts |
| ODinW13 | 13 | Task-incremental | Sequential domain/task learning on 13 Objects Detection in the Wild datasets |

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@InProceedings{Wang_2026_CVPR,
    author    = {Wang, Xu and Lin, Zihan and Zhang, Yixin and Wang, Zilei},
    title     = {Boosting Vision-Language Models Towards Cross-Domain Incremental Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {6249-6260}
}
```

## Acknowledgements

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
