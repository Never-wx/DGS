#!/bin/bash
# Unified training script with mode switching
# Usage: ./run_training.sh [mode] [method] [exp_name]
# Modes: fullval (default), minival

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_NUMS=$(nvidia-smi -L | wc -l)
    echo "CUDA_VISIBLE_DEVICES is empty, using nvidia-smi to detect GPUs: $GPU_NUMS"
else
    GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Using CUDA_VISIBLE_DEVICES to detect GPUs: $GPU_NUMS"
fi 

EXP_NAME=${1:-adapter_1e-3d128s1}  #  exp name
CKPT=${2:-weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth}  #  exp name
TASK_ID_MAPPING_PATH=${3:-none} 

DATASETS=(AerialMaritimeDrone Aquarium CottontailRabbits EgoHands NorthAmericaMushroom
          Packages PascalVOC pistols pothole Raccoon 
          ShellfishOpenImages thermalDogsAndPeople VehiclesOpenImages)

dataset_seen=$(IFS=,; echo "${DATASETS[*]},")
num_tasks=${#DATASETS[@]}
task_id=$((num_tasks - 1))

bash tools/dist_test.sh projects/DGS/configs/IVLOD/ZCOCO.py \
     $CKPT $GPU_NUMS \
    --work-dir work_dirs/ODinW13/$EXP_NAME/ZCOCO \
    --cfg-options \
    model.num_tasks=$num_tasks \
    model.task_id=$task_id \
    model.seen_tasks=$dataset_seen  \
    model.domain_predictor_cfg.task_id_mapping_path=$TASK_ID_MAPPING_PATH