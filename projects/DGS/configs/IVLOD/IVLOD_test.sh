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
METHOD=${1:-adapter}   # default: adapter
EXP_NAME=${2:-adapter_1e-3d128s1}  #  exp name
CKPT=${3:-weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth}  #  exp name
START_STEP=${4:-0}  #  start step, default: 0
SET=${5:-valid}
# Common datasets configuration
DATASETS=(AerialMaritimeDrone Aquarium CottontailRabbits EgoHands NorthAmericaMushroom
          Packages PascalVOC pistols pothole Raccoon 
          ShellfishOpenImages thermalDogsAndPeople VehiclesOpenImages)

# Training loop with mode-specific options
for ((i = $START_STEP; i < ${#DATASETS[@]}; i++)); do
    echo "start training from steps $START_STEP"
    dataset_cur=${DATASETS[i]}
    dataset_seen=$(IFS=,; echo "${DATASETS[*]:0:($i+1)},")
    dataset_all=$(IFS=,; echo "${DATASETS[*]}")
    num_tasks=${#DATASETS[@]}
    # task_id=$((num_tasks - 1))
    task_id=$START_STEP
    
    # Base command
    cmd="bash tools/dist_test.sh projects/DGS/configs/IVLOD/${EXP_NAME}_stage1.py $CKPT $GPU_NUMS
        --work-dir work_dirs/ODinW13/${EXP_NAME}/$dataset_cur \
        --cfg-options "       # extract with trianing sample
     
    # MOE config
    if [ "$METHOD" = "moe" ]; then
        cmd+=" model.num_tasks=$num_tasks \
              model.task_id=$task_id \
              model.seen_tasks=$dataset_seen"
    fi
    
    eval $cmd

done

bash tools/dist_test.sh projects/DGS/configs/IVLOD/ZCOCO.py \
     $CKPT $GPU_NUMS \
    --work-dir work_dirs/ODinW13/$EXP_NAME/ZCOCO \
    --cfg-options model.task='COCO'



