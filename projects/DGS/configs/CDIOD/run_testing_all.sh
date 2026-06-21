#!/bin/bash
# Unified training script with mode switching
# Usage: ./run_training.sh [method] [exp_name]

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_NUMS=$(nvidia-smi -L | wc -l)
    echo "CUDA_VISIBLE_DEVICES is empty, using nvidia-smi to detect GPUs: $GPU_NUMS"
else
    GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Using CUDA_VISIBLE_DEVICES to detect GPUs: $GPU_NUMS"
fi 

METHOD=${1:-moe}   # default: adapter
EXP_NAME=${2:-moe_expand_exp2k2d64}  #  exp name
CKPT=${3:-work_dirs/CDIOD/minival_moe_expand_exp2k2d64/epoch_10_RUOD_init.pth}  #  exp name
START_PHASE=${4:-9} 
TASK_ID_MAPPING_PATH=${5:-none} 

# Common datasets configuration
# DATASETS=(DIOR_Task2-1 DIOR_Task2-2 PascalVOC_Task2-1 PascalVOC_Task2-2 RUOD)
DATASETS=(DIOR_Task4-1 DIOR_Task4-2 DIOR_Task4-3 DIOR_Task4-4 PascalVOC_Task4-1 PascalVOC_Task4-2 PascalVOC_Task4-3 PascalVOC_Task4-4 RUOD_Task2-1 RUOD_Task2-2)
dataset_all=$(IFS=,; echo "${DATASETS[*]}")
dataset_seen=$(IFS=,; echo "${DATASETS[*]:0:($START_PHASE+1)},")
dataset_cur=${DATASETS[$START_PHASE]}
num_tasks=${#DATASETS[@]}
task_id=$START_PHASE
# model.seen_tasks=$dataset_all \
# num_tasks=${#DATASETS[@]} \
# task_id=$((${#DATASETS[@]} - 1))

if [ "$MODE" = "fullval" ]; then
    work_dirs="work_dirs/CDIOD/$EXP_NAME/$dataset_cur"
    cfg="projects/DGS/configs/CDIOD/${EXP_NAME}.py"
    cfg_opts="val_evaluator.evaluation_tasks=$dataset_all \
            test_evaluator.evaluation_tasks=$dataset_all"
    
    if [ "$METHOD" = "moe" ]; then
        cfg_opts=" $cfg_opts \
                    model.num_tasks=$num_tasks \
                    model.task_id=$task_id \
                    model.seen_tasks=$dataset_seen"
    fi
    
    if [ "$TASK_ID_MAPPING_PATH" != "none" ]; then
        cfg_opts="$cfg_opts \
                  model.domain_predictor_cfg.task_id_mapping_path=$TASK_ID_MAPPING_PATH"
    fi


bash tools/dist_test.sh $cfg $CKPT $GPU_NUMS \
    --cfg-options $cfg_opts \
    --work-dir $work_dirs
