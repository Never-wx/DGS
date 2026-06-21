#!/bin/bash
# Unified training script with mode switching
# Usage: ./run_training_shuffle.sh [method] [exp_name] [start_phase] [seed_id] [num_steps]
# seed_id: 1, 2, or 3 (default: 1)
# num_steps: 5 or 10 (default: 5)

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_NUMS=$(nvidia-smi -L | wc -l)
    echo "CUDA_VISIBLE_DEVICES is empty, using nvidia-smi to detect GPUs: $GPU_NUMS"
else
    GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Using CUDA_VISIBLE_DEVICES to detect GPUs: $GPU_NUMS"
fi   

METHOD=${1:-adapter}      # default: adapter
EXP_NAME=${2:-adapter_1e-3d128s1}  # exp name
SEED_ID=${3:-0}           # seed id: 0, 1, 2, or 3 (default: 0)
NUM_TASKS=${4:-10}         # number of steps: 5 or 10 (default: 10)
START_PHASE=${5:-0}        # start step, default: 0

# # Seed mapping
SEEDS=(1498185232 42 998234)
SEED=${SEEDS[$SEED_ID]}

# Dataset configuration based on NUM_TASKS
if [ "$NUM_TASKS" = "5" ]; then
    DATASETS=(DIOR_Task2-1 DIOR_Task2-2 PascalVOC_Task2-1 PascalVOC_Task2-2 RUOD)
elif [ "$NUM_TASKS" = "10" ]; then
    DATASETS=(DIOR_Task4-1 DIOR_Task4-2 DIOR_Task4-3 DIOR_Task4-4 PascalVOC_Task4-1 PascalVOC_Task4-2 PascalVOC_Task4-3 PascalVOC_Task4-4 RUOD_Task2-1 RUOD_Task2-2)
else
    echo "Error: NUM_TASKS must be 5 or 10"
    exit 1
fi

# Shuffle datasets using Python script
DATASETS=($(python3 "$(dirname "$0")/run_shuffle_tasks.py" $SEED "${DATASETS[@]}" -v))

work_dirs="work_dirs/CDIOD/${EXP_NAME}_seed${SEED_ID}_${NUM_TASKS}phases"
cfg="projects/DGS/configs/CDIOD/${EXP_NAME}.py"

# Training loop with mode-specific options
for ((i = $START_PHASE; i < ${#DATASETS[@]}; i++)); do
    echo "start training from phase $START_PHASE"
    dataset_cur=${DATASETS[i]}
    dataset_seen=$(IFS=,; echo "${DATASETS[*]:0:($i+1)},")
    dataset_all=$(IFS=,; echo "${DATASETS[*]}")

    # Generate annotation filename based on dataset name
    if [[ $dataset_cur == *"_Task"* ]]; then
        annotation="annotations_without_background_${dataset_cur##*_}.json"     # suffix of the dataset name
        dataset_path_cur=${dataset_cur%%_*}                                     # prefix of the dataset name
    else
        annotation="annotations_without_background.json"
        dataset_path_cur=$dataset_cur
    fi
    
    # load last step weights
    if [ $i -eq 0 ]; then
        load_path="./weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
    else
        dataset_pre=${DATASETS[i - 1]}
        load_path="$work_dirs/$dataset_pre/epoch_12.pth"
    fi

    cfg_opts="train_dataloader.dataset.metainfo=$dataset_cur \
            train_dataloader.dataset.data_root=data/CDIOD/$dataset_path_cur/  \
            train_dataloader.dataset.ann_file=train/$annotation \
            train_dataloader.dataset.seen_tasks=$dataset_seen \
            val_evaluator.evaluation_tasks=$dataset_all \
            test_evaluator.evaluation_tasks=$dataset_all \
            load_from=$load_path \
            randomness.seed=${SEED} "

    # MOE config
    if [ "$METHOD" = "moe" ]; then
        cfg_opts="$cfg_opts \
                model.num_tasks=${#DATASETS[@]} model.task_id=$i model.seen_tasks=$dataset_seen"
    fi

    bash tools/dist_train.sh $cfg $GPU_NUMS --amp \
        --work-dir $work_dirs/$dataset_cur \
        --cfg-options $cfg_opts

done

# Phase-wise aggregation
python projects/DGS/configs/aggregate_results_per_phase.py \
    --phases "[$(printf "'%s'," "${DATASETS[@]}" | sed 's/,$//')]" \
    --root_dir "$work_dirs" \
    --output_file "CDIOD_results_per_phase.json"
