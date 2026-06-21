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

SEEDS=(1498185232 42 998234)

if [[ "$SEED_ID" -ge 0 && "$SEED_ID" -le 2 ]]; then
    SEED=${SEEDS[$SEED_ID]}
else
    echo "Error: SEED_ID must be 0, 1, or 2"
    exit 1
fi

# Dataset configuration
if [ "$SEED_ID" = "0" ] && [ "$NUM_TASKS" = "5" ]; then
    DATASETS=(DIOR_Task2-1 DIOR_Task2-2 PascalVOC_Task2-1 PascalVOC_Task2-2 RUOD)
elif [ "$SEED_ID" = "0" ] && [ "$NUM_TASKS" = "10" ]; then
    DATASETS=(DIOR_Task4-1 DIOR_Task4-2 DIOR_Task4-3 DIOR_Task4-4 PascalVOC_Task4-1 PascalVOC_Task4-2 PascalVOC_Task4-3 PascalVOC_Task4-4 RUOD_Task2-1 RUOD_Task2-2)
elif [ "$SEED_ID" = "1" ] && [ "$NUM_TASKS" = "5" ]; then
    DATASETS=(PascalVOC_Task2-1 PascalVOC_Task2-2 RUOD DIOR_Task2-1 DIOR_Task2-2)
elif [ "$SEED_ID" = "1" ] && [ "$NUM_TASKS" = "10" ]; then
    DATASETS=(PascalVOC_Task4-1 PascalVOC_Task4-2 PascalVOC_Task4-3 PascalVOC_Task4-4 RUOD_Task2-1 RUOD_Task2-2 DIOR_Task4-1 DIOR_Task4-2 DIOR_Task4-3 DIOR_Task4-4)
elif [ "$SEED_ID" = "2" ] && [ "$NUM_TASKS" = "5" ]; then
    DATASETS=(RUOD PascalVOC_Task2-1 PascalVOC_Task2-2 DIOR_Task2-1 DIOR_Task2-2)
elif [ "$SEED_ID" = "2" ] && [ "$NUM_TASKS" = "10" ]; then
    DATASETS=(RUOD_Task2-1 RUOD_Task2-2 PascalVOC_Task4-1 PascalVOC_Task4-2 PascalVOC_Task4-3 PascalVOC_Task4-4 DIOR_Task4-1 DIOR_Task4-2 DIOR_Task4-3 DIOR_Task4-4)
else
    echo "Error: Invalid combination of SEED_ID ($SEED_ID) and NUM_TASKS ($NUM_TASKS)"
    echo "SEED_ID must be 0, 1, or 2, and NUM_TASKS must be 5 or 10"
    exit 1
fi

echo "=========================================="
echo "Dataset Configuration:"
echo "  Method: $METHOD"
echo "  Exp: $EXP_NAME"
echo "  Seed ID: $SEED_ID (Value: $SEED)"
echo "  Shuffled Order:"
for idx in "${!DATASETS[@]}"; do
    echo "    Phase $idx: ${DATASETS[$idx]}"
done
echo "  Start Phase: $START_PHASE"
echo "=========================================="

work_dirs="work_dirs/CDIOD/${EXP_NAME}_seed${SEED_ID}_${NUM_TASKS}phases"
cfg_s1="projects/DGS/configs/CDIOD/${EXP_NAME}_stage1.py"
cfg_s2="projects/DGS/configs/CDIOD/${EXP_NAME}_stage2.py"

# Training loop with mode-specific options
for ((i = $START_PHASE; i < ${#DATASETS[@]}; i++)); do
    dataset_cur=${DATASETS[i]}
    dataset_seen=$(IFS=,; echo "${DATASETS[*]:0:($i+1)}")
    dataset_all=$(IFS=,; echo "${DATASETS[*]}")

    # update task_id_mapping before training
    python projects/DGS/configs/adaptive_task_mapping.py \
        --config $cfg_s1 \
        --task_id $i \
        --seen_tasks $dataset_seen \
        --num_tasks ${#DATASETS[@]} \
        --work_dirs $work_dirs 
    
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

    base_cfg_opts="train_dataloader.dataset.data_root=data/CDIOD/$dataset_path_cur/  \
                train_dataloader.dataset.ann_file=train/$annotation \
                train_dataloader.dataset.seen_tasks=$dataset_seen \
                val_evaluator.evaluation_tasks=$dataset_all \
                test_evaluator.evaluation_tasks=$dataset_all \
                load_from=$load_path \
                randomness.seed=${SEED} \
                model.num_tasks=${#DATASETS[@]} model.task_id=$i model.seen_tasks=$dataset_seen"

    # Updated mapping file path
    mapping_file="$work_dirs/task_id_mapping.yaml"
    last_value=$(tail -n 1 "$mapping_file" | cut -d':' -f2 | tr -d ' ')
    count=$(grep -oE ": *$last_value" "$mapping_file" | wc -l)
    echo $count
    
    if (( count > 1 )); then    # merge into existing group  
        metainfo=$dataset_seen
        cfg=$cfg_s2
    else    # expansion
        metainfo=$dataset_cur
        cfg=$cfg_s1
    fi

    cfg_opts="train_dataloader.dataset.metainfo=$metainfo $base_cfg_opts"

    bash tools/dist_train.sh $cfg $GPU_NUMS --amp \
        --work-dir $work_dirs/$dataset_cur \
        --cfg-options $cfg_opts
done

# Phase-wise aggregation
python projects/DGS/configs/aggregate_results_per_phase.py \
    --phases "[$(printf "'%s'," "${DATASETS[@]}" | sed 's/,$//')]" \
    --root_dir "$work_dirs"  \
    --output_file "CDIOD_results_per_phase.json"


