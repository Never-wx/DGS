#!/bin/bash
# Unified training script with mode switching for ODinW13 (2-Stage Version)
# Usage: ./run_training.sh [mode] [method] [exp_name]
# Modes: fullval (default), minival

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_NUMS=$(nvidia-smi -L | wc -l)
    echo "CUDA_VISIBLE_DEVICES is empty, using nvidia-smi to detect GPUs: $GPU_NUMS"
else
    GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Using CUDA_VISIBLE_DEVICES to detect GPUs: $GPU_NUMS"
fi 

MODE=${1:-fullval}                  # fullval or minival
METHOD=${2:-adapter}                # default: adapter
EXP_NAME=${3:-adapter_1e-3d128s1}   # exp name
START_STEP=${4:-0}                  # start step, default: 0
SEED=42

# Common datasets configuration
DATASETS=(AerialMaritimeDrone Aquarium CottontailRabbits EgoHands NorthAmericaMushroom
          Packages PascalVOC pistols pothole Raccoon
          ShellfishOpenImages thermalDogsAndPeople VehiclesOpenImages)

# Define work directories and stage-specific config files
work_dirs="work_dirs/ODinW13/${EXP_NAME}_stage2"
cfg_s1="projects/DGS/configs/IVLOD/${EXP_NAME}_stage1.py"
cfg_s2="projects/DGS/configs/IVLOD/${EXP_NAME}_stage2.py"


# Training loop with mode-specific options
for ((i = $START_STEP; i < ${#DATASETS[@]}; i++)); do
    echo "start training from step $i"
    dataset_cur=${DATASETS[i]}
    dataset_seen=$(IFS=,; echo "${DATASETS[*]:0:($i+1)}")
    dataset_all=$(IFS=,; echo "${DATASETS[*]}")

    # Update task_id_mapping before each training stage
    python projects/DGS/configs/adaptive_task_mapping.py \
        --config $cfg_s1 \
        --task_id $i \
        --seen_tasks "$dataset_seen" \
        --num_tasks ${#DATASETS[@]} \
        --work_dirs "$work_dirs"

    # Load weights from the previous step
    if [ $i -eq 0 ]; then
        load_path="./weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
    else
        dataset_pre=${DATASETS[i - 1]}
        load_path="$work_dirs/$dataset_pre/epoch_12.pth"
    fi

    # Base configuration options applicable to both stages
    base_cfg_opts="train_dataloader.dataset.data_root=data/ODinW13/$dataset_cur/ \
                   train_dataloader.dataset.ann_file=train/annotations_without_background.json \
                   train_dataloader.dataset.seen_tasks=$dataset_seen \
                   val_evaluator.evaluation_tasks=$dataset_all \
                   test_evaluator.evaluation_tasks=$dataset_all \
                   load_from=$load_path \
                   randomness.seed=${SEED} \
                   model.num_tasks=${#DATASETS[@]} model.task_id=$i model.seen_tasks=$dataset_seen"
    
    # Decide whether to merge or expand based on the mapping file
    mapping_file="$work_dirs/task_id_mapping.yaml"
    last_value=$(tail -n 1 "$mapping_file" | cut -d':' -f2 | tr -d ' ')
    count=$(grep -oE ": *$last_value" "$mapping_file" | wc -l)
    echo "Task group count for current task: $count"

    if (( count > 1 )); then  # Merge into an existing group
        cfg=$cfg_s2
    else                      # Expand with a new group
        cfg=$cfg_s1
    fi
    
    # Finalize config options with the determined metainfo
    cfg_opts="train_dataloader.dataset.metainfo=$dataset_cur $base_cfg_opts"
    
    # Add mode-specific validation options if needed
    if [ "$MODE" = "minival" ]; then
        # Note: This assumes a 'valid' and 'test' split exists for each ODinW13 dataset
        cfg_opts+=" val_dataloader.dataset.data_root=data/ODinW13/$dataset_cur/ \
                    test_dataloader.dataset.data_root=data/ODinW13/$dataset_cur/ \
                    val_evaluator.ann_file=data/ODinW13/$dataset_cur/valid/annotations_without_background.json \
                    test_evaluator.ann_file=data/ODinW13/$dataset_cur/test/annotations_without_background.json"
    fi
    
    # Execute the training command
    bash tools/dist_train.sh $cfg $GPU_NUMS --amp \
        --work-dir $work_dirs/$dataset_cur \
        --cfg-options $cfg_opts
done

# Post-training operations
if [ "$MODE" = "fullval" ]; then
    # Zero-shot validation on COCO (example)
    # bash tools/dist_test.sh configs/gdino_inc/pretrain/ODinW13/ZCOCO.py \
    #     $work_dirs/${DATASETS[-1]}/epoch_12.pth $GPU_NUMS \
    #     --work-dir $work_dirs/ZCOCO

    # Phase-wise result aggregation
    python projects/DGS/configs/aggregate_results_per_phase.py \
        --phases "[$(printf "'%s'," "${DATASETS[@]}" | sed 's/,$//')]" \
        --root_dir "$work_dirs" \
        --output_file "ODinW13_results_per_phase.json"

elif [ "$MODE" = "minival" ]; then
    # Epoch-wise result aggregation
    python projects/DGS/configs/aggregate_results_per_epoch.py \
        --root_dir "$work_dirs" \
        --output_file "ODinW13_results_per_epoch.json"
fi