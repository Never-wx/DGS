#!/bin/bash
# CDIOD-style training script for COCO
# Usage: ./run_coco_training.sh [exp_name] [num_tasks] [start_phase]

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_NUMS=$(nvidia-smi -L | wc -l)
    echo "CUDA_VISIBLE_DEVICES is empty, using nvidia-smi to detect GPUs: $GPU_NUMS"
else
    GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Using CUDA_VISIBLE_DEVICES to detect GPUs: $GPU_NUMS"
fi   

EXP_NAME=${1:-dgs}
NUM_TASKS=${2:-5}           # 2 or 5 (default: 5)
START_PHASE=${3:-0}         # 0, 1, 2...
SEED=1498185232

# Define datasets and annotation directories based on NUM_TASKS
if [ "$NUM_TASKS" -eq 2 ]; then
    DATASETS=("coco_0-39" "coco_40-79")
    ANN_DIRS=("annotations/40+40" "annotations/40+40")
    VAL_ANN_FILES=("annotations/40+40/instances_val2017_0-39.json" "annotations/instances_val2017.json")
elif [ "$NUM_TASKS" -eq 5 ]; then
    DATASETS=("coco_0-39" "coco_40-49" "coco_50-59" "coco_60-69" "coco_70-79")
    ANN_DIRS=("annotations/40+40" "annotations/40+10_4" "annotations/40+10_4" "annotations/40+10_4" "annotations/40+10_4")
    VAL_ANN_FILES=("annotations/40+40/instances_val2017_0-39.json" "annotations/40+10_4/instances_val2017_0-49.json" "annotations/40+10_4/instances_val2017_0-59.json" "annotations/40+10_4/instances_val2017_0-69.json" "annotations/instances_val2017.json")
else
    echo "Error: Unsupported NUM_TASKS: $NUM_TASKS (Must be 2 or 5)"
    exit 1
fi

work_dirs="work_dirs/coco/${EXP_NAME}_${NUM_TASKS}tasks"
cfg_s1="projects/DGS/configs/coco/dgs_stage1.py"
cfg_s2="projects/DGS/configs/coco/dgs_stage2.py"

echo "=========================================="
echo "COCO Training Configuration:"
echo "  Exp: $EXP_NAME"
echo "  Num Tasks: $NUM_TASKS"
echo "  Start Phase: $START_PHASE"
echo "  Work Dirs: $work_dirs"
echo "=========================================="

# Training loop
for ((i = $START_PHASE; i < ${#DATASETS[@]}; i++)); do
    dataset_cur=${DATASETS[i]}
    ann_dir=${ANN_DIRS[i]}
    val_ann_file=${VAL_ANN_FILES[i]}
    
    mkdir -p "$work_dirs"
    
    # seen_tasks: "coco_0-39,coco_40-49,..."
    dataset_seen=$(IFS=,; echo "${DATASETS[*]:0:($i+1)}")

    # Step range estimation from dataset name (e.g., "coco_40-49" -> start=40, end=49)
    step_range=${dataset_cur#coco_}
    start=${step_range%-*}
    end_val=$(( ${step_range#*-} + 1 ))
    
    # 1. Update task_id_mapping before training
    python projects/DGS/configs/adaptive_task_mapping.py \
        --config $cfg_s1 \
        --task_id $i \
        --seen_tasks "$dataset_seen" \
        --num_tasks $NUM_TASKS \
        --work_dirs "$work_dirs" 
    
    # 2. Determine weights path
    if [ $i -eq 0 ]; then
        # Initial task: load from base GDINO OGC weights
        load_path="./weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
    else
        dataset_pre=${DATASETS[i - 1]}
        load_path="$work_dirs/$dataset_pre/epoch_12.pth"
    fi

    # 3. Construct annotation filenames
    # Pattern: instances_train2017_{start-end}.json
    ann_file="${ann_dir}/instances_train2017_${step_range}.json"

    # 4. Prepare config options
    base_cfg_opts="train_dataloader.dataset.data_root=data/coco/  \
                train_dataloader.dataset.ann_file=$ann_file \
                train_dataloader.dataset.start=$start \
                train_dataloader.dataset.end=$end_val \
                val_dataloader.dataset.ann_file=$val_ann_file \
                val_dataloader.dataset.start=0 \
                val_dataloader.dataset.end=$end_val \
                val_evaluator.ann_file=data/coco/$val_ann_file \
                load_from=$load_path \
                randomness.seed=${SEED} \
                model.num_tasks=$NUM_TASKS model.task_id=$i model.seen_tasks=$dataset_seen \
                model.bbox_head.trunc_class=[$start,$end_val]"

    # 5. Decide Expansion vs Merged (Stage 1 vs Stage 2)
    mapping_file="$work_dirs/task_id_mapping.yaml"
    if [ $i -eq 0 ]; then
        count=1  # Always expansion for the very first task
    else
        last_value=$(tail -n 1 "$mapping_file" | cut -d':' -f2 | tr -d ' ')
        count=$(grep -oE ": *$last_value" "$mapping_file" | wc -l)
    fi
    
    if (( count > 1 )); then    # Merged into existing group
        cfg=$cfg_s2
        # Use full_text for merged stages as they share experts with previous tasks
        setting_opts="train_dataloader.dataset.setting=full_text model.bbox_head.setting=full_text"
    else    # New expansion
        cfg=$cfg_s1
        setting_opts=""
    fi

    echo ">>> Phase $i ($dataset_cur): using $(basename $cfg), loading from $load_path"
    
    # 6. Run training
    bash tools/dist_train.sh $cfg $GPU_NUMS --amp \
        --work-dir $work_dirs/$dataset_cur \
        --cfg-options $base_cfg_opts $setting_opts
done

echo "COCO Training completed for $EXP_NAME"
