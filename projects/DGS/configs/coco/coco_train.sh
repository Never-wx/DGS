#!/bin/bash
set -e
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_NUMS=$(nvidia-smi -L | wc -l)
    echo "CUDA_VISIBLE_DEVICES is empty, using nvidia-smi to detect GPUs: $GPU_NUMS"
else
    GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Using CUDA_VISIBLE_DEVICES to detect GPUs: $GPU_NUMS"
fi   

SEED=1498185232

# Path settings
# Base weights for the first incremental task (40-49)
BASE_DISTILL_WEIGHTS="./work_dirs/gdino_inc_pretrain_coco/0-39_epoch_12.pth"
CONFIG_DIR="projects/DGS/configs/coco"
WORK_DIR="work_dirs/coco_40+10_4_dgs"

STEPS=("40-49" "50-59" "60-69" "70-79")

for i in "${!STEPS[@]}"; do
  STEP=${STEPS[$i]}
  CFG="${CONFIG_DIR}/gdino_inc_40+10_4_${STEP}_dgs.py"
  
  # Determine load path
  if [ "$i" -eq 0 ]; then
    LOAD_PATH=$BASE_DISTILL_WEIGHTS
  else
    PREV=${STEPS[$((i-1))]}
    LOAD_PATH="${WORK_DIR}/gdino_inc_40+10_4_${PREV}_dgs/epoch_12.pth"
  fi
  
  # Extra options
  CFG_OPTS="randomness.seed=${SEED} load_from=${LOAD_PATH}"

  echo ">>> Step $STEP: loading from $LOAD_PATH"
  bash tools/dist_train.sh "$CFG" "$GPU_NUMS" --amp \
       --work-dir ${WORK_DIR}/gdino_inc_40+10_4_${STEP}_dgs  \
       --cfg-options $CFG_OPTS
done
