#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# echo "[$(date '+%F %T')] Start: xrsy_v23_seqlen20"
# python train.py \
#   --discribe xrsy_v23_seqlen20 \
#   --exp_name xrsy_v23_seqlen20 \
#   --seqLen 20 \
#   --datasetname sdm_car \

echo "[$(date '+%F %T')] Start: xrsy_Net1_Net2"
python train.py \
  --discribe xrsy_Net1_Net2 \
  --exp_name xrsy_Net1_Net2 \
  --model_name Net1_Net2

echo "[$(date '+%F %T')] Start: xrsy_Net1_Net2"
python train.py \
  --discribe xrsy_Net1_Net2 \
  --exp_name xrsy_Net1_Net2 \
  --model_name Net1_Net2


echo "[$(date '+%F %T')] Start: xrsy_Net1_Net2"
python train.py \
  --discribe xrsy_Net1_Net2 \
  --exp_name xrsy_Net1_Net2 \
  --model_name Net1_Net2




echo "[$(date '+%F %T')] All training jobs finished."
