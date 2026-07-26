#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# echo "[$(date '+%F %T')] Start: xrsy_v23_seqlen20"
# python train.py \
#   --discribe xrsy_v23_seqlen20 \
#   --exp_name xrsy_v23_seqlen20 \
#   --seqLen 20 \
#   --datasetname sdm_car \



echo "[$(date '+%F %T')] Start: xrsy_v23_th2"
python train.py \
  --discribe xrsy_v23_th2 \
  --exp_name xrsy_v23_th2 \
  --thresh 2 \

echo "[$(date '+%F %T')] Start: xrsy_v23_th4"
python train.py \
  --discribe xrsy_v23_th4 \
  --exp_name xrsy_v23_th4 \
  --thresh 4 \




echo "[$(date '+%F %T')] All training jobs finished."
