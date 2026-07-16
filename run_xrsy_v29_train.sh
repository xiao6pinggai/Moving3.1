#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[$(date '+%F %T')] Start: xrsy_v29_10_th1"
python train.py \
  --discribe xrsy_v29_10_th1 \
  --exp_name xrsy_v29_10_th1 \
  --thresh 1

echo "[$(date '+%F %T')] Start: xrsy_v29_10_th5"
python train.py \
  --discribe xrsy_v29_10_th5 \
  --exp_name xrsy_v29_10_th5 \
  --thresh 5

echo "[$(date '+%F %T')] Start: xrsy_v29_10_th7"
python train.py \
  --discribe xrsy_v29_10_th7 \
  --exp_name xrsy_v29_10_th7 \
  --thresh 7

echo "[$(date '+%F %T')] Start: xrsy_v29_10_th3"
python train.py \
  --discribe v29_10_543 \
  --exp_name v29_10_th3 \
  --datasetname sdm_car \
  --thresh 3

echo "[$(date '+%F %T')] Start: xrsy_v29_15_th3"
python train.py \
  --discribe xrsy_rev29_15_th3 \
  --exp_name xrsy_rev29_15_th3

echo "[$(date '+%F %T')] All training jobs finished."
