#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# echo "[$(date '+%F %T')] Start: xrsy_v23_10_111_re111"
# python train.py \
#   --discribe xrsy_v23_10_111_re111 \
#   --exp_name xrsy_v23_10_111_re111 \
#   --tmc_topk [1,1,1] \
#   --model_name I2PSOD\
#   --datasetname sdm_car \

echo "[$(date '+%F %T')] Start: xrsy_Net1_Net2"
python train.py \
  --discribe xrsy_Net1_Net2 \
  --exp_name xrsy_Net1_Net2 \
  --model_name Net1_Net2\
  --datasetname sdm_car \

echo "[$(date '+%F %T')] Start: xrsy_CA"
python train.py \
  --discribe xrsy_CA \
  --exp_name xrsy_CA \
  --model_name I2PSOD \
  --datasetname sdm_car \
  --MFE CA \

echo "[$(date '+%F %T')] Start: xrsy_th1"
python train.py \
  --discribe xrsy_th1 \
  --exp_name xrsy_th1 \
  --model_name I2PSOD \
  --datasetname sdm_car \
  --thresh 1 \

echo "[$(date '+%F %T')] Start: xrsy_th5"
python train.py \
  --discribe xrsy_th5 \
  --exp_name xrsy_th5 \
  --model_name I2PSOD \
  --datasetname sdm_car \
  --thresh 5 \
echo "[$(date '+%F %T')] All training jobs finished."
