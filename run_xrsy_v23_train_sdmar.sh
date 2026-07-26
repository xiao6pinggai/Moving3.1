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

echo "[$(date '+%F %T')] Start: xrsy_v23_10_111_re321"
python train.py \
  --discribe xrsyre1_v23_10_111_re321 \
  --exp_name xrsyre1_v23_10_111_re321 \
  --tmc_topk [3,2,1] \
  --model_name I2PSOD\
  --datasetname sdm_car \

echo "[$(date '+%F %T')] Start: xrsy_v23_10_111_re321"
python train.py \
  --discribe xrsyre2_v23_10_111_re321 \
  --exp_name xrsyre2_v23_10_111_re321 \
  --tmc_topk [3,2,1] \
  --model_name I2PSOD\
  --datasetname sdm_car \

echo "[$(date '+%F %T')] Start: xrsy_v23_10_111_re321"
python train.py \
  --discribe xrsyre3_v23_10_111_re321 \
  --exp_name xrsyre3_v23_10_111_re321 \
  --tmc_topk [3,2,1] \
  --model_name I2PSOD\
  --datasetname sdm_car \

echo "[$(date '+%F %T')] Start: xrsy_v23_10_111_re321"
python train.py \
  --discribe xrsyre4_v23_10_111_re321 \
  --exp_name xrsyre4_v23_10_111_re321 \
  --tmc_topk [3,2,1] \
  --model_name I2PSOD\
  --datasetname sdm_car \
  
echo "[$(date '+%F %T')] Start: xrsy_v23_10_111_re321"
python train.py \
  --discribe xrsyre5_v23_10_111_re321 \
  --exp_name xrsyre5_v23_10_111_re321 \
  --tmc_topk [3,2,1] \
  --model_name I2PSOD\
  --datasetname sdm_car \

echo "[$(date '+%F %T')] Start: xrsy_v23_10_111_re321"
python train.py \
  --discribe xrsyre6_v23_10_111_re321 \
  --exp_name xrsyre6_v23_10_111_re321 \
  --tmc_topk [3,2,1] \
  --model_name I2PSOD\
  --datasetname sdm_car \
echo "[$(date '+%F %T')] All training jobs finished."
