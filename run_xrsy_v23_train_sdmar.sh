#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# echo "[$(date '+%F %T')] Start: xrsy_v23_10_111_re321"
# python train.py \
#   --discribe xrsy_v23_10_111_re321 \
#   --exp_name xrsy_v23_10_111_re321 \
#   --tmc_topk [3,2,1] \
#   --datasetname sdm_car \

python train.py \
  --discribe xrsy_10_rebaseline \
  --exp_name xrsy_10_rebaseline \
  --datasetname sdm_car \
  --MFE_skip [0,0,0] \
  --MFE "" \


echo "[$(date '+%F %T')] All training jobs finished."
