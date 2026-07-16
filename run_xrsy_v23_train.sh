#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[$(date '+%F %T')] Start: xrsy_v23_10_111"
python train.py \
  --discribe xrsy_v23_10_111 \
  --exp_name xrsy_v23_10_111 \
  --tmc_topk [1,1,1]

echo "[$(date '+%F %T')] Start: xrsy_v23_10_543"
python train.py \
  --discribe xrsy_v23_10_543 \
  --exp_name xrsy_v23_10_543 \
  --tmc_topk [5,4,3]

echo "[$(date '+%F %T')] Start: xrsy_v23_seqlen20"
python train.py \
  --discribe xrsy_v23_seqlen20 \
  --exp_name xrsy_v23_seqlen20 \
  --seqLen 20

echo "[$(date '+%F %T')] Start: xrsy_v23_seqlen1"
python train.py \
  --discribe xrsy_v23_seqlen1 \
  --exp_name xrsy_v23_seqlen1 \
  --seqLen 1


echo "[$(date '+%F %T')] Start: xrsy_v23_seqlen3"
python train.py \
  --discribe xrsy_v23_seqlen3 \
  --exp_name xrsy_v23_seqlen3 \
  --seqLen 3

echo "[$(date '+%F %T')] Start: xrsy_v23_seqlen5"
python train.py \
  --discribe xrsy_v23_seqlen5 \
  --exp_name xrsy_v23_seqlen5 \
  --seqLen 5

echo "[$(date '+%F %T')] All training jobs finished."
