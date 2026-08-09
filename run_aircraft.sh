#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# echo "[$(date '+%F %T')] Start: SGNet_aircraft_t5"
# python train.py \
#   --discribe SGNet_aircraft_t5_woTAA \
#   --exp_name SGNet_aircraft_t5_woTAA \
#   --datasetname aircraft \
#   --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": true, "f1_mode": ["iou"]}' \
#   --load_model "" \
#   --resume True \
#   --model_name I2PSOD \
#   --seqLen 5 \
#   --MFE ""

# python test.py \
#   --discribe SGNet_aircraft \
#   --exp_name SGNet_aircraft \
#   --datasetname aircraft \
#   --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": true, "f1_mode": ["iou"]}' \
#   --load_model weights/aircraft_multi/I2PSOD/SGNet_aircraft_t5_supMode_0_seglen5_weights2026_08_03_20_05_25/model_best_iou_f1_best.pth \
#   --model_name I2PSOD \
#   --seqLen 5 \
#   --MFE "cosv23"

# python train.py \
#   --discribe UNet3D_lr3_3264128256 \
#   --exp_name UNet3D_lr3_3264128256 \
#   --datasetname aircraft \
#   --load_model "" \
#   --resume True \
#   --model_name UNet3D \
#   --seqLen 10 \
#   --feat_channels '[32,64,128,256]' \
#   --batch_size 4 \
#   --lr 1e-3 \
#   --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' 

# python train.py \
#   --discribe UNet3D_lr3_163264128 \
#   --exp_name UNet3D_lr3_163264128 \
#   --datasetname aircraft \
#   --load_model "" \
#   --resume True \
#   --model_name UNet3D \
#   --seqLen 10 \
#   --feat_channels '[16,32,64,128]' \
#   --batch_size 4 \
#   --lr 1e-3 \
#   --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' 

# python train.py \
#   --discribe UNet3D_lr3_3264128256 \
#   --exp_name UNet3D_lr3_3264128256 \
#   --datasetname aircraft \
#   --load_model "" \
#   --resume True \
#   --model_name UNet3D \
#   --seqLen 5 \
#   --feat_channels '[32,64,128,256]' \
#   --batch_size 4 \
#   --lr 1e-3 \
#   --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' 

# python train.py \
#   --discribe UNet3D_lr3_163264128 \
#   --exp_name UNet3D_lr3_163264128 \
#   --datasetname aircraft \
#   --load_model "" \
#   --resume True \
#   --model_name UNet3D \
#   --seqLen 5 \
#   --feat_channels '[16,32,64,128]' \
#   --batch_size 4 \
#   --lr 1e-3 \
#   --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' 

# python train.py \
#   --discribe UNet3D_lr3_163264128 \
#   --exp_name UNet3D_lr3_163264128 \
#   --datasetname aircraft \
#   --load_model "" \
#   --resume True \
#   --model_name UNet3D \
#   --seqLen 20 \
#   --feat_channels '[16,32,64,128]' \
#   --batch_size 4 \
#   --lr 1e-3 \
#   --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

# python train.py \
#   --discribe UNet3D_163264128_snack111555_womask \
#   --exp_name UNet3D_163264128_snack111555_womask \
#   --datasetname aircraft \
#   --load_model "" \
#   --resume True \
#   --model_name UNet3D \
#   --seqLen 10 \
#   --feat_channels '[16,32,64,128]' \
#   --batch_size 4 \
#   --lr 1e-3 \
#   --Snack_skip '[1,1,1]' \
#   --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python train.py \
  --discribe UNet3D_163264128_snack111555_L10_womask \
  --exp_name UNet3D_163264128_snack111555_L10_womask \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[1,1,1]' \
  --Snack_max_offset '[10,10,10]' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python train.py \
  --discribe UNet3D_163264128_snack111555_L5_womask \
  --exp_name UNet3D_163264128_snack111555_L5_womask \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[1,1,1]' \
  --Snack_max_offset '[5,5,5]' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python train.py \
  --discribe UNet3D_tconv111555 \
  --exp_name UNet3D_tconv111555 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[1,1,1]' \
  --Snack_max_offset '[5,5,5]' \
  --UNet3D_skip 'tconv' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python train.py \
  --discribe UNet3D_tdcn111555 \
  --exp_name UNet3D_tdcn111555 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[1,1,1]' \
  --Snack_max_offset '[5,5,5]' \
  --UNet3D_skip 'tdcn' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python train.py \
  --discribe UNet3D_TZSConv111555 \
  --exp_name UNet3D_TZSCConv111555 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[0,0,0]' \
  --TZSConv_skip '[1,1,1]' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python train.py \
  --discribe UNet3D_163264128_snack111555_L3_womask \
  --exp_name UNet3D_163264128_snack111555_L3_womask \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[1,1,1]' \
  --Snack_max_offset '[3,3,3]' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python train.py \
  --discribe UNet3D_163264128_snack111555_wobn_wotanh_woL_womask \
  --exp_name UNet3D_163264128_snack111555_wobn_wotanh_woL_womask \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[1,1,1]' \
  --UNet3D_skip 'snack' \
  --TZSConv_skip '[0,0,0]' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python train.py \
  --discribe UNet3D_163264128_snack111555_wobn_L5_womask \
  --exp_name UNet3D_163264128_snack111555_wobn_L5_womask \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[1,1,1]' \
  --UNet3D_skip 'snack' \
  --TZSConv_skip '[0,0,0]' \
  --Snack_max_offset '[5,5,5]' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'

python test.py \
  --datasetname aircraft \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_mode": ["iou"]}' \
  --load_model /root/autodl-tmp/SGNet/weights/aircraft_multi/UNet3D/UNet3D_163264128_snack111555_L5_womask_seglen10_2026_08_06_03_12_41/model_best_ap50_renamed.pth \
  --model_name UNet3D \
  --seqLen 10 \
  --UNet3D_skip 'snack' \
  --Snack_skip '[1,1,1]' \
  --TZSConv_skip '[0,0,0]'\
  --Snack_max_offset '[5,5,5]' 

python test.py \
  --datasetname aircraft \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_mode": ["iou"]}' \
  --load_model /root/autodl-tmp/SGNet/weights/aircraft_multi/UNet3D/UNet3D_163264128_snack111555_wobn_L5_womask_seglen10_2026_08_09_18_34_18/model_best_ap50.pth \
  --model_name UNet3D \
  --seqLen 10 \
  --UNet3D_skip 'snack' \
  --Snack_skip '[1,1,1]' \
  --TZSConv_skip '[0,0,0]'\
  --Snack_max_offset '[5,5,5]'

python train.py \
  --discribe UNet3D_163264128_snack111555_wobn_L7_womask \
  --exp_name UNet3D_163264128_snack111555_wobn_L7_womask \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels '[16,32,64,128]' \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip '[1,1,1]' \
  --UNet3D_skip 'snack' \
  --TZSConv_skip '[0,0,0]' \
  --Snack_max_offset '[7,7,7]' \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}'
 