#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

python train.py \
  --discribe 0811UNet3D_snack111_511_511_L1 \
  --exp_name 0810UNet3D_snack111_511_511_L1 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[1,1,1]" \
  --TZSConv_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0811UNet3D_snack111_511_511_L3_TMixer_GD1234 \
  --exp_name 0811UNet3D_snack111_511_511_L3_TMixer_GD1234 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --GD_skip "[1,1,1]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train


python train.py \
  --discribe 0810UNet3D_snack111_711_711 \
  --exp_name 0810UNet3D_snack111_711_711 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[7,1,1],[7,1,1],[7,1,1]]" \
  --OffsetKernel "[[7,1,1],[7,1,1],[7,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[1,1,1]" \
  --TZSConv_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

 python test.py \
  --datasetname aircraft \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_mode": ["iou"]}' \
  --load_model /root/autodl-tmp/SGNet/weights/aircraft_multi/UNet3D/0811UNet3D_Snack511511_L3_TMixer_GL_seglen10_2026_08_13_02_26_50/model_best_ap50.pth \
  --model_name UNet3D \
  --seqLen 10 \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_skip '[1,1,1]' \
  --TZSConv_skip '[0,0,0]'\
  --TMixer_skip TMixer_GL \
  --GD_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --Snack_max_offset '[3,3,3]' 

python train.py \
  --discribe 0811UNet3D_TMixer \
  --exp_name 0811UNet3D_TMixer \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[0,0,0]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train


python train.py \
  --discribe 0811re1_UNet3D_snack111_511_511_L3 \
  --exp_name 0811re1_UNet3D_snack111_511_511_L3 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[0,0,0]" \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0811re2_UNet3D_snack111_511_511_L3 \
  --exp_name 0811re2_UNet3D_snack111_511_511_L3 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[0,0,0]" \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0811UNet3D_Snack511511_L3_TMixer_attn_ \
  --exp_name 0811UNet3D_TMixer \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_attn \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0811UNet3D_Snack511511_L3_TMixer_GL \
  --exp_name 0811UNet3D_Snack511511_L3_TMixer_GL \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0811UNet3D_Snack511511_L3_TMixer_STP  \
  --exp_name 0811UNet3D_Snack511511_L3_TMixer_STP  \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_STP  \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0811UNet3D_Snack511511_L3_TMixer_BNReLU  \
  --exp_name 0811UNet3D_Snack511511_L3_TMixer_BNReLU  \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer  \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0811UNet3D_Snack511511_L3_TMixer_GL_test_woconv3d111_biasFalse_add  \
  --exp_name 0811UNet3D_Snack511511_L3_TMixer_GL_test_woconv3d111_biasFalse_add  \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL_test  \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0811UNet3D_TMixer_GL_beforesplitconv111  \
  --exp_name 0811UNet3D_TMixer_GL_beforesplitconv111  \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[0,0,0]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL  \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train



python train.py \
  --discribe 0813UNet3D_Snack511511_L3_TMixer_Lonly \
  --exp_name 0813UNet3D_Snack511511_L3_TMixer_Lonly \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_Lonly \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0813re1UNet3D_Snack511511_L3_TMixer_GL \
  --exp_name 0813re1UNet3D_Snack511511_L3_TMixer_GL \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0813re2UNet3D_Snack511511_L3_TMixer_GL \
  --exp_name 0813re2UNet3D_Snack511511_L3_TMixer_GL \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814UNet3D_Snack511511_L3_TMixer_GL_Dilation1234 \
  --exp_name 0814UNet3D_Snack511511_L3_TMixer_GL_Dilation1234 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL_Dilation1234 \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814UNet3D_Snack511511_L3_TMixer_GL_BNRelu \
  --exp_name 0814UNet3D_Snack511511_L3_TMixer_GL_BNRelu \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL_BNRelu \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814UNet3D_Snack511511_L3_TMixer_GL_groupT_135 \
  --exp_name 0814UNet3D_Snack511511_L3_TMixer_GL_groupT_135 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL_groupT \
  --TMixer_groupT_kernels "[1,3,5]" \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train


  
python train.py \
  --discribe 0814xrsyUNet3D_Snack311311_L3_TMixer_GL \
  --exp_name 0814xrsyUNet3D_Snack311311_L3_TMixer_GL \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[3,1,1],[3,1,1],[3,1,1]]" \
  --OffsetKernel "[[3,1,1],[3,1,1],[3,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814xrsyUNet3D_Snack711711_L3_TMixer_GL \
  --exp_name 0814xrsyUNet3D_Snack711711_L3_TMixer_GL \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[7,1,1],[7,1,1],[7,1,1]]" \
  --OffsetKernel "[[7,1,1],[7,1,1],[7,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

# 相当于linear(T,T)，效果好但是real低
python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L3_TMixer_GL_group2L_add \ 
  --exp_name 0814xrsyUNet3D_Snack511511_L3_TMixer_GL_group2L_add \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL_group2 \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L3_TMixer_GLinear_Ldpconv_add \
  --exp_name 0814xrsyUNet3D_Snack511511_L3_TMixer_GLinear_Ldpconv_add \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GLinear_Ldpconv \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train


python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L1_repeat2 \
  --exp_name 0814xrsyUNet3D_Snack511511_L1_repeat2 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_repeat 2 \
  --Snack_max_offset "[1,1,1]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[0,0,0]" \
  --TMixer_skip TMixer_GLinear_Ldpconv \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L3_repeat2 \
  --exp_name 0814xrsyUNet3D_Snack511511_L3_repeat2 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_repeat 2 \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[0,0,0]" \
  --TMixer_skip TMixer_GLinear_Ldpconv \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L1_repeat2_TMixer \
  --exp_name 0814xrsyUNet3D_Snack511511_L1_repeat2_TMixer \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_repeat 2 \
  --Snack_max_offset "[1,1,1]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L3_repeat2_TMixer \
  --exp_name 0814xrsyUNet3D_Snack511511_L3_repeat2_TMixer \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_repeat 2 \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L3_repeat2_MultiTconv_1357 \
  --exp_name 0814xrsyUNet3D_Snack511511_L3_repeat2_MultiTconv_1357 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_repeat 2 \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip Multi_TConv \
  --TMixer_groupT_kernels "[1,3,5,7]" \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L3_repeat2_MultiTconv_135 \
  --exp_name 0814xrsyUNet3D_Snack511511_L3_repeat2_MultiTconv_135 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_repeat 2 \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip Multi_TConv \
  --TMixer_groupT_kernels "[1,3,5]" \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train

python train.py \
  --discribe 0814xrsyUNet3D_Snack511511_L3_Multi_TConv_LinearTT_135 \
  --exp_name 0814xrsyUNet3D_Snack511511_L3_Multi_TConv_LinearTT_135 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_repeat 1 \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip Multi_TConv_LinearTT \
  --TMixer_groupT_kernels "[1,3,5]" \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train


python train.py \
  --discribe 0815xrsyUNet3D_Snack511511_L3_TMixer_GL_135 \
  --exp_name 0815xrsyUNet3D_Snack511511_L3_TMixer_GL_135 \
  --datasetname aircraft \
  --load_model "" \
  --resume True \
  --model_name UNet3D \
  --seqLen 10 \
  --feat_channels "[16,32,64,128]" \
  --batch_size 4 \
  --lr 1e-3 \
  --Snack_skip "[1,1,1]" \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --UNet3D_skip snack \
  --Snack_repeat 1 \
  --Snack_max_offset "[3,3,3]" \
  --TZSConv_skip "[0,0,0]" \
  --Skip_TMixer "[1,1,1]" \
  --TMixer_skip TMixer_GL_groupT \
  --TMixer_groupT_kernels "[1,3,5]" \
  --TMixer_num_heads 4 \
  --GD_skip "[0,0,0]" \
  --metric '{"inference": true, "run_ap": true, "run_f1": true, "save_json": true, "save_mat": false, "f1_source": "json", "f1_mode": ["iou"], "eval_splits": ["all", "real", "sim"], "best_metric": "ap50"}' \
  --run_test_after_train