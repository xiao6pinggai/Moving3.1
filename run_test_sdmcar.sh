#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
# python test.py \
#   --load_model weights/sdm_car_multi/Net1/v23_net1_sep10sample10_unet888_supMode_0_seglen10_weights2026_07_08_15_05_53/model_best_dis_f1_best.pth \
#   --model_name Net1 \
#   --datasetname sdm_car \

# echo "V0 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/Net1_SpDetHead/xrsy_Net1_SpDetHead_supMode_0_seglen10_weights2026_07_21_18_09_37/model_best_dis_f1_best.pth \
  --model_name Net1_SpDetHead \
  --datasetname sdm_car \

echo "V1 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/xrsy_10_rebaseline_supMode_0_seglen10_weights2026_07_12_15_56_27/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --MFE "" \
  --datasetname sdm_car 

echo "V2 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/Net1_Net2/xrsy_reNet1_Net2_supMode_0_seglen10_weights2026_07_30_12_33_26/model_best_dis_f1_best.pth \
  --model_name Net1_Net2 \
  --datasetname sdm_car 

echo "V3 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/xrsy_CA_supMode_0_seglen10_weights2026_07_30_01_08_31/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --MFE CA \
  --datasetname sdm_car 

echo "V4 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/high_xrsyre1_v23_10_111_re321_supMode_0_seglen10_weights2026_07_23_09_38_29/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --datasetname sdm_car 

echo "V5 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/xrsy_th5_supMode_0_seglen10_weights2026_07_30_20_47_44/model_best_dis_f1_best.pth \
  --model_name I2PSOD_test \
  --datasetname sdm_car \
  --thresh 5

echo "th5 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/high_xrsyre1_v23_10_111_re321_supMode_0_seglen10_weights2026_07_23_09_38_29/model_best_dis_f1_best.pth \
  --model_name I2PSOD_test \
  --datasetname sdm_car \
  --thresh 3

echo "th3 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/xrsy_th1_supMode_0_seglen10_weights2026_07_30_02_36_29/model_best_dis_f1_best.pth \
  --model_name I2PSOD_test \
  --datasetname sdm_car \
  --thresh 1

echo "th1 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/xrsy_th5_supMode_0_seglen10_weights2026_07_30_20_47_44/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --datasetname sdm_car \
  --thresh 5

echo "th5 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/high_xrsyre1_v23_10_111_re321_supMode_0_seglen10_weights2026_07_23_09_38_29/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --datasetname sdm_car \
  --thresh 3

echo "th3 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/xrsy_th1_supMode_0_seglen10_weights2026_07_30_02_36_29/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --datasetname sdm_car \
  --thresh 1

echo "th1 test ok!"

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/high_xrsyre1_v23_10_111_re321_supMode_0_seglen10_weights2026_07_23_09_38_29/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --tmc_topk [1,1,1] \
  --datasetname sdm_car 

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/NewSelectedxrsy_v23_10_111_re543_supMode_0_seglen10_weights2026_07_22_03_21_21/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --tmc_topk [5,4,3] \
  --datasetname sdm_car 

python test.py \
  --load_model weights/sdm_car_multi/I2PSOD/high_xrsyre1_v23_10_111_re321_supMode_0_seglen10_weights2026_07_23_09_38_29/model_best_dis_f1_best.pth \
  --model_name I2PSOD \
  --datasetname sdm_car 