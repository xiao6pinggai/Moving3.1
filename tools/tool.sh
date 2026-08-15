# --point_radius 6          # 采样点大小
# --line_thickness 2        # 采样点连线粗细
# --batch_patches 4         # 多 patch 批量前向，加快速度
# --max_videos 3            # 只看前 3 个视频
# --max_frames 20           # 最多保存 20 张
# --videos videoA videoB    # 只看指定视频
# --draw_labels             # 画 track id 和 t 偏移标签

  
# 每隔 30 帧保存一张
# 按视频名建文件夹
# 同一轨迹的 GT 投影框用同一种颜色
# 中心帧 anchor 框用白色粗框
# skip1 落入对应帧 GT 框内：绿色圆点
# skip2 落入对应帧 GT 框内：黄色三角
# skip3 落入对应帧 GT 框内：蓝色方块
# 未落入对应帧 GT 框内：红色，形状仍区分 skip 层
python3 tools/visualize_snake_sampling.py \
  --datasetname aircraft \
  --load_model /root/autodl-tmp/SGNet/weights/aircraft_multi/UNet3D/UNet3D_163264128_snack111555_wobn_L5_womask_seglen10_2026_08_09_18_34_18/model_best_ap50.pth \
  --model_name UNet3D \
  --seqLen 10 \
  --UNet3D_skip snack \
  --Snack_skip '[1,1,1]' \
  --Snack_max_offset '[5,5,5]' \
  --TZSConv_skip '[0,0,0]' \
  --frame_interval 30 \
  --point_radius 1 \
  --batch_patches 2 \
  --max_frames 20 \
  --line_thickness 1 \
  --max_videos 20

python3 tools/visualize_snake_sampling.py \
  --datasetname aircraft \
  --load_model /root/autodl-tmp/SGNet/weights/aircraft_multi/UNet3D/UNet3D_vsnack111555_L5_wbntanh_vscope_seglen10_2026_08_10_16_09_11/model_best_ap50.pth \
  --model_name UNet3D \
  --seqLen 10 \
  --UNet3D_skip v_snack \
  --Snack_skip '[1,1,1]' \
  --Snack_max_offset '[5,5,5]' \
  --TZSConv_skip '[0,0,0]' \
  --frame_interval 20 \
  --point_radius 0 \
  --line_thickness 0 \
  --batch_patches 2 \
  --max_videos 10 \
  --hit_mode corresponding \
  --show_neighbor_boxes \
  --videos cityB_1159_1_4_5_10 cityB_1159_2_2_3_5 cloudB_852_2_3_3_10 movingA_852_2_1_3_10 seaB_852_1_1_3_5 realobject_1228_2_3_3_10 \
  --layers '["skip1"]' \
  --anchor_point_percent 25 \
  --point_radius 0 \
  --line_thickness 0 

python3 tools/visualize_snake_sampling.py \
  --datasetname aircraft \
  --load_model /root/autodl-tmp/SGNet/weights/aircraft_multi/UNet3D/0811re2_UNet3D_snack111_511_511_L3_seglen10_2026_08_12_03_27_00/model_best_ap50.pth \
  --model_name UNet3D \
  --seqLen 10 \
  --Snack_skip '[1,1,1]' \
  --Snack_max_offset '[3,3,3]' \
  --TZSConv_skip '[0,0,0]' \
  --frame_interval 20 \
  --point_radius 0 \
  --line_thickness 0 \
  --batch_patches 2 \
  --max_videos 10 \
  --hit_mode corresponding \
  --show_neighbor_boxes \
  --videos cityB_1159_1_4_5_10 cityB_1159_2_2_3_5 cloudB_852_2_3_3_10 movingA_852_2_1_3_10 seaB_852_1_1_3_5 realobject_1228_2_3_3_10 \
  --layers '["skip1"]' \
  --anchor_point_percent 50 \
  --point_radius 0 \
  --line_thickness 0  \
  --vis_features \
  --vis_sample \
  --VSnack_residual 0 \
  --UNet3D_skip snack \
  --Snack_repeat 1 \
  --Snack_skip '[1,1,1]' \
  --TZSConv_skip '[0,0,0]' \
  --TKernel "[[5,1,1],[5,1,1],[5,1,1]]" \
  --OffsetKernel "[[5,1,1],[5,1,1],[5,1,1]]" 
