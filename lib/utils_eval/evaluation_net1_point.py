import numpy as np
import os
import sys

def eval_net1_points(results_dir_root, data_dir=None, data_name=None, xmlname='xml_det', save_name='net1_point.txt'):
    """
    专门测评 Net1 生成的点云质量
    新增指标：Cov (Coverage) - 点云在GT框内的覆盖面积比例
    """
    
    # 1. 参数标准化处理
    if isinstance(results_dir_root, str):
        results_dir_root = [results_dir_root]
        
    if data_name is None:
        dataName = [2, 3, 5, 6, 8, 9, 10]
    else:
        dataName = data_name

    if data_dir is None:
        ANN_PATH0 = '/media/xc/BA61C62ABCE29FF2/xc/dataset/RsCarData/images/test1024/'
    else:
        ANN_PATH0 = data_dir
        if not ANN_PATH0.endswith('/'):
            ANN_PATH0 += '/'

    # 2. 初始化评估工具
    from lib.utils1.utils_eval import eval_metric
    det_metric = eval_metric(dis_th=5, iou_th=0.0001, eval_mode='iou') 

    print("================ Evaluation Start: Net1 Point Cloud ================")
    
    for res_dir in results_dir_root:
        print(f"Processing: {res_dir}")
        os.makedirs(res_dir, exist_ok=True)
        save_path = os.path.join(res_dir, save_name)
        fid = open(save_path, 'w+')
        fid.write(f"Results Directory: {res_dir}\n")
        # [MODIFIED] 增加 Cov 列
        fid.write("DataName\tRecall\tPrecision\tCov\tF1\tPd\tFa_1\tFa_2\n")
        
        all_results_list = []
        
        for folder_id in dataName:
            folder_str = '%03d' % folder_id
            
            # 4.1 构造路径
            ann_path = os.path.join(ANN_PATH0, folder_str, xmlname)
            if not os.path.exists(ann_path):
                print(f"Skip {folder_str}: GT path not found")
                continue
                
            txt_path = os.path.join(res_dir, folder_str, f'{folder_str}.txt')
            if not os.path.exists(txt_path):
                print(f"Skip {folder_str}: Prediction txt not found")
                continue
            
            # 4.2 读取点云数据
            try:
                points_data = np.loadtxt(txt_path, delimiter=',', skiprows=1, dtype=np.int32)
                if points_data.ndim == 1 and points_data.size > 0:
                    points_data = points_data[None, :] 
            except Exception as e:
                print(f"Error reading {txt_path}: {e}")
                points_data = np.empty((0, 3))

            # 将点云转为字典索引
            preds_by_frame = {}
            if points_data.size > 0:
                for row in points_data:
                    p_fid, y, x = row 
                    if p_fid not in preds_by_frame:
                        preds_by_frame[p_fid] = []
                    # 存入列表，用于转bbox
                    preds_by_frame[p_fid].append([float(x), float(y), float(x)+1, float(y)+1, 1.0])
            
            # 4.3 逐帧评估
            det_metric.reset()
            
            # [ADDED] 覆盖率统计变量
            total_coverage_sum = 0.0
            total_gt_count_cov = 0
            
            xml_files = sorted([f for f in os.listdir(ann_path) if f.endswith('.xml')])
            num_images = len(xml_files)
            
            # 假设图片尺寸为 1024x1024，用于计算覆盖率掩码
            IMG_H, IMG_W = 1024, 1024
            
            for xml_file in xml_files:
                try:
                    frame_id = int(os.path.splitext(xml_file)[0])
                except ValueError:
                    continue
                
                full_ann_path = os.path.join(ann_path, xml_file)
                gt_boxes = det_metric.getGtFromXml(full_ann_path) # [[x1, y1, x2, y2], ...]
                
                preds = preds_by_frame.get(frame_id, [])
                
                # --- 原有评估逻辑 ---
                if len(preds) > 0:
                    det_arr = np.array(preds)
                else:
                    det_arr = np.empty((0, 5))
                det_metric.update(gt_boxes, det_arr)
                
                # --- [ADDED] 计算 BBox 覆盖率 ---
                if len(gt_boxes) > 0:
                    # 1. 创建当前帧的点云掩码
                    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
                    if len(preds) > 0:
                        # 提取 x, y 坐标
                        pts = np.array(preds)[:, :2].astype(np.int32)
                        # 边界保护
                        pts_x = np.clip(pts[:, 0], 0, IMG_W - 1)
                        pts_y = np.clip(pts[:, 1], 0, IMG_H - 1)
                        mask[pts_y, pts_x] = 1
                    
                    # 2. 统计每个 GT 框内的覆盖情况
                    for box in gt_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        # 边界保护
                        x1, x2 = max(0, x1), min(IMG_W, x2)
                        y1, y2 = max(0, y1), min(IMG_H, y2)
                        
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area > 0:
                            # 提取 ROI 区域的点数
                            roi_points = np.sum(mask[y1:y2, x1:x2])
                            coverage = roi_points / box_area
                            
                            total_coverage_sum += coverage
                            total_gt_count_cov += 1

            # 4.4 计算当前视频的指标
            result = det_metric.get_result(img_size=[IMG_H, IMG_W], seq_len=num_images)
            
            # [ADDED] 计算平均覆盖率
            avg_coverage = (total_coverage_sum / total_gt_count_cov) * 100 if total_gt_count_cov > 0 else 0.0
            
            # 打印并记录 (增加了 %.2f 的 avg_coverage)
            res_str = '%s\t%.3f\t%.3f\t%.2f\t%.3f\t%.3f\t%.2e\t%.2e' % (
                folder_str, 
                result['recall'], result['prec'], avg_coverage, result['f1'], 
                result['pd'], result['fa_1'], result['fa_2']
            )
            print(res_str)
            fid.write(res_str + '\n')
            
            # 收集结果用于计算平均值
            all_results_list.append([
                result['recall'], result['prec'], avg_coverage, result['f1'], 
                result['pd'], result['fa_1'], result['fa_2']
            ])
            
        # 5. 计算并写入平均值
        if len(all_results_list) > 0:
            avg_res = np.mean(np.array(all_results_list), axis=0)
            avg_str = 'Avg\t%.3f\t%.3f\t%.2f\t%.3f\t%.3f\t%.2e\t%.2e' % (
                avg_res[0], avg_res[1], avg_res[2], avg_res[3], avg_res[4], avg_res[5], avg_res[6]
            )
            print("-" * 70)
            print(avg_str)
            print("-" * 70)
            fid.write('-' * 20 + '\n')
            fid.write(avg_str + '\n')
        else:
            print("No valid results found.")
            
        fid.close()
        print(f"Saved results to {save_path}\n")

if __name__ == '__main__':
    # 请替换为你的实际路径
    results_dirs = [
        './weights/rs_car_new_multi/I2PSOD/I2PSOD_xrsy_supMode_0_seglen8_weights2025_12_23_20_24_37/results_model_best_dis_f1_best/', 
    ]
    data_dir = '/media/xc/BA61C62ABCE29FF2/xc/dataset/RsCarData/images/test1024/'
    data_name = [2, 3, 5, 6, 8, 9, 10] 
    
    eval_net1_points(results_dirs, data_dir=data_dir, data_name=data_name, xmlname='xml_det')