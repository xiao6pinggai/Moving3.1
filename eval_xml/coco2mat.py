import json
import os
import scipy.io as sio
from collections import defaultdict


def coco_pred_to_mat(pred_json_path, gt_json_path, save_path, threshold=0.5, topk=100):
    """
    将COCO格式的预测结果转换为MAT文件

    Args:
        pred_json_path: 预测结果JSON文件路径
        gt_json_path: 真值JSON文件路径
        save_path: 保存路径
        threshold: 置信度阈值
        topk: 每张图片最多保留的检测框数量
    """

    # 读取预测结果JSON
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)

    # 读取真值JSON
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    # 创建image_id到file_name的映射
    image_id_to_info = {}
    for image in gt_data['images']:
        image_id_to_info[image['id']] = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height']
        }

    # 按image_id分组预测结果
    pred_by_image = defaultdict(list)
    for pred in pred_data:
        if pred['score'] >= threshold:
            pred_by_image[pred['image_id']].append(pred)

    # 处理每个图像的预测结果
    for image_id, preds in pred_by_image.items():
        if image_id not in image_id_to_info:
            continue

        # 获取图像信息
        image_info = image_id_to_info[image_id]
        file_path = image_info['file_name']

        # 解析文件夹名和文件名
        # 假设路径格式为: "images/test/cityA_1159_1_2_5_5/img1/000001.png"
        parts = file_path.split('/')
        folder_name = parts[-3]  # cityA_1159_1_2_5_5
        file_name = parts[-1].replace('.png', '').replace('.jpg','')  # 000001

        # 按得分排序并取前topk个
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)[:topk]

        # 转换bbox格式并准备保存数据
        detections = []
        for pred in preds_sorted:
            bbox = pred['bbox']  # [x, y, w, h]
            # 转换为xyxy格式
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            score = pred['score']

            detections.append([x1, y1, x2, y2, score])

        # 转换为numpy数组格式（MAT文件兼容）
        detections_array = [det for det in detections]

        # 创建保存目录
        folder_save_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_save_path, exist_ok=True)

        # 保存为.mat文件
        mat_file_path = os.path.join(folder_save_path, f"{file_name}.mat")
        sio.savemat(mat_file_path, {'A': detections_array})

        print(f"Saved {len(detections)} detections to {mat_file_path}")


def main():
    from datetime import datetime
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    # 使用示例
    pred_json_path = "runs/detect/yolo26mval/exp_sdmcar_yolo26m/predictions_sdmcar.json"
    # pred_json_path = 'map_out/coco_eval_rscarnew/eval_results.json'
    dataname = pred_json_path.split('/')[-2]
    model_name = 'yolo26m'
    pred_json_name = pred_json_path.split('/')[-1].replace('.', '_')
    # gt_json_path = r"/root/autodl-tmp/RsCarData_New_Part/annotations/instances_test1024_new.json"
    gt_json_path = r"/root/autodl-tmp/SDM-Car-New/annotations/annotations_test_new.json"
    save_path = f"runs/detect/yolo26mval/exp_sdmcar_yolo26m/{time_str}_{dataname}_{model_name}_{pred_json_name}"
    # save_path = f"map_out/coco_eval_rscarnew/{time_str}_{dataname}_{model_name}_{pred_json_name}"
    threshold = 0
    # topk = 128
    topk = 360

    coco_pred_to_mat(pred_json_path, gt_json_path, save_path, threshold, topk)


if __name__ == "__main__":
    main()