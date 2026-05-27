import os
import cv2
import scipy.io as sio
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm

def draw_bboxes(img, bboxes, color=(0, 255, 0), thickness=1, threshold=None):
    """
    在图片上绘制 bbox
    bboxes: numpy array, shape (N, 5) or (N, 4)
            如果是 (N, 5) -> [x1, y1, x2, y2, score]
            如果是 (N, 4) -> [x1, y1, x2, y2] (通常是真值)
    threshold: 阈值，如果提供了阈值且bbox包含分数，低于该值的框将被忽略
    """
    # 维度保护：如果是一维数组，转为二维
    if bboxes.ndim == 1:
        bboxes = bboxes[np.newaxis, :]

    for box in bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        
        # 判断是否需要阈值筛选
        if len(box) >= 5:
            score = box[4]
            if threshold is not None and score <= threshold:
                continue # 分数过低，跳过
        
        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # (可选) 绘制分数，真值通常没有分数
        # if len(box) >= 5:
        #     cv2.putText(img, f"{score:.2f}", (x1, y1 - 2), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
    return img

def parse_xml_bboxes(xml_path):
    """
    解析 XML 文件获取真值框
    返回: numpy array (N, 4) -> [[x1, y1, x2, y2], ...]
    """
    bboxes = []
    if not os.path.exists(xml_path):
        return np.array([])
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                bboxes.append([xmin, ymin, xmax, ymax])
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        
    return np.array(bboxes)

def visualize_dataset(img_root, mat_root, save_root, conf_thresh=0.3):
    # 1. 获取所有视频文件夹
    video_folders = [f for f in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, f))]
    video_folders.sort()
    
    # 仅处理前两个用于测试
    video_folders = video_folders[:2]
    
    print(f"发现 {len(video_folders)} 个视频文件夹。")

    for video_name in tqdm(video_folders, desc="Processing Videos"):
        # 路径构建
        img_video_dir = os.path.join(img_root, video_name, "img1") # 注意这里可能有 img1
        mat_video_dir = os.path.join(mat_root, video_name)
        save_video_dir = os.path.join(save_root, video_name)

        if not os.path.exists(img_video_dir):
            print(f"Skipping {video_name}, img1 folder not found.")
            continue

        # 创建保存目录
        os.makedirs(save_video_dir, exist_ok=True)

        # 获取该视频下的所有图片
        img_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
            img_paths.extend(glob(os.path.join(img_video_dir, ext)))
        
        # 遍历每一帧图片
        for img_path in img_paths:
            file_name_with_ext = os.path.basename(img_path)
            file_id = os.path.splitext(file_name_with_ext)[0]
            
            # --- 1. 准备路径 ---
            # Mat 路径
            mat_path = os.path.join(mat_video_dir, f"{file_id}.mat")
            
            # XML 路径：将 img1 替换为 xml1，后缀改为 .xml
            # 假设 img_path 是 .../video/img1/001.jpg
            # 这里的 replace 比较粗暴，但只要路径规范一般没问题
            xml_path = img_path.replace('img1', 'xml1')
            xml_path = os.path.splitext(xml_path)[0] + '.xml'

            # --- 2. 读取图片 ---
            img = cv2.imread(img_path)
            if img is None:
                continue

            # --- 3. 绘制真值框 (红色, Red: B=0, G=0, R=255) ---
            gt_bboxes = parse_xml_bboxes(xml_path)
            if gt_bboxes.size > 0:
                img = draw_bboxes(img, gt_bboxes, color=(0, 0, 255), thickness=1, threshold=None)

            # --- 4. 绘制预测框 (绿色, Green: B=0, G=255, R=0) ---
            if os.path.exists(mat_path):
                try:
                    mat_data = sio.loadmat(mat_path)
                    pred_bboxes = None
                    
                    # 自动寻找变量名
                    for key in mat_data:
                        if key.startswith('__'): continue 
                        val = mat_data[key]
                        if isinstance(val, np.ndarray) and (val.ndim == 2 and val.shape[1] == 5):
                            pred_bboxes = val
                            break
                        elif isinstance(val, np.ndarray) and (val.ndim == 2 and val.shape[0] == 5):
                            pred_bboxes = val.T
                            break
                    
                    if pred_bboxes is not None and pred_bboxes.size > 0:
                        # 传入 conf_thresh 进行筛选
                        img = draw_bboxes(img, pred_bboxes, color=(0, 255, 0), thickness=1, threshold=conf_thresh)

                except Exception as e:
                    print(f"Error reading MAT {mat_path}: {e}")

            # --- 5. 保存结果 ---
            save_path = os.path.join(save_video_dir, file_name_with_ext)
            cv2.imwrite(save_path, img)

if __name__ == "__main__":
    # --- 配置路径 ---
    IMAGE_ROOT = r"/root/autodl-tmp/SDM-Car-New/images/test/" 
    MAT_ROOT   = r"./weights_SDMCar/sdm_car_multi/I2PSOD/I2PSOD_supMode_0_seglen5_weights2025_12_28_15_44_29/results_latest"
    SAVE_ROOT  = r"./weights_SDMCar/sdm_car_multi/I2PSOD/I2PSOD_supMode_0_seglen5_weights2025_12_28_15_44_29/results_latest/vis"
    
    # 设置预测框的显示阈值 (例如 0.3)
    CONF_THRESHOLD = 0.1

    visualize_dataset(IMAGE_ROOT, MAT_ROOT, SAVE_ROOT, CONF_THRESHOLD)
    print("\n处理完成！")