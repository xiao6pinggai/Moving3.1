from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils1.decode import ctdet_decode
from lib.utils1.post_process import ctdet_post_process
import cv2
from concurrent.futures import ThreadPoolExecutor  # 多线程处理IO
try:
    from lib.external1.nms import soft_nms
except:
    from lib.external1.nms import soft_nms

from lib.test_utils.get_coords import *
import xml.etree.ElementTree as ET
import numpy as np
import os
'''
def pre_process(image, scale=1):
    height, width = image.shape[2:4]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height,
            'out_width': inp_width}
    return meta

def preprocess(img_list, dataset):
    seq_num = len(img_list)
    img = np.zeros([dataset.resolution[0], dataset.resolution[1], 3, seq_num]) # 原始图像
    imgs = np.zeros([dataset.resolution[0], dataset.resolution[1], 3, seq_num]) # 存储归一化后的图像，将变形为inp
    imgs_gray = np.zeros([dataset.resolution[0], dataset.resolution[1], 1, seq_num]) # 存储未归一化的灰度图
    a1 = time.time()
    for ii in range(seq_num):
        img_id_cur = img_list[ii]
        im = cv2.imread(img_id_cur)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img[:, :, :, ii] = im
        ###
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        imgs_gray[:, :, 0, ii] = im_gray
        # normalize
        inp_i = (im.astype(np.float32) / 255.)
        inp_i = (inp_i - dataset.mean) / dataset.std
        imgs[:, :, :, ii] = inp_i

    a2 = time.time()
    inp = np.expand_dims(imgs.transpose(2, 3, 0, 1).astype(np.float32),0)
    inp_gray = np.expand_dims(imgs_gray.transpose(2, 3, 0, 1).astype(np.float32),0)
    # 读入图像
    meta = pre_process(inp, 1)
    # batch_dict = get_points(inp, inp_gray)
    batch_dict ={}
    input_imgs = {}
    input_imgs['input'] = inp # 归一化
    input_imgs['input_gray'] = inp_gray # 未归一化
    return batch_dict, meta, img, input_imgs

'''
'''
def pre_process(image, scale=1):
    # 原函数逻辑无明显性能问题，仅微调索引（原image.shape[2:4]可能越界，修正为[0:2]）
    height, width = image.shape[2:4] if len(image.shape) >= 4 else image.shape[0:2]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height,
            'out_width': inp_width}
    return meta

def read_single_xml(args):
    """
    args: (xml_path, orig_shape, target_shape)
    读取 VOC 格式 XML，提取 [x1, y1, x2, y2, class, id]
    """
    xml_path, (orig_h, orig_w), (target_h, target_w) = args
    
    # 计算缩放因子
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    # 你保留的调试代码
    if scale_x != 1 or scale_y != 1:
        # 实际训练中这里通常不需要打印error，除非你预期不缩放
        # print('error!') 
        pass 
    
    boxes = []
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 遍历所有 object 标签
            for obj in root.findall('object'):
                b = obj.find('bndbox')
                
                # 1. 坐标读取与缩放
                x1 = float(b.find('xmin').text) * scale_x
                y1 = float(b.find('ymin').text) * scale_y
                x2 = float(b.find('xmax').text) * scale_x
                y2 = float(b.find('ymax').text) * scale_y
                
                # 2. 读取跟踪ID (转为float以存入numpy数组)
                track_id = float(obj.find('id').text)
                
                # 3. 固定类别为 1
                class_id = 1
                
                # 存入列表：[x1, y1, x2, y2, class, id]
                boxes.append([x1, y1, x2, y2, class_id, track_id])
                
        except Exception:
            pass 
            
    # --- 定长填充逻辑 (Max=512, Dim=6) ---
    # 修改：初始化形状改为 (512, 6)
    np_boxes = np.zeros((512, 6), dtype=np.float32)
    
    if len(boxes) > 0:
        boxes_arr = np.array(boxes, dtype=np.float32)
        
        # 截断处理
        valid_num = min(len(boxes), 512)
        
        # 填充数据
        np_boxes[:valid_num] = boxes_arr[:valid_num]
        
    return np_boxes

# 封装单张图像读取函数（供多线程调用）
def read_single_image(img_path):
    im = cv2.imread(img_path)
    if im is None:
        raise ValueError(f"无法读取图像：{img_path}")
    # 提前统一尺寸（避免后续赋值时尺寸不匹配）
    return im


def preprocess(img_list, dataset, xml_list):
    seq_num = len(img_list)
    if seq_num == 0:
        raise ValueError("图像列表为空")
    # 统一图像尺寸（与dataset.resolution匹配）
    target_h, target_w = dataset.resolution[0], dataset.resolution[1] # 先行后列
    # ========== 优化1：多线程并行读取图像（解决IO瓶颈） ==========
    a1 = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:  # 线程数可根据CPU核心调整
        img_list_read = list(executor.map(read_single_image, img_list))
        # 复用同一个 executor 提交 XML 读取任务
        # 利用刚刚读到的图片获取原始尺寸
        xml_args = [
            (xml_list[i], img_list_read[i].shape[:2], (target_h, target_w)) 
            for i in range(seq_num)
        ]
        bbox_list = list(executor.map(read_single_xml, xml_args))
    # 堆叠 Bbox -> (seq_num, 512, 4)
    bbox_seq = np.stack(bbox_list, axis=0)
    a1_io = time.time()

    # ========== 优化2：批量转换为4维numpy数组（避免循环赋值） ==========
    
    # 批量调整尺寸 + 转换为4维数组 (height, width, 3, seq_num)
    img_batch = np.stack([cv2.resize(im, (target_w, target_h)) for im in img_list_read], axis=-1)
    a1_batch = time.time()
    
    # ========== 核心优化：灰度转换+归一化 ==========
    # 1. 提前转float32，避免多次类型转换
    img_batch_float = img_batch.astype(np.float32)
    
    # 2. 灰度转换：np.einsum替代相乘+sum，更高效
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    imgs_gray = np.expand_dims(
        np.einsum('hwcn, c -> hwn', img_batch_float, weights).astype(np.uint8),
        axis=2
    )
    # 3. 归一化：复用float数组，减少内存操作
    mean = dataset.mean.reshape(1, 1, 3, 1)
    std = dataset.std.reshape(1, 1, 3, 1)
    imgs_normalized = (img_batch_float / 255.0 - mean) / std
    a1_calc = time.time()

    # ========== 维度变换（批量处理） ==========
    # 调整维度为模型输入格式：(1, 3, seq_num, h, w)
    inp = np.expand_dims(imgs_normalized.transpose(2, 3, 0, 1).astype(np.float32), axis=0)
    inp_gray = np.expand_dims(imgs_gray.transpose(2, 3, 0, 1).astype(np.float32), axis=0)
    # Bbox 增加 Batch 维度 -> (1, seq_num, 512, 4)
    inp_bboxes = np.expand_dims(bbox_seq, axis=0)
    # 生成meta
    meta = pre_process(inp, 1)

    # 构造返回数据
    batch_dict = {}
    input_imgs = {
        'input': inp,  # 归一化后的输入
        'input_gray': inp_gray,  # 未归一化的灰度图
        'bboxes': inp_bboxes
    }
    a2 = time.time()

    # # 可选：打印耗时分析（便于调优）
    # print(f"IO读取耗时：{a1_io - a1:.4f}s")
    # print(f"批量转换耗时：{a1_batch - a1_io:.4f}s")
    # print(f"批量计算耗时：{a1_calc - a1_batch:.4f}s")
    # print(f"总预处理耗时：{a2 - a1:.4f}s")

    return batch_dict, meta, img_batch, input_imgs
'''
def pre_process(image, scale=1):
    # image shape expected: (1, 3, N, H, W)
    # image.shape[3:5] corresponds to H, W
    height, width = image.shape[3], image.shape[4]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height,
            'out_width': inp_width}
    return meta

def read_single_xml(args):
    # ... (保持你原有的逻辑不变，它是正确的) ...
    # 为了节省篇幅省略，逻辑无误
    xml_path, (orig_h, orig_w), (target_h, target_w) = args
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    boxes = []
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                b = obj.find('bndbox')
                x1 = float(b.find('xmin').text) * scale_x
                y1 = float(b.find('ymin').text) * scale_y
                x2 = float(b.find('xmax').text) * scale_x
                y2 = float(b.find('ymax').text) * scale_y
                track_id = float(obj.find('id').text)
                class_id = 1
                boxes.append([x1, y1, x2, y2, class_id, track_id])
        except Exception:
            pass 
            
    np_boxes = np.zeros((512, 6), dtype=np.float32)
    if len(boxes) > 0:
        boxes_arr = np.array(boxes, dtype=np.float32)
        valid_num = min(len(boxes), 512)
        np_boxes[:valid_num] = boxes_arr[:valid_num]
    return np_boxes

def process_single_image_task(args):
    padding_test = False
    """
    单个图像处理任务：读取 -> Resize -> 转灰度(可选)
    返回: (resized_img_bgr, resized_img_gray, orig_shape)
    """
    img_path, target_h, target_w = args
    im = cv2.imread(img_path)
    if im is None:
        raise ValueError(f"无法读取图像：{img_path}")
    orig_shape = im.shape[:2] # (h, w)
    if padding_test:
        
        # 处理1080尺寸
        ########################################################
        h, w = im.shape[:2]  # 1080, 1920
        # target_h, target_w = self.resolution_ori[0], self.resolution_ori[1]
        if h!=target_h or w!=target_w:
            pad_h = target_h - h  # 8
            pad_w = target_w - w  # 0
            # 3. 实施填充 (只在底部填充 8 像素)
            # 参数顺序：top, bottom, left, right
            im_resized = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, 
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            im_resized = im
        ########################################################
         ###
    else:
        im_resized = im
    # # Resize
    # im_resized = cv2.resize(im, (target_w, target_h)) # (H, W, 3)
    
    # 提前计算灰度 (利用 cv2 的优化，通常比 numpy 快)
    # 保持维度为 (H, W, 1) 以便后续堆叠
    im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
    im_gray = np.expand_dims(im_gray, axis=2) 
    
    return im_resized, im_gray, orig_shape

def preprocess(img_list, dataset, xml_list):
    seq_num = len(img_list)
    if seq_num == 0:
        raise ValueError("图像列表为空")
    
    target_h, target_w = dataset.resolution[0], dataset.resolution[1] # H, W
    
    # ========== 优化：合并读取与预处理任务 ==========
    # 构造任务参数
    img_args = [(p, target_h, target_w) for p in img_list]
    
    a1 = time.time()
    
    # 列表用于收集结果
    img_batch_list = []
    gray_batch_list = []
    xml_args = []
    
    with ThreadPoolExecutor(max_workers=8) as executor: # IO密集型可以开大点 workers
        # 1. 并行处理图像 (Read + Resize + Gray)
        results = list(executor.map(process_single_image_task, img_args))
        
        for i, res in enumerate(results):
            im_resized, im_gray, orig_shape = res
            img_batch_list.append(im_resized)
            gray_batch_list.append(im_gray)
            
            # 准备 XML 读取参数
            xml_args.append((xml_list[i], orig_shape, (target_h, target_w)))
            
        # 2. 并行处理 XML (此时图像已经处理完，CPU空闲出来解析XML)
        bbox_list = list(executor.map(read_single_xml, xml_args))

    a1_io = time.time()
    
    # ========== 堆叠与归一化 ==========
    
    # 堆叠 -> (N, H, W, 3)  <-- 内存连续性更好
    img_batch = np.stack(img_batch_list, axis=0)
    gray_batch = np.stack(gray_batch_list, axis=0) # (N, H, W, 1)
    bbox_seq = np.stack(bbox_list, axis=0) # (N, 512, 6)
    
    # 归一化 (利用广播机制)
    # mean/std shape: (1, 1, 1, 3) 用于匹配 (N, H, W, 3)
    mean = dataset.mean.reshape(1, 1, 1, 3).astype(np.float32)
    std = dataset.std.reshape(1, 1, 1, 3).astype(np.float32)
    
    # 转换为 float32 并归一化
    # 这一步是内存消耗大户，如果内存吃紧，可以考虑原地操作或分块
    img_batch_float = img_batch.astype(np.float32)
    imgs_normalized = (img_batch_float / 255.0 - mean) / std
    
    # 灰度图不需要 float32 归一化吗？看你原代码没有归一化，只转了 float
    # 这里保持一致，只转 float32 (N, H, W, 1)
    inp_gray = gray_batch.astype(np.float32)

    # ========== 维度变换 (N, H, W, C) -> (1, C, N, H, W) ==========
    
    # Transpose: (N, H, W, C) -> (C, N, H, W)
    # 0->1, 1->2, 2->3, 3->0
    inp = np.expand_dims(imgs_normalized.transpose(3, 0, 1, 2), axis=0)
    
    # Gray Transpose
    inp_gray = np.expand_dims(inp_gray.transpose(3, 0, 1, 2), axis=0)
    
    # Bbox: (1, N, 512, 6)
    inp_bboxes = np.expand_dims(bbox_seq, axis=0)
    
    meta = pre_process(inp, 1)

    batch_dict = {}
    input_imgs = {
        'input': inp, 
        'input_gray': inp_gray, 
        'bboxes': inp_bboxes
    }
    
    # 此时 img_batch 是 (N, H, W, 3) 且为 uint8，适合可视化
    # 如果外部需要 (H, W, 3, N)，则 transpose 一下
    img_batch_return = img_batch.transpose(1, 2, 3, 0)

    return batch_dict, meta, img_batch_return, input_imgs
def process(model, batch, return_time, opt, K=128):
    with torch.no_grad():
        output = model(batch)[-1]
        hm = output['hm']
        wh = output['wh']
        if opt.off_flag:
            reg = output['reg']
        else:
            reg = None
        torch.cuda.synchronize()
        forward_time = time.time()

        if reg is not None:
            dets =  ctdet_decode(hm[0].transpose(0,1), wh[0].transpose(0,1),
                    reg=reg[0].transpose(0,1), K=K)
        else:
            dets =  ctdet_decode(hm[0].transpose(0,1), wh[0].transpose(0,1),
                       reg=None, K=K)
    if return_time:
        return output, dets, forward_time
    else:
        return output, dets # dets本应该是全部框，但是也被topk了

def post_process(dets_all, meta, num_classes=1, scale=1, max_per_image=100):
    # 后处理
    rets = []
    dets_post = []
    dets_all = dets_all.unsqueeze(1).detach().cpu().numpy()
    for iii in range(dets_all.shape[0]):
        dets = dets_all[iii]
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], num_classes)
        for j in range(1, num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        detection = []
        det = dets[0]
        dets_post.append(det) # 模型输出的全部框，用于调试查看原始结果
        detection.append(det)
        ret = merge_outputs(detection, num_classes, max_per_image)
        rets.append(ret) # 模型输出经过NMS后前K个框，用于可视化
    return rets, dets_post

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.1, method=1)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results








