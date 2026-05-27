from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import json
import cv2
import os
from lib.utils1.image import flip, color_aug
from lib.utils1.image import get_affine_transform, affine_transform
from lib.utils1.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils1.image import draw_dense_reg
import math
from lib.utils1.augmentations import Augmentation_st
from lib.dataset.data_aug.data_aug import RandomSampleCrop
import torch.utils.data as data

class CTDetDataset(data.Dataset):

    def get_im_ids(self, img_id):
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        video_id = img_info['video_id']
        video_frame_id = img_info['video_frame_id']
        video_len = img_info['video_len']
        video_info = self.video_to_images[video_id]

        if video_len - self.seqLen +1< video_frame_id: # 当剩余帧不足以构成一个完整序列时，注意seqlen是基于当前帧向后取 326-20+1=307
            video_frame_id_cur = video_len - self.seqLen+1 # 306
            img_id = video_info[video_frame_id_cur-1][0] # img_id被重新赋值时，说明到达了video的末尾patch

        im_ids = [img_id + i for i in range(self.seqLen)]
        return im_ids, video_frame_id
    '''def get_heatmap_large_gaussian(self, num_classes, output_h, output_w, c, s, bbox_tol, cls_id_tol):
        # 1. 基础变换准备
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        
        # 2. 初始化热力图
        hm_large = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        
        num_objs = min(len(bbox_tol), self.max_objs)
        
        for k in range(num_objs):
            bbox = bbox_tol[k].copy() # 避免修改原数据
            cls_id = int(cls_id_tol[k])
            
            # 3. 坐标变换：将原图 bbox 映射到特征图尺度
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:4] = affine_transform(bbox[2:], trans_output)

            # 4. 计算宽、高和中心点
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # 边界裁剪，保证宽高有效
            if h > 0 and w > 0:
                # 计算中心点坐标（浮点与整型）
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                
                # -----------------------------------------------------------------
                # 核心修改：计算各向异性的 Sigma 以满足 3-sigma 内切条件 # 3σ太瘦，对应 /6.0
                # 3 * sigma_x = w / 2  =>  sigma_x = w / 6
                # 3 * sigma_y = h / 2  =>  sigma_y = h / 6
                # -----------------------------------------------------------------
                sigma_x = w / 3.0
                sigma_y = h / 3.0
                
                # 避免 sigma 过小导致除零错误
                if sigma_x < 1e-2 or sigma_y < 1e-2:
                    continue

                # 5. 确定绘制范围 (ROI)
                # 既然 3sigma 处为边界，我们绘制范围取 3sigma (即半径) 即可覆盖主要能量
                # 为了防止数值截断，也可以取 mask_radius 稍大一点，比如 ceil(w/2)
                # radius_x = 30# int(math.ceil(w / 2))
                # radius_y = 30# int(math.ceil(h / 2))
                radius_x =  int(math.ceil(w))
                radius_y =  int(math.ceil(h))
                # 计算 ROI 在特征图上的坐标范围（处理边界溢出）
                # 左上角
                ul = [int(ct_int[0] - radius_x), int(ct_int[1] - radius_y)]
                # 右下角
                br = [int(ct_int[0] + radius_x + 1), int(ct_int[1] + radius_y + 1)]
                
                # 如果 ROI 完全在图像外，跳过
                if (ul[0] >= output_w or ul[1] >= output_h or br[0] < 0 or br[1] < 0):
                    continue

                # 6. 生成高斯网格 (Grid Generation)
                # 计算实际在图内的有效半径长度
                g_x = max(0, -ul[0]), min(br[0], output_w) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], output_h) - ul[1]
                
                # 对应的图像区域坐标
                img_x = max(0, ul[0]), min(br[0], output_w)
                img_y = max(0, ul[1]), min(br[1], output_h)

                # 生成网格坐标 (相对于中心点 ct_int 的偏移)
                # 注意：这里我们生成的是以 radius 为基准的网格，然后根据有效区域切片
                # grid范围: [-radius_x, radius_x]
                xs = np.arange(ul[0], br[0], dtype=np.float32) - ct_int[0]
                ys = np.arange(ul[1], br[1], dtype=np.float32) - ct_int[1]
                
                # 截取有效区域
                xs = xs[g_x[0]:g_x[1]]
                ys = ys[g_y[0]:g_y[1]]
                
                if len(xs) == 0 or len(ys) == 0:
                    continue

                # 创建二维网格
                XX, YY = np.meshgrid(xs, ys)
                
                # 7. 计算椭圆高斯分布值
                # Formula: exp( -0.5 * ( (dx/sigma_x)^2 + (dy/sigma_y)^2 ) )
                dist_sq = (XX ** 2) / (sigma_x ** 2) + (YY ** 2) / (sigma_y ** 2)
                gaussian_patch = np.exp(-0.5 * dist_sq)
    
                # 8. 更新热力图 (使用 max 操作保留最强响应)
                # 提取当前热力图区域
                current_hm_slice = hm_large[cls_id, img_y[0]:img_y[1], img_x[0]:img_x[1]]
                
                # Element-wise Maximum
                np.maximum(current_hm_slice, gaussian_patch, out=current_hm_slice)
                
                # 将更新后的切片写回
                hm_large[cls_id, img_y[0]:img_y[1], img_x[0]:img_x[1]] = current_hm_slice

        ret = {'hm_large_heatmap': hm_large}
        return ret'''
    def get_heatmap_large_gaussian(self, num_classes, output_h, output_w, c, s, bbox_tol, cls_id_tol):
        # =========================================================================
        # 模式选择开关
        # True  : 使用平顶高斯 (Super-Gaussian), Power=4, Sigma=w/2, Radius=w
        # False : 使用标准高斯 (Standard Gaussian), Power=2, Sigma=w/3, Radius=w
        # =========================================================================
        use_SuperGaussian = False 
        
        # 1. 基础变换准备
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        
        # 2. 初始化热力图
        hm_large = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        
        num_objs = min(len(bbox_tol), self.max_objs)
        
        for k in range(num_objs):
            bbox = bbox_tol[k].copy()
            cls_id = int(cls_id_tol[k])
            
            # 3. 坐标变换
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:4] = affine_transform(bbox[2:], trans_output)

            # 4. 计算宽、高和中心点
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            if h > 0 and w > 0:
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                
                # -----------------------------------------------------------------
                # 核心分支 1：Sigma 与 半径计算
                # -----------------------------------------------------------------
                if use_SuperGaussian:
                    # [模式 A: 平顶高斯]
                    # Sigma = w/2 (边缘即为拐点)
                    sigma_x = w / 2.0
                    sigma_y = h / 2.0
                    # 半径 = w (用户指定)
                    radius_x = int(math.ceil(w))
                    radius_y = int(math.ceil(h))
                else:
                    # [模式 B: 标准高斯]
                    # Sigma = w/3 (遵循 3-sigma 原则，w/2 处约为 0.1)
                    sigma_x = w / 3.0
                    sigma_y = h / 3.0
                    # 半径 = w (3倍 sigma = w，绘制半径取 w 足够覆盖 3-sigma 范围)
                    radius_x = int(math.ceil(w))
                    radius_y = int(math.ceil(h))
                
                # 避免除零保护
                if sigma_x < 1e-2 or sigma_y < 1e-2:
                    continue

                # 5. 确定绘制范围 (ROI)
                ul = [int(ct_int[0] - radius_x), int(ct_int[1] - radius_y)]
                br = [int(ct_int[0] + radius_x + 1), int(ct_int[1] + radius_y + 1)]
                
                # 边界检查
                if (ul[0] >= output_w or ul[1] >= output_h or br[0] < 0 or br[1] < 0):
                    continue

                # 6. 生成高斯网格
                g_x = max(0, -ul[0]), min(br[0], output_w) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], output_h) - ul[1]
                
                img_x = max(0, ul[0]), min(br[0], output_w)
                img_y = max(0, ul[1]), min(br[1], output_h)

                xs = np.arange(ul[0], br[0], dtype=np.float32) - ct_int[0]
                ys = np.arange(ul[1], br[1], dtype=np.float32) - ct_int[1]
                
                xs = xs[g_x[0]:g_x[1]]
                ys = ys[g_y[0]:g_y[1]]
                
                if len(xs) == 0 or len(ys) == 0:
                    continue

                XX, YY = np.meshgrid(xs, ys)
                
                # -----------------------------------------------------------------
                # 核心分支 2：公式计算
                # -----------------------------------------------------------------
                if use_SuperGaussian:
                    # [模式 A: 平顶高斯 / Generalized Gaussian]
                    # Exp( - (|x|/sigma)^4 - (|y|/sigma)^4 )
                    power = 4
                    term_x = (np.abs(XX) / sigma_x) ** power
                    term_y = (np.abs(YY) / sigma_y) ** power
                    # 注意：这里不需要乘 0.5，直接是 dist^power
                    gaussian_patch = np.exp(-(term_x + term_y))
                else:
                    # [模式 B: 标准高斯]
                    # Exp( -0.5 * ( (x/sigma)^2 + (y/sigma)^2 ) )
                    dist_sq = (XX ** 2) / (sigma_x ** 2) + (YY ** 2) / (sigma_y ** 2)
                    gaussian_patch = np.exp(-0.5 * dist_sq)
    
                # 8. 更新热力图
                current_hm_slice = hm_large[cls_id, img_y[0]:img_y[1], img_x[0]:img_x[1]]
                np.maximum(current_hm_slice, gaussian_patch, out=current_hm_slice)
                hm_large[cls_id, img_y[0]:img_y[1], img_x[0]:img_x[1]] = current_hm_slice

        ret = {'hm_large_heatmap': hm_large}
        return ret
    def get_heatmap(self, num_classes, output_h, output_w, c, s, bbox_tol, cls_id_tol):
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h]) # 缩放
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        hm_mask = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        draw_gaussian = draw_umich_gaussian
        num_objs = min(len(bbox_tol), self.max_objs)
        gt_det = []
        for k in range(num_objs):
            bbox = bbox_tol[k]
            cls_id = int(cls_id_tol[k])
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:4] = affine_transform(bbox[2:], trans_output)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            h = np.clip(h, 0, output_h - 1)
            w = np.clip(w, 0, output_w - 1)
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct[0] = np.clip(ct[0], 0, output_w - 1)
                ct[1] = np.clip(ct[1], 0, output_h - 1)
                ct_int = ct.astype(np.int32)

                draw_gaussian(hm[cls_id], ct_int, radius)

                hm_mask[cls_id, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1

                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                # if self.dense_wh:
                #     draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret_hm = {}
        ret_hm['hm'] = hm # 对于每个物体，它的中心点位置值为 1。以中心点为圆心，周围的值呈高斯分布衰减（0到1之间）。
        ret_hm['hm_mask'] = hm_mask # 在物体检测框覆盖的矩形区域内，值为 1；其余为 0。
        ret_hm['reg_mask'] = reg_mask # 值为 1 表示该索引处有真实物体，值为 0 表示是填充的空位（Padding） 形状：(max_objs)
        ret_hm['ind'] = ind # 物体中心点在特征图上的一维索引。 将二维坐标 (y, x) 扁平化（Flatten）后的索引值：ind = y * width + x  形状：(max_objs)
        ret_hm['wh'] = wh
        ret_hm['reg'] = reg # 由于特征图是下采样后的（例如缩小 4 倍），图像上的浮点数坐标映射到特征图上取整时会产生误差。 reg = 真实浮点中心 - 整数化中心 形状：(max_objs, 2)
        return ret_hm

    def get_single(self, img_id):
        #get info
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name'] # 应该是 "images/test/cityA_1159_1_2_5_5/img1/000001.png" or "images/test1024/002/img1/000001.jpg"
        base_name, _ = os.path.splitext(file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        #read the images
        im = cv2.imread(self.img_dir + file_name)
        # 处理1080尺寸
        ########################################################
        h, w = im.shape[:2]  # 1080, 1920
        target_h, target_w = self.resolution_ori[0], self.resolution_ori[1]
        if h!=target_h or w!=target_w:
            pad_h = target_h - h  # 8
            pad_w = target_w - w  # 0
            # 3. 实施填充 (只在底部填充 8 像素)
            # 参数顺序：top, bottom, left, right
            im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, 
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ########################################################
        ###
        if self.opt.sup_mode == 0:  #directly load the annotated labels
            anns0 = self.coco.loadAnns(ids=ann_ids)
            anns1 = [[i['bbox'][0], i['bbox'][1], i['bbox'][0]+i['bbox'][2], i['bbox'][1]+i['bbox'][3],
                     self.cat_ids[i['category_id']], i['track_id']
                     ] for i in anns0]
        elif self.opt.sup_mode == 1:  #load the generated unfilt labels
            coords = np.loadtxt(self.img_dir+base_name.replace('images', 'lrsd').replace('img1', 'coords_unfilt') +'.txt').reshape(-1,6)
            anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                     coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
        elif self.opt.sup_mode == 2: #load the generated filt labels
            coords = np.loadtxt(self.img_dir+base_name.replace('images', 'lrsd').replace('img1', 'coords_filt') +'.txt').reshape(-1,6)
            anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                     coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
        elif self.opt.sup_mode == 3: #load the generated updated labels
            coords = np.loadtxt(
                self.img_dir + base_name.replace('images', 'lrsd').replace('img1', 'coords_update') +'.txt').reshape(-1, 6)
            anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                      coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
        else:
            raise Exception('Not a valid sup_mode!!!!')
        #augmentation
        if self.aug is not None:
            im, anns = self.apply_aug(im ,anns1) # 在这里 /255. 了
        else:
            im = (im.astype(np.float32) / 255.)
            anns = np.array(anns1) # val报错修改 lhg
        #get gray image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = np.expand_dims(im_gray, 0)
        ##normalization
        im = (im - self.mean) / self.std
        im = im.transpose(2, 0, 1)
        ##get hm
        _, input_h, input_w = im.shape
        output_h = input_h // self.opt.down_ratio # 不下采样 1
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        c = np.array([input_w / 2., input_h / 2.], dtype=np.float32)
        s = max(input_h, input_w) * 1.0

        # print(f"file_name: {file_name}, anns shape: {anns.shape}, anns content: {anns}")
        #空标注 → 转为(0,4)的二维空数组
        if len(anns) == 0:
            anns = np.empty((0, 6), dtype=np.float32)
        bbox_temp = anns[:,:4].tolist()
        cls_id_temp = anns[:,4].tolist()
        ret_single = self.get_heatmap(num_classes, output_h, output_w, c, s,bbox_temp, cls_id_temp)
        # ret_single-->{'hm', 'hm_mask', 'reg_mask', 'ind', 'wh', 'reg'}
        ret_hm_large_heatmap = self.get_heatmap_large_gaussian(num_classes, output_h, output_w, c, s,bbox_temp, cls_id_temp)
        ret_single.update(ret_hm_large_heatmap) # {'hm_large_heatmap':hm_large_heatmap} # lhg 生成大的点云监督信号
        # ================== 【新增代码 START】 ==================
        # 构造 bboxes (128, 6) -> [x1, y1, x2, y2, cls, track_id]
        # 直接使用 anns，因为它已经是 [n, 6] 且经过了 augmentation (如果开启)
        # 这里的 anns 对应的是输入图像 im 的坐标，无需额外缩放
        bboxes_out = np.zeros((256, 6), dtype=np.float32)
        if len(anns) > 0:
            # 截断处理
            valid_num = min(len(anns), 256)
            # 填充数据
            bboxes_out[:valid_num] = anns[:valid_num]
        # ================== 【新增代码 END】 ==================
        #####
        ret_single.update({'input': im, 'input_gray': im_gray*255, 'file_name': file_name, 'bboxes': bboxes_out})
        return ret_single

    def get_multi(self, img_id):
        multi_ids, _ = self.get_im_ids(img_id)
        img_id = multi_ids[0] # 更新最后一个patch不够长的情况
        # print(img_id, multi_ids)
        ret_multi = {'input': [], 'input_gray': [], 'hm': [], 'hm_mask': [],'reg_mask': [], 'ind': [], 'wh': [], 'hm_large_heatmap': [], 'file_name': []}
        ret_multi.update({'reg': []})
        ret_multi.update({'bboxes': []})
        #choose which frame as input
        img_chosen_id = self.seqLen // 2
        for c_id  in multi_ids:
            ret_tmp = self.get_single(c_id)
            for k,v in ret_tmp.items():
                # if k == 'im_ids' or k == 'file_name' or k=='meta': # 添加file_name
                if k == 'im_ids' or k=='meta':
                    continue
                else:
                    if  k=='reg_mask' or k=='ind' or k=='wh' or k=='reg':
                        ret_multi[k].append(np.expand_dims(v, 0))
                    elif k=='bboxes':
                        ret_multi[k].append(np.expand_dims(v, 0)) # 这样 bboxes (128, 6) 会变成 (1, 128, 6) 存入列表，方便后续按时间维度堆叠
                    elif k=='hm' or k=='hm_mask' or k=='hm_large_heatmap':
                        ret_multi[k].append(np.expand_dims(v, 1))
                    elif k=='file_name':
                        ret_multi[k].append(v)
                    else:
                        ret_multi[k].append(np.expand_dims(v,1))
        for k,v in ret_multi.items():
            if k == 'reg_mask' or k == 'ind' or k == 'wh' or k == 'reg':
                v1 = np.concatenate(v, axis=0)
                # v1 = v1[img_chosen_id]
            elif k == 'bboxes': # 拼接后维度变为 (Time, 128, 6)
                v1 = np.concatenate(v, axis=0)
            elif k == 'hm' or k == 'hm_mask' or k == 'hm_large_heatmap':
                v1 = np.concatenate(v, axis=1)
                # v1 = v1[:,img_chosen_id]
            elif k == 'file_name':
                v1 = v
            else:
                v1 = np.concatenate(v, axis=1)
            ret_multi[k] = v1
        return img_id, ret_multi

    def get_aug(self, annos=None):
        s0 = np.random.choice(np.arange(0.9, 1.2, 0.1))
        c = np.array([self.resolution_ori[1] / 2., self.resolution_ori[0] / 2.], dtype=np.float32)
        s = max(self.resolution_ori[0], self.resolution_ori[1]) * s0 # 是x还是/
        self.trans_ori = get_affine_transform(c, s, 0, [self.resolution_ori[1], self.resolution_ori[0]])  # random scale
        #######crop target region
        annos_t = []
        if annos is not None:
            for anno in annos:
                anno = affine_transform(anno, self.trans_ori)
                if anno[0]>0 and anno[1]>0 and anno[0]<self.resolution_ori[0] and anno[1]<self.resolution_ori[1]:
                    annos_t.append(anno)
        if len(annos_t)>0:
            id = np.random.randint(0, len(annos_t))
            anno_crop = annos_t[id][0:2]
            anno_crop = anno_crop[::-1]
        else:
            anno_crop= None
        #######
        self.crop = RandomSampleCrop(self.resolution_ori, self.resolution, anno_coord=anno_crop)  # random crop
        self.aug = Augmentation_st()  # random mirror
        self.color_aug = color_aug(self._eig_val, self._eig_vec)  # random color

    def apply_aug(self, im, anns):
        num_objs = len(anns)
        # random scale
        try:
            im = cv2.warpAffine(im, self.trans_ori,
                                (self.resolution_ori[1], self.resolution_ori[0]),
                                flags=cv2.INTER_LINEAR)
        except:
            a=1
        bboxes = []
        for k in range(num_objs):
            bbox = anns[k]
            ##random scale
            bbox[:2] = affine_transform(bbox[:2], self.trans_ori)
            bbox[2:4] = affine_transform(bbox[2:4], self.trans_ori)
            bboxes.append(bbox)
        # random crop
        im, anns_new = self.crop(im, np.array(bboxes).reshape(-1, 6))
        #random mirror
        im, bbox_tol, cls_id_tol = self.aug(im, anns_new[:,:4], anns_new[:,4:])
        anns_new[:, :4] = bbox_tol
        #random color
        im = (im.astype(np.float32) / 255.)
        # im = self.color_aug(im)
        return im, anns_new

    def __getitem__(self, index):
        ######get image id
        img_id = self.images[index]
        
        if self.split != 'train' and self.opt.multi2multi: 
            img_ids_,video_frame_id_ = self.get_im_ids(img_id)
            img_id_ = img_ids_[0]
            if video_frame_id_ % self.opt.seqLen == 1 or img_id_ != img_id:# video_frame_id_为当前选中的帧，%20==1则需要返回值，img_id_ != img_id说明到达了末尾patch，需要返回值，并且如果多次返回相同img_id_表示正常情况或者末尾重复返回，不需要返回值
                pass
            else:
                return img_id, {} # 此时不需要返回值
        ######get params
        self.down_ratio = self.opt.down_ratio
        self.max_objs = self.opt.K
        ######aug or not
        if self.split == 'train':
            file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
            base_name, _ = os.path.splitext(file_name)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            # read anns
            if self.opt.sup_mode == 0:  # directly load the annotated labels
                anns0 = self.coco.loadAnns(ids=ann_ids)
                anns1 = [[i['bbox'][0], i['bbox'][1], i['bbox'][0] + i['bbox'][2], i['bbox'][1] + i['bbox'][3],
                          self.cat_ids[i['category_id']], i['track_id']
                          ] for i in anns0]
            elif self.opt.sup_mode == 1:  # load the generated unfilt labels
                coords = np.loadtxt(
                    self.img_dir + base_name.replace('images', 'lrsd').replace('img1', 'coords_unfilt') +'.txt').reshape(
                    -1, 6)
                anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                          coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
            elif self.opt.sup_mode == 2:  # load the generated filt labels
                coords = np.loadtxt(
                    self.img_dir + base_name.replace('images', 'lrsd').replace('img1', 'coords_filt') +'.txt').reshape(
                    -1, 6)
                anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                          coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
            elif self.opt.sup_mode == 3:  # load the generated updated labels
                coords = np.loadtxt(
                    self.img_dir + base_name.replace('images', 'lrsd').replace('img1', 'coords_update') +'.txt').reshape(-1, 6)
                anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                          coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
            else:
                raise Exception('Not a valid sup_mode!!!!')
            self.get_aug(anns1) # 对后面统一做相同的数据增强策略
            # self.get_aug()
        else:
            self.aug = None
            
        #####switch mode
        if self.opt.data_mode == 'single':
            ret = self.get_single(img_id)
        elif self.opt.data_mode == 'multi':
            self.seqLen = self.opt.seqLen
            img_id, ret = self.get_multi(img_id) # 自动更新img_id
        else:
            raise Exception('Not a valid data mode!!!')
        ####get results
        return img_id, ret