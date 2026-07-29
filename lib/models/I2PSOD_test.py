import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import cv2
import os, sys
import time
import atexit
import math
import inspect

# 向上查找项目根目录并加入 sys.path（支持 autodl/本地 Windows 双环境）
_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_cur, 'path_setup.py')):
    _cur = os.path.dirname(_cur)
if _cur not in sys.path:
    sys.path.insert(0, _cur)


from lib.utils1.enlarge_mask import dilate_mask_fast
from lib.utils1.bbox2binarymask import bboxes_to_binary_mask

from lib.models.spconv_unet import UNetV2, UNetV2_3, UNetV2_2, UNetV2_3_32, UNetV2_3_T_nodown, UNetV2_3_T_nodown_maxpool, UNetV2_3_T_nodown_v2, UNetV2_3_T_nodown_v3
from lib.models.spconv_utils import replace_feature, spconv
from lib.models.noramlconv_unet3d2_1 import UNet2DWithNormalConv2D, UNet3DWithNormalConv3D, LightWeightedConv3D, EncoderOnlyConv3DProposalNet, TOSConvNet, TPConvNet, TZSConvNet, DynamicTOSConvNet
from lib.models.profile_utils import count_parameters, estimate_module_flops, estimate_sigmoid_flops, extract_feature_tensor
from lib.utils1.show_one_img import show_one_img
from lib.utils1.save_img import save_net1_output
import torch

class Img2PointsSmallObjectDetection(nn.Module):
    def __init__(self, heads, image_size = [512,512], img_num = 20, layers = 3, thresh=None, input_channels=1,
                 feat_channels=[16,32,64,128], T_pooling=False,groups=1,downsample_mode='stride',
                 net1name='UNet3DWithNormalConv3D', opt=None):
        super().__init__()
        self.print = 0
        self.use_runtime = getattr(opt, 'use_runtime', True)
        self.runtime_component_names = (
            'I2PNet',
            'ACS',
            'BackboneNoMFE',
            'MFE',
            'DetHead',
        )
        self.component_display_names = {
            'I2PNet': 'SRC w/o ACS',
            'ACS': 'ACS',
            'BackboneNoMFE': 'GCC w/o TAA',
            'MFE': 'TAA',
            'DetHead': 'Det. Head',
            'Total': 'Total',
        }
        self._mfe_module_prefixes = (
            'shortcut1',
            'shortcut2',
            'shortcut3',
            'shortcut1fusion',
            'shortcut2fusion',
            'shortcut3fusion',
            'sptial2d1',
            'sptial2d2',
            'sptial2d3',
        )
        self._component_param_cache = None
        self._unclassified_param_count = 0
        self._runtime_frame_divisor = 1.0
        self._component_profile_printed = False
        self._component_profile_finalize_registered = False
        if not self._component_profile_finalize_registered:
            atexit.register(self._finalize_component_profile)
            self._component_profile_finalize_registered = True
        self.reset_runtime_stats()

        # ===== 数据集级累计统计量（非训练阶段使用）=====
        # 点级召回：采样点在框内 / 框内总像素  (按帧-框平均)
        self._stat_recall_sum   = 0.0   # 所有有效框的"帧级占比"之和
        self._stat_recall_cnt   = 0     # 有效框总数（跨所有batch/帧）
        # 实例级命中率：至少一个采样点落在框内的框数 / 有效框总数
        self._stat_hit_num      = 0     # 命中框数
        self._stat_hit_den      = 0     # 有效框总数
        # 点级虚警率：采样点落在框外的点数 / 全图像素数  (按帧平均)
        self._stat_fa_sum       = 0.0   # 各帧虚警率之和
        self._stat_fa_cnt       = 0     # 帧总数
        self._stat_batch_idx    = 0     # 已处理 batch 数
        # SR/GFLOPS：按 batch 统计，GFLOPS 按图像帧平均（除以 B*T）
        self._stat_sr_sum       = 0.0
        self._stat_sr_cnt       = 0
        self._stat_stage1_gflops_sum = 0.0
        self._stat_total_gflops_sum  = 0.0
        self._stat_gflops_cnt        = 0
        # ==================================================
        # points generate net
        self.net1name=net1name
        temporal_mode = getattr(opt, 'use_tzsconv', 'tzsconv')
        self.net1_feature_channels = feat_channels[0]
        if net1name in ('UNet3DWithNormalConv3D', 'Unet3'):
            self.I2PNet = UNet3DWithNormalConv3D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode, use_final_conv=True,TConvOnly=False)
        elif net1name in ('UNet2DWithNormalConv2D', 'Unet2'):
            self.I2PNet = UNet2DWithNormalConv2D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode, use_final_conv=True)
        elif net1name=='LightWeightedConv3D':
            self.I2PNet = LightWeightedConv3D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='EncoderOnlyConv3DProposalNet':
            self.I2PNet = EncoderOnlyConv3DProposalNet(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode, use_final_conv=True, TConvOnly=False)
        elif net1name=='TOSConvNet':
            self.I2PNet = TOSConvNet(num_channels=3, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode,
                                 use_final_conv=True, use_tzsconv=temporal_mode,
                                 seq_len=img_num)
        elif net1name=='TPConvNet':
            self.I2PNet = TPConvNet(num_channels=3, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode,
                                 use_final_conv=True, use_tzsconv=temporal_mode,
                                 seq_len=img_num, tpdilation=opt.tpdilation, tprepeat=opt.tprepeat)
        elif net1name in ('DynamicTOSConvNet', 'DynamicTOSconvNet'):
            self.I2PNet = DynamicTOSConvNet(num_channels=3, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode,
                                 use_final_conv=True, use_tzsconv=temporal_mode,
                                 seq_len=img_num)
        elif net1name in ('TZSConvNet', 'TZSconvNet'):
            self.I2PNet = TZSConvNet(num_channels=3, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode,
                                 use_final_conv=True, use_tzsconv=temporal_mode,
                                 seq_len=img_num)
        else:
            print('net1name 错误！')
        self.sigmoid = nn.Sigmoid()
        # 稀疏
        self.thresh=thresh
        head_conv=128
        grid_size = np.array([image_size[1], image_size[0], img_num - 1])
        self.points_all = img_num*image_size[0]*image_size[1]
        if  layers==4:
            self.sp_backbone = UNetV2(self.net1_feature_channels, grid_size)
        elif layers==3:
            self.sp_backbone = UNetV2_3(self.net1_feature_channels, grid_size)
        elif layers == 2:
            self.sp_backbone = UNetV2_2(self.net1_feature_channels, grid_size)
        elif layers == 3.5:
            self.sp_backbone = UNetV2_3_32(self.net1_feature_channels, grid_size)
        elif layers == 3.6:
            self.sp_backbone = UNetV2_3_T_nodown(self.net1_feature_channels, grid_size)
        elif layers == 3.61:
            self.sp_backbone = UNetV2_3_T_nodown_v2(self.net1_feature_channels, grid_size, opt=opt)  
        elif layers == 3.62:
            self.sp_backbone = UNetV2_3_T_nodown_v3(self.net1_feature_channels, grid_size)  
        elif layers == 3.7:
            self.sp_backbone = UNetV2_3_T_nodown_maxpool(self.net1_feature_channels, grid_size)
        else:
            raise Exception('Not a valid mode!!!!!')
        head_input_channel = self.sp_backbone.num_point_features
        ###get head conv
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            name_1 = 'subm1'+head
            name_2 = 'subm2'+head
            if head_conv > 0:
                if 'hm' in head:
                    fc = spconv.SparseSequential(
                            spconv.SubMConv3d(head_input_channel, head_conv, 3, padding=1, bias=False, indice_key=name_1),
                        nn.ReLU(),
                        spconv.SubMConv3d(head_conv, classes, 3, padding=1, bias=True, indice_key=name_2),
                        )
                else:
                    fc = spconv.SparseSequential(
                        spconv.SubMConv3d(head_input_channel, head_conv, 3, padding=1, bias=False, indice_key=name_1),
                        nn.ReLU(),
                        spconv.SubMConv3d(head_conv, classes, 3, padding=1, bias=False, indice_key=name_2),
                        )
            else:
                fc = spconv.SubMConv3d(head_input_channel, classes, 3, padding=1, bias=True, indice_key=name_1)
            ###
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            self.__setattr__(head, fc)

            self.sigmoid = nn.Sigmoid()

            # self.tau = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            # self.tau.data.fill_(1)
            # self.conv_std = nn.Sequential(
            #     nn.AdaptiveAvgPool2d([1, 1]),
            #     nn.Conv2d(img_num, img_num, 1),
            #     nn.ReLU(inplace=True)
            # )

            self.relu = nn.ReLU(inplace=True)
    '''
    def get_mask_by_mean_std(self, soft_mask, var_coeff=3, min_thresh=0.01):
        """
        基于soft_mask的均值+方差计算动态阈值，并保证有效点数≥50（不足则取Top50）
        :param soft_mask: 目标概率掩码，shape [B, 1, T, H, W]
        :param var_coeff: 方差系数（控制阈值严格程度，推荐0.3~1.0）
        :param min_thresh: 阈值下限（避免方差为0时阈值过低）
        :return: binary_mask: 最终二值掩码 [B, 1, T, H, W]
        """
        var_coeff = var_coeff if var_coeff is not None else 3
        B, C, T, H, W = soft_mask.shape
        assert C == 1, "soft_mask通道数必须为1"
        device = soft_mask.device
        
        # 1. 展平T/H/W维度，保留batch维度 → [B, 1, T*H*W]
        mask_flat = soft_mask.view(B, 1, -1)  # 展平后便于统计均值/方差和TopK
        
        # 2. 按每个样本计算均值和方差（dim=-1：对展平后的维度计算）
        mask_mean = torch.mean(mask_flat, dim=-1, keepdim=True)  # [B, 1, 1]
        mask_std = torch.std(mask_flat, dim=-1, keepdim=True)    # [B, 1, 1]

        
        # 3. 计算动态阈值：均值 - 系数*标准差（优先保留高于均值的高置信度区域）
        #    若想更严格：均值 + 系数*标准差（仅保留远高于均值的区域）
        dynamic_thresh = mask_mean + var_coeff * mask_std
        # 阈值兜底：不低于min_thresh，避免无意义的极低阈值
        dynamic_thresh = torch.clamp(dynamic_thresh, min=min_thresh, max=1)  # [B, 1, 1]
        
        # 4. 初始按阈值生成二值掩码
        binary_mask_flat = (mask_flat > dynamic_thresh).float()  # [B, 1, T*H*W]
        
        # 5. 统计每个样本的有效点数，不足50则取Top50
        valid_pts_count = torch.sum(binary_mask_flat, dim=-1)  # [B, 1] → 每个样本的有效点数
        
        for b in range(B):
            if valid_pts_count[b] < 50:
                # 该样本有效点不足50，取Top50个最高置信度点
                single_mask_flat = mask_flat[b, 0, :]  # [T*H*W] → 取出当前样本的展平掩码
                # 取Top50的索引（k=50，dim=0）
                _, top100_idx = torch.topk(single_mask_flat, k=50, dim=0, largest=True)
                # 初始化空掩码，仅Top50位置设为1
                top100_mask_flat = torch.zeros_like(single_mask_flat, device=device)
                top100_mask_flat[top100_idx] = 1.0
                # 替换当前样本的掩码
                binary_mask_flat[b, 0, :] = top100_mask_flat
        # if self.print % 50 == 0:
        #     print(torch.sum(binary_mask_flat, dim=-1))
        self.print += 1
        # 6. 恢复掩码形状 [B,1,T,H,W]
        binary_mask = binary_mask_flat.view(B, 1, T, H, W)
        
        return binary_mask
    '''

    # ========== 集成到你的forward函数中 ==========
    def get_mask_by_mean_std(self, soft_mask, var_coeff=3):
        """
        基于soft_mask的空间维度(HW)均值+方差计算动态阈值。
        当阈值筛出的点数少于50时，按每个样本在 T*H*W 范围内补足 top50。
        """
        var_coeff = var_coeff if var_coeff is not None else 3
        B, C, T, H, W = soft_mask.shape
        assert C == 1, "soft_mask通道数必须为1"

        # 对 H, W 求均值/标准差 -> [B, 1, T, 1, 1]
        mask_mean = torch.mean(soft_mask, dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1)
        mask_std = torch.std(soft_mask, dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1)

        # 动态阈值：均值 + 系数 * 标准差
        dynamic_thresh = mask_mean + var_coeff * mask_std
        # dynamic_thresh = 0.3
        # 初始二值掩码 [B, 1, T, H, W]
        binary_mask = (soft_mask > dynamic_thresh).float()

        # 若某个样本有效点不足50，则用该样本全时空范围内的 top50 兜底
        binary_mask_flat = binary_mask.view(B, -1)
        soft_mask_flat = soft_mask.view(B, -1)
        valid_pts_count = torch.sum(binary_mask_flat, dim=-1)
        min_points = min(50, soft_mask_flat.shape[-1])

        for batch_idx in range(B):
            if valid_pts_count[batch_idx] < min_points:
                topk_idx = torch.topk(soft_mask_flat[batch_idx], k=min_points, dim=0, largest=True).indices
                binary_mask_flat[batch_idx].zero_()
                binary_mask_flat[batch_idx, topk_idx] = 1.0

        binary_mask = binary_mask_flat.view(B, 1, T, H, W)
        return binary_mask

    # ------------------------------------------------------------------
    # 第一阶段检测质量指标（仅非训练阶段调用）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_and_print_stage1_metrics(self, binary_mask, bboxes, b, t, h, w, device,
                                          sampling_rate=None, flops_dict=None):
        """
        计算并实时打印第一阶段质量、采样率和计算量指标，同时累计数据集总平均。

        参数
        ----
        binary_mask : Tensor [B, 1, T, H, W]  0/1 float，阈值过滤后的采样点图
        bboxes      : Tensor [B, T, 512, 6]   真值框，前4位为 x1 y1 x2 y2（0值框为padding）
        b, t, h, w  : int，batch_size / 帧数 / 高 / 宽

        指标定义
        --------
        1. 第一阶段命中覆盖率（coverage_rate）—— 点级，有效点在框内的平均覆盖占比
             对每个有效框计算：该帧采样点中落在框内的点数 / 框内像素数
             取所有有效框的平均值作为当前 batch 的指标。

        2. 第一阶段命中率（hit_rate）—— 实例级
             有效框中至少有 1 个采样点落在框内 → 命中
             命中框数 / 有效框总数。

        3. 点级虚警率（false_alarm_rate）—— 点级
             对每帧计算：采样点中落在所有真值框**外**的点数 / 全图像素数（H×W）
             取所有帧的平均值作为当前 batch 的指标。

        4. 采样率（SR）—— 当前 batch 的采样点数 / B×T×H×W。

        5. 平均计算量（GFLOPS）—— hook 估算 FLOPs / 1e9 / (B×T)。
             Stage1 包含 I2PNet + ACS；Total 包含模型已分类组件之和。
        """
        # ---- 准备 binary_mask：压缩通道维度 → [B, T, H, W] ----
        bm = binary_mask.squeeze(1)   # [B, T, H, W]  float 0/1

        # ---- 准备真值框：转移到 CPU 做循环，避免 GPU 碎片 ----
        bboxes_cpu = bboxes.cpu().float()   # [B, T, 512, 6]
        bm_cpu     = bm.detach().cpu()      # [B, T, H, W]

        # ---- 批级累计变量 ----
        batch_recall_sum  = 0.0
        batch_recall_cnt  = 0          # 有效框计数
        batch_hit_num     = 0          # 命中框数
        batch_hit_den     = 0          # 有效框总数
        batch_fa_sum      = 0.0        # 各帧虚警率之和
        batch_fa_cnt      = 0          # 帧计数

        for bi in range(b):
            for ti in range(t):
                frame_mask = bm_cpu[bi, ti]        # [H, W]  0/1 float
                frame_boxes = bboxes_cpu[bi, ti]   # [512, 6]

                # 筛选有效框（任意坐标非零）
                valid_mask_box = (frame_boxes[:, :4].abs().sum(dim=1) > 0)  # [512]
                valid_boxes = frame_boxes[valid_mask_box]                    # [K, 6]
                K = valid_boxes.shape[0]

                # 构建全帧的"框内像素"联合 mask（用于虚警率）
                frame_gt_union = torch.zeros(h, w, dtype=torch.float32)

                total_pixels = h * w  # 全图像素数

                for ki in range(K):
                    x1, y1, x2, y2 = valid_boxes[ki, :4]
                    # 坐标 clamp 到合法范围，并转 int
                    x1i = int(x1.clamp(0, w - 1).round())
                    y1i = int(y1.clamp(0, h - 1).round())
                    x2i = int(x2.clamp(0, w).round())
                    y2i = int(y2.clamp(0, h).round())

                    if x2i <= x1i or y2i <= y1i:
                        # 无效尺寸框，跳过
                        K -= 1
                        continue

                    # 框内 patch 的采样点
                    patch_mask = frame_mask[y1i:y2i, x1i:x2i]   # [dh, dw]
                    box_pixels  = (y2i - y1i) * (x2i - x1i)     # 框内像素数

                    sampled_in_box = int(patch_mask.sum().item())

                    # --- 点级召回：框内采样点 / 框内像素 ---
                    recall_ratio = sampled_in_box / box_pixels if box_pixels > 0 else 0.0
                    batch_recall_sum += recall_ratio
                    batch_recall_cnt += 1

                    # --- 实例级命中：至少 1 个点落在框内 ---
                    batch_hit_num += 1 if sampled_in_box > 0 else 0
                    batch_hit_den += 1

                    # 累计联合 GT mask（用于虚警率）
                    frame_gt_union[y1i:y2i, x1i:x2i] = 1.0

                # --- 点级虚警率：框外采样点 / 全图像素 ---
                # 框外 = 采样点 & (1 - gt_union)
                outside_pts = (frame_mask * (1.0 - frame_gt_union)).sum().item()
                fa_rate = outside_pts / total_pixels if total_pixels > 0 else 0.0
                batch_fa_sum += fa_rate
                batch_fa_cnt += 1

        # ---- 当前 batch 均值 ----
        cur_recall  = batch_recall_sum / batch_recall_cnt if batch_recall_cnt > 0 else 0.0
        cur_hit     = batch_hit_num    / batch_hit_den    if batch_hit_den    > 0 else 0.0
        cur_fa      = batch_fa_sum     / batch_fa_cnt     if batch_fa_cnt     > 0 else 0.0

        # ---- 累计到数据集统计量 ----
        self._stat_recall_sum  += batch_recall_sum
        self._stat_recall_cnt  += batch_recall_cnt
        self._stat_hit_num     += batch_hit_num
        self._stat_hit_den     += batch_hit_den
        self._stat_fa_sum      += batch_fa_sum
        self._stat_fa_cnt      += batch_fa_cnt
        self._stat_batch_idx   += 1

        # ---- SR / GFLOPS 累计 ----
        cur_sr = None
        if sampling_rate is not None:
            cur_sr = float(sampling_rate)
            self._stat_sr_sum += cur_sr
            self._stat_sr_cnt += 1

        cur_stage1_gflops = None
        cur_total_gflops = None
        if flops_dict is not None:
            frame_divisor = max(float(b * t), 1.0)
            stage1_flops = float(flops_dict.get('I2PNet', 0.0)) + float(flops_dict.get('ACS', 0.0))
            total_flops = sum(float(flops_dict.get(name, 0.0)) for name in self.runtime_component_names)
            cur_stage1_gflops = stage1_flops / 1e9 / frame_divisor
            cur_total_gflops = total_flops / 1e9 / frame_divisor
            self._stat_stage1_gflops_sum += cur_stage1_gflops
            self._stat_total_gflops_sum += cur_total_gflops
            self._stat_gflops_cnt += 1

        # ---- 数据集累计均值 ----
        ds_recall = self._stat_recall_sum / self._stat_recall_cnt if self._stat_recall_cnt > 0 else 0.0
        ds_hit    = self._stat_hit_num    / self._stat_hit_den    if self._stat_hit_den    > 0 else 0.0
        ds_fa     = self._stat_fa_sum     / self._stat_fa_cnt     if self._stat_fa_cnt     > 0 else 0.0
        ds_sr     = self._stat_sr_sum     / self._stat_sr_cnt     if self._stat_sr_cnt     > 0 else None
        ds_stage1_gflops = (
            self._stat_stage1_gflops_sum / self._stat_gflops_cnt
            if self._stat_gflops_cnt > 0 else None
        )
        ds_total_gflops = (
            self._stat_total_gflops_sum / self._stat_gflops_cnt
            if self._stat_gflops_cnt > 0 else None
        )

        cur_profile = ""
        if cur_sr is not None:
            cur_profile += f"  SR={cur_sr*100:.4f}%"
        if cur_stage1_gflops is not None:
            cur_profile += (
                f"  Stage1_GFLOPS={cur_stage1_gflops:.6f}"
                f"  Total_GFLOPS={cur_total_gflops:.6f}"
            )

        ds_profile = ""
        if ds_sr is not None:
            ds_profile += f"  SR={ds_sr*100:.4f}%"
        if ds_stage1_gflops is not None:
            ds_profile += (
                f"  Stage1_GFLOPS={ds_stage1_gflops:.6f}"
                f"  Total_GFLOPS={ds_total_gflops:.6f}"
            )

        # ---- 打印 ----
        print(
            f"[Stage1 Metrics] Batch {self._stat_batch_idx:04d} | "
            f"当前: 命中率={cur_hit*100:.2f}%  命中覆盖率={cur_recall*100:.2f}%  "
            f"虚警率={cur_fa*100:.4f}%{cur_profile} | "
            f"数据集累计: 命中率={ds_hit*100:.2f}%  命中覆盖率={ds_recall*100:.2f}%  "
            f"虚警率={ds_fa*100:.4f}%{ds_profile}"
        )

    def print_dataset_stage1_summary(self):
        """推理结束后主动调用，打印整个数据集的汇总指标。"""
        ds_recall = self._stat_recall_sum / self._stat_recall_cnt if self._stat_recall_cnt > 0 else 0.0
        ds_hit    = self._stat_hit_num    / self._stat_hit_den    if self._stat_hit_den    > 0 else 0.0
        ds_fa     = self._stat_fa_sum     / self._stat_fa_cnt     if self._stat_fa_cnt     > 0 else 0.0
        ds_sr     = self._stat_sr_sum     / self._stat_sr_cnt     if self._stat_sr_cnt     > 0 else None
        ds_stage1_gflops = (
            self._stat_stage1_gflops_sum / self._stat_gflops_cnt
            if self._stat_gflops_cnt > 0 else None
        )
        ds_total_gflops = (
            self._stat_total_gflops_sum / self._stat_gflops_cnt
            if self._stat_gflops_cnt > 0 else None
        )
        print("=" * 80)
        print(f"[Stage1 Dataset Summary]  共处理 {self._stat_batch_idx} 个 batch，"
              f"有效框 {self._stat_hit_den} 个，帧 {self._stat_fa_cnt} 帧")
        print(f"  第一阶段命中率（hit_rate）        : {ds_hit*100:.4f}%")
        print(f"  第一阶段命中覆盖率（coverage_rate）: {ds_recall*100:.4f}%")
        print(f"  点级虚警率（false_alarm）         : {ds_fa*100:.6f}%")
        if ds_sr is not None:
            print(f"  采样率（SR）                      : {ds_sr*100:.6f}%")
        if ds_stage1_gflops is not None:
            print(f"  第一阶段平均计算量（Stage1 GFLOPS）: {ds_stage1_gflops:.6f}")
            print(f"  模型总平均计算量（Total GFLOPS）   : {ds_total_gflops:.6f}")
        print("=" * 80)
    # ------------------------------------------------------------------


    def reset_runtime_stats(self):
        self._runtime_sum = {name: 0.0 for name in self.runtime_component_names}
        self._flops_sum = {name: 0.0 for name in self.runtime_component_names}
        self._runtime_count = 0

    def get_runtime_stats(self, per_frame_divisor=1):
        if self._runtime_count == 0:
            return {}
        per_frame_divisor = max(float(per_frame_divisor), 1.0)
        return {
            name: self._runtime_sum[name] / self._runtime_count / per_frame_divisor
            for name in self.runtime_component_names
        }

    def get_flops_stats(self, per_frame_divisor=1):
        if self._runtime_count == 0:
            return {}
        per_frame_divisor = max(float(per_frame_divisor), 1.0)
        return {
            name: self._flops_sum[name] / self._runtime_count / per_frame_divisor
            for name in self.runtime_component_names
        }

    def get_component_display_name(self, component_name):
        return self.component_display_names.get(component_name, component_name)

    def get_component_stats(self, per_frame_divisor=1):
        runtime_stats = self.get_runtime_stats(per_frame_divisor=per_frame_divisor)
        flops_stats = self.get_flops_stats(per_frame_divisor=per_frame_divisor)
        param_stats = self.get_component_param_stats()
        stats = {}
        total_runtime = sum(float(runtime_stats.get(name, 0.0)) for name in self.runtime_component_names)
        total_flops = sum(float(flops_stats.get(name, 0.0)) for name in self.runtime_component_names)
        total_params = self.get_total_param_count()
        for name in self.runtime_component_names:
            runtime = float(runtime_stats.get(name, 0.0))
            flops = float(flops_stats.get(name, 0.0))
            params = int(param_stats.get(name, 0))
            stats[name] = {
                'display_name': self.get_component_display_name(name),
                'runtime': runtime,
                'runtime_percent': runtime / total_runtime * 100.0 if total_runtime > 0 else 0.0,
                'flops': flops,
                'gflops': flops / 1e9,
                'params': params,
                'params_percent': params / total_params * 100.0 if total_params > 0 else 0.0,
            }
        stats['Total'] = {
            'display_name': self.get_component_display_name('Total'),
            'runtime': total_runtime,
            'runtime_percent': 100.0 if total_runtime > 0 else 0.0,
            'flops': total_flops,
            'gflops': total_flops / 1e9,
            'params': total_params,
            'params_percent': 100.0 if total_params > 0 else 0.0,
        }
        return stats

    def print_component_stats_summary(self, per_frame_divisor=None, mark_printed=True):
        if self._runtime_count == 0:
            return
        if per_frame_divisor is None:
            per_frame_divisor = self._runtime_frame_divisor
        stats = self.get_component_stats(per_frame_divisor=per_frame_divisor)
        print('=' * 80)
        print('[Five-Stage Component Profiling Summary]')
        print('  Stage                 runtime(s)  runtime(%)  Params      GFLOPS')
        for component_name in self.runtime_component_names:
            item = stats[component_name]
            print(
                f"  {item['display_name']:<20} {item['runtime']:.6f}    "
                f"{item['runtime_percent']:>7.2f}%  {item['params']:<10d} {item['gflops']:.6f}"
            )
        total = stats['Total']
        print(
            f"  {total['display_name']:<20} {total['runtime']:.6f}    "
            f"{total['runtime_percent']:>7.2f}%  {total['params']:<10d} {total['gflops']:.6f}"
        )
        if self._unclassified_param_count:
            print(f"  UnclassifiedParams: {self._unclassified_param_count}")
        print('=' * 80)
        if mark_printed:
            self._component_profile_printed = True

    def _finalize_component_profile(self):
        try:
            if not self.use_runtime:
                return
            if self._component_profile_printed:
                return
            if self._runtime_count == 0:
                return
            self.print_component_stats_summary(mark_printed=True)
        except Exception as exc:
            print(f"[Component Profiling Summary] 打印失败: {exc}")

    def _runtime_enabled(self):
        return self.use_runtime and (not self.training)

    def _flops_enabled(self, force=False):
        return (self.use_runtime or force) and (not self.training)

    def _is_mfe_local_name(self, local_name):
        for prefix in self._mfe_module_prefixes:
            if local_name == prefix or local_name.startswith(prefix + '.'):
                return True
        return False

    def _classify_module_component(self, module_name):
        if module_name.startswith('I2PNet.'):
            return 'I2PNet'
        if module_name.startswith('sp_backbone.'):
            local_name = module_name[len('sp_backbone.'):]
            return 'MFE' if self._is_mfe_local_name(local_name) else 'BackboneNoMFE'
        for head in self.heads:
            if module_name == head or module_name.startswith(head + '.'):
                return 'DetHead'
        if module_name == 'sigmoid' or module_name == 'conv_std' or module_name.startswith('conv_std.'):
            return None
        return None

    def _classify_parameter_component(self, param_name):
        if param_name.startswith('I2PNet.'):
            return 'I2PNet'
        if param_name.startswith('sp_backbone.'):
            local_name = param_name[len('sp_backbone.'):]
            return 'MFE' if self._is_mfe_local_name(local_name) else 'BackboneNoMFE'
        for head in self.heads:
            if param_name.startswith(head + '.'):
                return 'DetHead'
        if param_name == 'tau' or param_name.startswith('conv_std.'):
            return 'ACS'
        return None

    def get_component_param_stats(self, refresh=False):
        if self._component_param_cache is not None and not refresh:
            return dict(self._component_param_cache)
        param_stats = {name: 0 for name in self.runtime_component_names}
        unclassified = 0
        for param_name, param in self.named_parameters():
            component_name = self._classify_parameter_component(param_name)
            if component_name is None:
                unclassified += int(param.numel())
                continue
            param_stats[component_name] += int(param.numel())
        self._unclassified_param_count = unclassified
        self._component_param_cache = dict(param_stats)
        return param_stats

    def get_total_param_count(self):
        return count_parameters(self)

    @staticmethod
    def _sync_device(device):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def _runtime_start(self, device):
        self._sync_device(device)
        return time.perf_counter()

    def _runtime_stop(self, start_time, device):
        self._sync_device(device)
        return time.perf_counter() - start_time

    def _is_top_level_mfe_runtime_module(self, module_name):
        if not module_name.startswith('sp_backbone.'):
            return False
        local_name = module_name[len('sp_backbone.'):]
        if '.' in local_name:
            return False
        return local_name in (
            'shortcut1', 'shortcut2', 'shortcut3',
            'shortcut1fusion', 'shortcut2fusion', 'shortcut3fusion',
        )

    def _start_mfe_runtime_capture(self, device):
        if not self._runtime_enabled():
            return None, None
        meter = {'MFE': 0.0}
        handles = []

        def _pre_hook(module, inputs):
            module._codex_runtime_start = self._runtime_start(device)

        def _post_hook(module, inputs, output):
            start_time = getattr(module, '_codex_runtime_start', None)
            if start_time is None:
                return
            meter['MFE'] += self._runtime_stop(start_time, device)
            module._codex_runtime_start = None

        for module_name, module in self.named_modules():
            if not self._is_top_level_mfe_runtime_module(module_name):
                continue
            handles.append(module.register_forward_pre_hook(_pre_hook))
            handles.append(module.register_forward_hook(_post_hook))
        return meter, handles

    @staticmethod
    def _stop_runtime_capture(handles):
        if handles is None:
            return
        for handle in handles:
            handle.remove()

    def _new_runtime_dict(self):
        return {name: 0.0 for name in self.runtime_component_names}

    def _merge_runtime_dicts(self, *runtime_dicts):
        merged = self._new_runtime_dict()
        for runtime_dict in runtime_dicts:
            if runtime_dict is None:
                continue
            for name in self.runtime_component_names:
                merged[name] += float(runtime_dict.get(name, 0.0))
        return merged

    def _update_runtime_stats(self, runtime_dict, flops_dict=None):
        if runtime_dict is None and flops_dict is None:
            return
        if runtime_dict is not None:
            for name in self.runtime_component_names:
                self._runtime_sum[name] += float(runtime_dict.get(name, 0.0))
        if flops_dict is not None:
            for name in self.runtime_component_names:
                self._flops_sum[name] += float(flops_dict.get(name, 0.0))
        self._runtime_count += 1

    @staticmethod
    def _extract_feature_tensor(obj):
        return extract_feature_tensor(obj)

    def _estimate_module_flops(self, module, inputs, output):
        return estimate_module_flops(module, inputs, output)

    def _start_flops_capture(self, force=False):
        if not self._flops_enabled(force=force):
            return None, None
        flops_dict = self._new_runtime_dict()
        handles = []

        def _make_hook(component_name):
            def _hook(module, inputs, output):
                flops_dict[component_name] += self._estimate_module_flops(module, inputs, output)
            return _hook

        attention_prefixes = []
        for module_name, module in self.named_modules():
            class_name = module.__class__.__name__
            if "SparseSymmetricCosineAttention" in class_name or "TripletMotionConsistency" in class_name:
                attention_prefixes.append(module_name)

        def _skip_attention_child(module_name):
            for prefix in attention_prefixes:
                if not prefix or not module_name.startswith(prefix + '.'):
                    continue
                suffix = module_name[len(prefix) + 1:]
                # Parent TAA hook estimates attention math; keep only its optional conv branch.
                return not (suffix == 'conv' or suffix.startswith('conv.'))
            return False

        for module_name, module in self.named_modules():
            if module_name == '':
                continue
            if _skip_attention_child(module_name):
                continue
            component_name = self._classify_module_component(module_name)
            if component_name is None:
                continue
            handles.append(module.register_forward_hook(_make_hook(component_name)))
        return flops_dict, handles

    @staticmethod
    def _stop_flops_capture(handles):
        if handles is None:
            return
        for handle in handles:
            handle.remove()

    @staticmethod
    def _estimate_sigmoid_flops(tensor):
        return estimate_sigmoid_flops(tensor)

    def _estimate_acs_flops(self, soft_mask):
        if soft_mask is None:
            return 0.0
        b, c, t, h, w = soft_mask.shape
        elements = float(b * c * t * h * w)
        frames = float(b * c * t)
        flops = 0.0
        flops += self._estimate_sigmoid_flops(soft_mask)
        flops += elements
        flops += 3.0 * elements
        flops += 4.0 * frames
        flops += elements
        return flops

    def _get_softmask_vis_context(self, batch=None):
        if isinstance(batch, dict):
            save_dir = batch.get("softmask_save_dir")
            frame_names = batch.get("softmask_frame_names")
            video_name = batch.get("softmask_video_name")
            if isinstance(save_dir, (list, tuple)):
                save_dir = save_dir[0] if len(save_dir) > 0 else None
            if save_dir is not None:
                save_dir = str(save_dir)
                if video_name is not None:
                    save_dir = os.path.join(save_dir, str(video_name))
                return save_dir, frame_names

        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            while frame is not None:
                local_vars = frame.f_locals
                save_mat_folder = local_vars.get("save_mat_folder")
                save_mat_path_upper = local_vars.get("save_mat_path_upper")
                patch_ims = local_vars.get("patch_ims")
                if save_mat_folder is not None:
                    video_name = os.path.basename(str(save_mat_folder))
                    if save_mat_path_upper is None:
                        save_mat_path_upper = os.path.dirname(str(save_mat_folder))
                    save_dir = os.path.join(str(save_mat_path_upper), "softmask_vis", video_name)
                    return save_dir, patch_ims
                frame = frame.f_back
        finally:
            del frame

        return "net1_softmask_vis", None

    def _save_net1_softmask_vis(self, net1_output, batch=None, max_images=10):
        save_dir, frame_names = self._get_softmask_vis_context(batch)
        counters = getattr(self, "_softmask_vis_counter_by_dir", None)
        if counters is None:
            counters = {}
            self._softmask_vis_counter_by_dir = counters
        saved_count = counters.get(save_dir, 0)
        if saved_count >= max_images:
            return

        with torch.no_grad():
            soft_mask = torch.sigmoid(net1_output.detach())
            if soft_mask.shape[1] > 1:
                soft_mask = soft_mask.mean(dim=1, keepdim=True)

        os.makedirs(save_dir, exist_ok=True)
        bsz, _, frames, _, _ = soft_mask.shape
        for b_i in range(bsz):
            for t_i in range(frames):
                if saved_count >= max_images:
                    counters[save_dir] = saved_count
                    return
                if frame_names is not None and t_i < len(frame_names):
                    stem = os.path.splitext(os.path.basename(str(frame_names[t_i])))[0]
                    if bsz > 1:
                        stem = f"{stem}_b{b_i}"
                else:
                    stem = f"{saved_count:06d}_b{b_i}_t{t_i}_softmask"
                imgpath = os.path.join(save_dir, stem + ".png")
                save_net1_output(soft_mask[b_i, 0, t_i], imgpath, mode="255")
                saved_count += 1
        counters[save_dir] = saved_count

    def _get_taa_heatmap_vis_context(self, batch=None):
        softmask_dir, frame_names = self._get_softmask_vis_context(batch)
        softmask_dir = str(softmask_dir)
        norm_dir = softmask_dir.rstrip(os.sep)
        video_name = os.path.basename(norm_dir) or "unknown_video"
        softmask_parent = os.path.dirname(norm_dir)
        if os.path.basename(softmask_parent) == "softmask_vis":
            root_dir = os.path.dirname(softmask_parent)
        else:
            root_dir = os.path.dirname(norm_dir) or "."
        return os.path.join(root_dir, "heatmap_taa"), video_name, frame_names

    def _taa_heatmap_vis_has_budget(self, batch=None, max_images=10):
        root_dir, video_name, _ = self._get_taa_heatmap_vis_context(batch)
        key = os.path.join(root_dir, video_name)
        counters = getattr(self, "_taa_heatmap_vis_counter_by_video", None)
        return counters is None or counters.get(key, 0) < max_images


    @staticmethod
    def _taa_heatmap_frame_stem(frame_names, t_i, saved_count, b_i, bsz):
        stem = None
        if frame_names is not None and t_i < len(frame_names):
            stem = os.path.splitext(os.path.basename(str(frame_names[t_i])))[0]
        if not stem:
            stem = f"{saved_count + 1:06d}"
        elif stem.isdigit():
            stem = f"{int(stem):06d}"
        if bsz > 1:
            stem = f"{stem}_b{b_i}"
        return stem


    @staticmethod
    def _taa_snapshot_shape(snapshot):
        shape = list(snapshot.get("spatial_shape") or [])
        while len(shape) < 3:
            shape.append(0)
        indices = snapshot.get("indices")
        if indices is not None and torch.is_tensor(indices) and indices.numel() > 0:
            max_coords = indices.long().max(dim=0)[0]
            shape[0] = max(int(shape[0]), int(max_coords[1].item()) + 1)
            shape[1] = max(int(shape[1]), int(max_coords[2].item()) + 1)
            shape[2] = max(int(shape[2]), int(max_coords[3].item()) + 1)
        return max(int(shape[0]), 1), max(int(shape[1]), 1), max(int(shape[2]), 1)

    def _sparse_snapshot_to_frame_map(self, snapshot, b_i, t_i, mode):
        _, h, w = self._taa_snapshot_shape(snapshot)
        heat = np.full((h, w), np.nan, dtype=np.float32)
        features = snapshot.get("features")
        indices = snapshot.get("indices")
        if features is None or indices is None or not torch.is_tensor(features) or not torch.is_tensor(indices):
            return heat
        if features.numel() == 0 or indices.numel() == 0:
            return heat

        indices = indices.long()
        frame_mask = (indices[:, 0] == int(b_i)) & (indices[:, 1] == int(t_i))
        if not bool(frame_mask.any().item()):
            return heat

        frame_features = features[frame_mask].float()
        if mode == "max":
            values = frame_features.max(dim=1)[0]
        else:
            values = frame_features.mean(dim=1)

        frame_coords = indices[frame_mask]
        ys = frame_coords[:, 2].clamp(0, h - 1).cpu().numpy()
        xs = frame_coords[:, 3].clamp(0, w - 1).cpu().numpy()
        heat[ys, xs] = values.detach().cpu().numpy().astype(np.float32)
        return heat


    @staticmethod
    def _taa_heatmap_vmax(*frame_maps):
        value_parts = []
        for frame_map in frame_maps:
            if frame_map is None:
                continue
            valid_mask = np.isfinite(frame_map)
            if valid_mask.any():
                value_parts.append(frame_map[valid_mask].astype(np.float32))
        if not value_parts:
            return 1.0
        values = np.concatenate(value_parts)
        return max(float(np.nanmax(values)), 1e-6)

    @staticmethod
    def _colorize_taa_heatmap(frame_map, vmax, valid_mask=None):
        lut = Img2PointsSmallObjectDetection._taa_heatmap_lut()
        vmax = max(float(vmax), 1e-6)
        norm = np.nan_to_num(frame_map, nan=0.0, posinf=vmax, neginf=0.0) / vmax
        gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        color = lut[gray]
        if valid_mask is not None:
            color[~valid_mask] = lut[0]
        return color

    @staticmethod
    def _save_taa_heatmap_pair(before_map, after_map, before_path, after_path):
        valid_before = np.isfinite(before_map)
        valid_after = np.isfinite(after_map)
        vmax = Img2PointsSmallObjectDetection._taa_heatmap_vmax(before_map, after_map)

        os.makedirs(os.path.dirname(before_path), exist_ok=True)
        os.makedirs(os.path.dirname(after_path), exist_ok=True)
        cv2.imwrite(before_path, Img2PointsSmallObjectDetection._colorize_taa_heatmap(before_map, vmax, valid_before))
        cv2.imwrite(after_path, Img2PointsSmallObjectDetection._colorize_taa_heatmap(after_map, vmax, valid_after))
        return vmax

    @staticmethod
    def _save_src_heatmap(src_map, src_path, vmax):
        os.makedirs(os.path.dirname(src_path), exist_ok=True)
        cv2.imwrite(src_path, Img2PointsSmallObjectDetection._colorize_taa_heatmap(src_map, vmax))

    @staticmethod
    def _taa_src_root_dir(taa_root_dir):
        norm_dir = str(taa_root_dir).rstrip(os.sep)
        if os.path.basename(norm_dir) == "heatmap_taa":
            return os.path.join(os.path.dirname(norm_dir), "heatmap_src")
        return norm_dir + "_src"

    @staticmethod
    def _prepare_src_heatmap_array(source_heatmap):
        if source_heatmap is None:
            return None
        if torch.is_tensor(source_heatmap):
            with torch.no_grad():
                src = source_heatmap.detach()
                if src.dim() != 5:
                    return None
                if src.shape[1] > 1:
                    src = src.mean(dim=1, keepdim=True)
                return src[:, 0].float().cpu().numpy().astype(np.float32)
        src = np.asarray(source_heatmap, dtype=np.float32)
        if src.ndim == 5:
            if src.shape[1] > 1:
                src = src.mean(axis=1, keepdims=True)
            src = src[:, 0]
        if src.ndim != 4:
            return None
        return src.astype(np.float32)

    @staticmethod
    def _src_heatmap_to_frame_map(source_heatmap_np, b_i, t_i, target_shape):
        if source_heatmap_np is None:
            return None
        if b_i >= source_heatmap_np.shape[0] or t_i >= source_heatmap_np.shape[1]:
            return None
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        frame_map = np.nan_to_num(source_heatmap_np[b_i, t_i], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if frame_map.shape != (target_h, target_w):
            interpolation = cv2.INTER_AREA if frame_map.shape[0] >= target_h and frame_map.shape[1] >= target_w else cv2.INTER_LINEAR
            frame_map = cv2.resize(frame_map, (target_w, target_h), interpolation=interpolation).astype(np.float32)
        return frame_map

    @staticmethod
    def _taa_heatmap_lut():
        cached_lut = getattr(Img2PointsSmallObjectDetection, "_taa_heatmap_lut_cache", None)
        if cached_lut is not None:
            return cached_lut
        stops = np.asarray([
            [0.00, 128, 32, 0],
            [0.18, 255, 96, 0],
            [0.38, 255, 210, 0],
            [0.58, 80, 230, 255],
            [0.78, 0, 140, 255],
            [1.00, 0, 0, 255],
        ], dtype=np.float32)
        sample_points = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        lut = np.empty((256, 3), dtype=np.uint8)
        for channel in range(3):
            lut[:, channel] = np.interp(sample_points, stops[:, 0], stops[:, channel + 1]).astype(np.uint8)
        Img2PointsSmallObjectDetection._taa_heatmap_lut_cache = lut
        return lut

    def _save_taa_heatmap_vis(self, taa_skip_features, batch=None, source_heatmap=None, max_images=10):
        if not taa_skip_features:
            return
        root_dir, video_name, frame_names = self._get_taa_heatmap_vis_context(batch)
        src_root_dir = self._taa_src_root_dir(root_dir)
        source_heatmap_np = self._prepare_src_heatmap_array(source_heatmap)
        key = os.path.join(root_dir, video_name)
        counters = getattr(self, "_taa_heatmap_vis_counter_by_video", None)
        if counters is None:
            counters = {}
            self._taa_heatmap_vis_counter_by_video = counters
        saved_count = counters.get(key, 0)
        if saved_count >= max_images:
            return

        skip_items = sorted(taa_skip_features, key=lambda item: int(item.get("skip", 0)))
        bsz = 1
        frames = 1
        for item in skip_items:
            for phase in ("before", "after"):
                snapshot = item.get(phase)
                if not snapshot:
                    continue
                cur_t, _, _ = self._taa_snapshot_shape(snapshot)
                frames = max(frames, cur_t)
                bsz = max(bsz, int(snapshot.get("batch_size", 1)))
        if frame_names is not None:
            frames = max(frames, len(frame_names))

        for b_i in range(bsz):
            for t_i in range(frames):
                if saved_count >= max_images:
                    counters[key] = saved_count
                    return
                frame_stem = self._taa_heatmap_frame_stem(frame_names, t_i, saved_count, b_i, bsz)
                for mode in ("max", "mean"):
                    for item in skip_items:
                        skip = int(item.get("skip", 0))
                        before_snapshot = item.get("before")
                        after_snapshot = item.get("after")
                        if not before_snapshot or not after_snapshot:
                            continue
                        before_map = self._sparse_snapshot_to_frame_map(before_snapshot, b_i, t_i, mode)
                        after_map = self._sparse_snapshot_to_frame_map(after_snapshot, b_i, t_i, mode)
                        save_dir = os.path.join(root_dir, mode, video_name, f"skip{skip}")
                        before_path = os.path.join(save_dir, f"{frame_stem}_0before.png")
                        after_path = os.path.join(save_dir, f"{frame_stem}_1after.png")
                        vmax = self._save_taa_heatmap_pair(before_map, after_map, before_path, after_path)
                        src_map = self._src_heatmap_to_frame_map(source_heatmap_np, b_i, t_i, before_map.shape)
                        if src_map is not None:
                            src_dir = os.path.join(src_root_dir, mode, video_name, f"skip{skip}")
                            src_path = os.path.join(src_dir, f"{frame_stem}_src.png")
                            self._save_src_heatmap(src_map, src_path, vmax)
                saved_count += 1
        counters[key] = saved_count


    @staticmethod
    def _taa_snapshot_width(snapshot):
        _, _, width = Img2PointsSmallObjectDetection._taa_snapshot_shape(snapshot)
        return width


    @staticmethod
    def _merge_taa_snapshot_width(left_snapshot, right_snapshot):
        if left_snapshot is None:
            return right_snapshot
        if right_snapshot is None:
            return left_snapshot
        width_offset = Img2PointsSmallObjectDetection._taa_snapshot_width(left_snapshot)
        right_indices = right_snapshot["indices"].clone()
        if right_indices.numel() > 0:
            right_indices[:, 3] += int(width_offset)
        left_shape = Img2PointsSmallObjectDetection._taa_snapshot_shape(left_snapshot)
        right_shape = Img2PointsSmallObjectDetection._taa_snapshot_shape(right_snapshot)
        return {
            "features": torch.cat([left_snapshot["features"], right_snapshot["features"]], dim=0),
            "indices": torch.cat([left_snapshot["indices"], right_indices], dim=0),
            "spatial_shape": (max(left_shape[0], right_shape[0]), max(left_shape[1], right_shape[1]), left_shape[2] + right_shape[2]),
            "batch_size": max(int(left_snapshot.get("batch_size", 1)), int(right_snapshot.get("batch_size", 1))),
        }


    @staticmethod
    def _merge_taa_heatmap_features(left_items, right_items):
        if not left_items:
            return right_items
        if not right_items:
            return left_items
        right_by_skip = {int(item.get("skip", 0)): item for item in right_items}
        merged = []
        for left_item in left_items:
            skip = int(left_item.get("skip", 0))
            right_item = right_by_skip.pop(skip, None)
            if right_item is None:
                merged.append(left_item)
                continue
            merged.append({
                "skip": skip,
                "before": Img2PointsSmallObjectDetection._merge_taa_snapshot_width(left_item.get("before"), right_item.get("before")),
                "after": Img2PointsSmallObjectDetection._merge_taa_snapshot_width(left_item.get("after"), right_item.get("after")),
            })
        merged.extend(right_by_skip.values())
        return merged

    # ############ 原始 forward（非滑窗版本，已切换至下方滑窗版本）############
    # def forward(self, batch):
    #     device = batch['input'].device
    #     b, c, t, h, w = batch['input'].shape
    #     if self.net1name == 'UNet3DwithZZLB':
    #         voxel_features = self.I2PNet(batch)
    #     else:
    #         voxel_features = self.I2PNet(batch['input'],)
    #     voxel_features_ori = voxel_features.clone()
    #     soft_mask = self.sigmoid(voxel_features)
    #     binary_mask = self.get_mask_by_mean_std(
    #         soft_mask=soft_mask, var_coeff=self.thresh, min_thresh=0.01)
    #     coords = torch.nonzero(binary_mask.squeeze(1)).contiguous()
    #     total_points = b * t * h * w
    #     sampled_points = coords.shape[0]
    #     sampling_rate = sampled_points / total_points if total_points > 0 else 0
    #     batch_idx = coords[:, 0]; t_idx = coords[:, 1]
    #     h_idx = coords[:, 2];     w_idx = coords[:, 3]
    #     flattened_indices = batch_idx*t*h*w + t_idx*h*w + h_idx*w + w_idx
    #     batch_dict = {}
    #     batch_dict['voxel_features'] = voxel_features.reshape(b*t*h*w, 1)[flattened_indices]
    #     batch_dict['voxel_coords'] = coords.to(device)
    #     batch_dict['batch_size'] = b
    #     sp_backbone_out = self.sp_backbone(batch_dict)
    #     z = {}
    #     for head in self.heads:
    #         input_sp_tensor = sp_backbone_out['encoded_spconv_tensor']
    #         out_h = self.__getattr__(head)(input_sp_tensor)
    #         if 'hm' in head:
    #             out_h = replace_feature(out_h, self.sigmoid(out_h.features))
    #             spatial_features = torch.clamp(out_h.dense(), min=1e-4, max=1-1e-4)
    #         else:
    #             spatial_features = out_h.dense()
    #         z[head] = spatial_features
    #     z['hm_large_heatmap'] = voxel_features_ori
    #     z['voxel_coords'] = batch_dict['voxel_coords']
    #     z['soft_mask'] = soft_mask
    #     z['sampling_rate'] = torch.tensor(sampling_rate, device=device)
    #     if not self.training and 'bboxes' in batch:
    #         self._compute_and_print_stage1_metrics(
    #             binary_mask=binary_mask, bboxes=batch['bboxes'],
    #             b=b, t=t, h=h, w=w, device=device)
    #     return [z]
    # ############ 原始 forward 结束 ############

    ################################################滑窗推理######################################################
    def _forward_patch(self, batch, patch_w):
        """
        处理单个图像块的核心前向逻辑
        patch_w: 当前块的真实宽度 (用于确保展平和 SPConv 恢复时的尺寸正确)
        """
        device = batch['input'].device
        b, c, t, h, w = batch['input'].shape
        self._runtime_frame_divisor = max(float(b * t), 1.0)
        runtime_dict = self._new_runtime_dict() if self._runtime_enabled() else None
        need_stage1_profile = (not self.training) and ('bboxes' in batch)
        flops_dict, flops_handles = self._start_flops_capture(force=need_stage1_profile)

        try:
            ################################运行Net1##################################
            if runtime_dict is not None:
                time_start = self._runtime_start(device)
            net1_output = self.I2PNet(batch['input'])
            if runtime_dict is not None:
                runtime_dict['I2PNet'] += self._runtime_stop(time_start, device)
            ##########################################################################

            if runtime_dict is not None:
                time_start = self._runtime_start(device)
            if net1_output.shape[1] > 1:
                voxel_score_logits = net1_output.mean(dim=1, keepdim=True)
            else:
                voxel_score_logits = net1_output
            soft_mask = self.sigmoid(voxel_score_logits)
            if flops_dict is not None:
                flops_dict['ACS'] += self._estimate_acs_flops(soft_mask)

            binary_mask = self.get_mask_by_mean_std(
                soft_mask=soft_mask,
                var_coeff=self.thresh,
            )

            coords = torch.nonzero(binary_mask.squeeze(1)).contiguous()
            batch_idx = coords[:, 0]
            t_idx = coords[:, 1]
            h_idx = coords[:, 2]
            w_idx = coords[:, 3]
            flattened_indices = batch_idx * t * h * patch_w + t_idx * h * patch_w + h_idx * patch_w + w_idx

            batch_dict = {}
            voxel_feature_channels = net1_output.shape[1]
            voxel_features_flat = net1_output.permute(0, 2, 3, 4, 1).reshape(b * t * h * patch_w, voxel_feature_channels)
            batch_dict['voxel_features'] = voxel_features_flat[flattened_indices]
            batch_dict["voxel_coords"] = coords.to(device)
            batch_dict["batch_size"] = b
            # TAA heatmap vis switch: change True to False to disable.
            if True and not self.training and self._taa_heatmap_vis_has_budget(batch):
                batch_dict["capture_taa_heatmap"] = True
            if runtime_dict is not None:
                runtime_dict['ACS'] += self._runtime_stop(time_start, device)

            ################################运行Net2##################################
            if runtime_dict is not None:
                mfe_runtime_meter, mfe_runtime_handles = self._start_mfe_runtime_capture(device)
                time_start = self._runtime_start(device)
            else:
                mfe_runtime_meter, mfe_runtime_handles = None, None
            try:
                sp_backbone_out = self.sp_backbone(batch_dict)
                taa_skip_features = sp_backbone_out.get("taa_skip_features") if isinstance(sp_backbone_out, dict) else None
            finally:
                self._stop_runtime_capture(mfe_runtime_handles)
            if runtime_dict is not None:
                stage2_total_runtime = self._runtime_stop(time_start, device)
                returned_mfe_runtime = float(sp_backbone_out.get('mfe_runtime', 0.0)) if isinstance(sp_backbone_out, dict) else 0.0
                hooked_mfe_runtime = float(mfe_runtime_meter.get('MFE', 0.0)) if mfe_runtime_meter is not None else 0.0
                mfe_runtime = returned_mfe_runtime if returned_mfe_runtime > 0.0 else hooked_mfe_runtime
                runtime_dict['MFE'] += mfe_runtime
                runtime_dict['BackboneNoMFE'] += max(stage2_total_runtime - mfe_runtime, 0.0)
            ##########################################################################

            z = {}
            if runtime_dict is not None:
                time_start = self._runtime_start(device)
            input_sp_tensor = sp_backbone_out['encoded_spconv_tensor']
            for head in self.heads:
                out_h = getattr(self, head)(input_sp_tensor)

                if 'hm' in head:
                    if flops_dict is not None:
                        flops_dict['DetHead'] += self._estimate_sigmoid_flops(out_h.features)
                    out_h = replace_feature(out_h, self.sigmoid(out_h.features))
                    spatial_features = out_h.dense()
                    spatial_features = torch.clamp(spatial_features, min=1e-4, max=1 - 1e-4)
                    if flops_dict is not None:
                        flops_dict['DetHead'] += float(spatial_features.numel())
                else:
                    spatial_features = out_h.dense()

                z[head] = spatial_features[..., :patch_w]
            if runtime_dict is not None:
                runtime_dict['DetHead'] += self._runtime_stop(time_start, device)

            z['hm_large_heatmap'] = net1_output
            z['voxel_coords'] = batch_dict['voxel_coords']
            z['soft_mask'] = soft_mask
            z["binary_mask"] = binary_mask
            if taa_skip_features is not None:
                z["taa_skip_features"] = taa_skip_features

            return z, runtime_dict, flops_dict
        finally:
            self._stop_flops_capture(flops_handles)

    def forward(self, batch):
        device = batch['input'].device
        b, c, t, h, w = batch['input'].shape

        # 触发条件：非训练模式，且宽度足够大 (例如 1920)
        if not self.training and w >= 1920:
            w_half = w // 2

            # ========== 1. 处理左半部分 ==========
            batch_left = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_left['input'] = batch['input'][..., :w_half]

            with torch.no_grad():
                z_left, runtime_left, flops_left = self._forward_patch(batch_left, patch_w=w_half)

            del batch_left
            torch.cuda.empty_cache()

            # ========== 2. 处理右半部分 ==========
            batch_right = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_right['input'] = batch['input'][..., w_half:]

            with torch.no_grad():
                # 注意右半部分的宽度是 w - w_half (处理奇数宽度的严谨写法)
                z_right, runtime_right, flops_right = self._forward_patch(batch_right, patch_w=w - w_half)

            del batch_right
            torch.cuda.empty_cache()

            # ========== 3. 完美拼接输出 ==========
            z_merged = {}

            # (1) 拼接所有的密集特征图 (沿 Width 维度 dim=-1)
            for head in self.heads:
                z_merged[head] = torch.cat([z_left[head], z_right[head]], dim=-1)

            z_merged['hm_large_heatmap'] = torch.cat([z_left['hm_large_heatmap'], z_right['hm_large_heatmap']], dim=-1)
            z_merged['soft_mask']        = torch.cat([z_left['soft_mask'],        z_right['soft_mask']],        dim=-1)
            taa_skip_features = self._merge_taa_heatmap_features(
                z_left.pop("taa_skip_features", None),
                z_right.pop("taa_skip_features", None),
            )
            if taa_skip_features is not None:
                z_merged["taa_skip_features"] = taa_skip_features

            # (2) 拼接稀疏点云坐标 (核心操作：右半图坐标的 W 需要加上偏移量)
            coords_left  = z_left['voxel_coords']
            coords_right = z_right['voxel_coords']
            # coords 形状为 [N, 4]，对应 [Batch, T, H, W]，因此索引 3 就是宽度 W
            coords_right[:, 3] += w_half
            z_merged['voxel_coords'] = torch.cat([coords_left, coords_right], dim=0)

            # ====================== 计算采样率 ======================
            total_points   = b * t * h * w
            sampled_points = z_merged['voxel_coords'].shape[0]
            sampling_rate  = sampled_points / total_points if total_points > 0 else 0
            z_merged['sampling_rate'] = torch.tensor(sampling_rate, device=device)

            # ====================== 第一阶段检测质量指标 ======================
            # 沿 W 维拼接左右两半的 binary_mask，还原全图 [B, 1, T, H, W]
            binary_mask_full = torch.cat([z_left['binary_mask'], z_right['binary_mask']], dim=-1)
            runtime_merged = self._merge_runtime_dicts(runtime_left, runtime_right)
            flops_merged = self._merge_runtime_dicts(flops_left, flops_right)
            if 'bboxes' in batch:
                self._compute_and_print_stage1_metrics(
                    binary_mask=binary_mask_full,
                    bboxes=batch['bboxes'],
                    b=b, t=t, h=h, w=w,
                    device=device,
                    sampling_rate=sampling_rate,
                    flops_dict=flops_merged,
                )
            # ==============================================================

            self._update_runtime_stats(runtime_merged, flops_merged)

            if False and not self.training:
                self._save_net1_softmask_vis(z_merged["hm_large_heatmap"], batch)
            if True and not self.training:
                self._save_taa_heatmap_vis(z_merged.get("taa_skip_features"), batch, source_heatmap=z_merged.get("soft_mask"))
            z_merged.pop("taa_skip_features", None)

            return [z_merged]

        else:
            # ========== 正常模式 (训练时或小图) ==========
            z, runtime_dict, flops_dict = self._forward_patch(batch, patch_w=w)

            # ====================== 计算采样率 ======================
            total_points   = b * t * h * w
            sampled_points = z['voxel_coords'].shape[0]
            sampling_rate  = sampled_points / total_points if total_points > 0 else 0
            z['sampling_rate'] = torch.tensor(sampling_rate, device=device)

            # ====================== 第一阶段检测质量指标 ======================
            if not self.training and 'bboxes' in batch:
                self._compute_and_print_stage1_metrics(
                    binary_mask=z['binary_mask'],
                    bboxes=batch['bboxes'],
                    b=b, t=t, h=h, w=w,
                    device=device,
                    sampling_rate=sampling_rate,
                    flops_dict=flops_dict,
                )
            # ==============================================================

            self._update_runtime_stats(runtime_dict, flops_dict)

            if True and not self.training:
                self._save_net1_softmask_vis(z["hm_large_heatmap"], batch)
            if True and not self.training:
                self._save_taa_heatmap_vis(z.get("taa_skip_features"), batch, source_heatmap=z.get("soft_mask"))
            z.pop("taa_skip_features", None)

            return [z]
    ################################################滑窗推理######################################################

def I2PSOD_test(heads, image_size = [512,512], img_num = 20, layers=4, thresh=None,input_channels=1,feat_channels=[16,32,64],T_pooling=False,groups=2,downsample_mode='maxpool',net1name='UNet3D', opt=None):
    model =Img2PointsSmallObjectDetection(heads,  image_size = image_size, img_num = img_num, 
                                          layers=layers, thresh=thresh,
                                          feat_channels=feat_channels,T_pooling=T_pooling,
                                          groups=groups,downsample_mode=downsample_mode,net1name=net1name, opt=opt)
    return model



if __name__ == '__main__':
    import time
    import torch
    from lib.models.profile_utils import profile_model
    import sys
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    # 测试两种上采样方式（验证时间维度不池化）
    for upsample_mode in ["trilinear", ]: # "deconv"
        print(f"\n=== 测试上采样模式: {upsample_mode} ===")
        # 初始化模型
        model = I2PSOD_test(
            heads=heads, image_size = [512, 512], img_num = 10, layers=3.61, thresh=3,net1name='TOSConvNet',
            feat_channels=[16, 16],
        ).to(device).eval()
        print("\n=== 网络结构概要 ===")
        print(model)  # 打印完整模型结构
        # 测试输入：[B, C, D, H, W] = [1, 3, 5, 512, 512]（D=5为时间维度）
        test_input = torch.randn(1, 3, 10, 512, 512).to(device)  # 移除过时的Variable
        batch = {'input': test_input}
        # 前向传播 & 推理耗时
        start_time = time.time()
        # with torch.no_grad():
        #     output = model(batch)
        model.train() # 切换到训练模式以启用梯度计算检查显存占用
        output = model(batch)
        infer_time = time.time() - start_time

        # 基础信息打印
        print(f"输入尺寸: {test_input.shape}")
        # print(f"输出尺寸: {output[0]['hm'].shape}")
        print(f"推理耗时: {infer_time:.4f}s")
        # assert output.shape[2] == test_input.shape[2], "时间维度（D）尺寸被错误修改！"

        model.eval()
        flops, params, _ = profile_model(model, inputs=(batch,))
        print(f"Total Parameters: {params / 1e6:.8f} M")
        print(f"Total FLOPs (MACs): {flops / 1e9:.8f} G")
        # else:
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table

        # # 1. 计算 FLOPs (fvcore 会自动尝试追踪 F.conv3d 等函数)
        # # 这里的 batch 必须符合你 forward 的输入格式
        # flops = FlopCountAnalysis(model, batch) 
        # print(f"Total FLOPs: {flops.total() / 1e9:.4f} G")

        # # 2. 打印精美的参数量表格 (按层分类)
        # print(parameter_count_table(model))

        # # 如果想看哪些层被识别了，哪些没识别：
        # print(flops.by_module())
        
