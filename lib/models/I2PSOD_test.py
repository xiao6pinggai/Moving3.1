import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import os, sys
import time
import atexit
import math

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
from lib.models.noramlconv_unet3d2_1 import UNet2DWithNormalConv2D, UNet3DWithNormalConv3D, LightWeightedConv3D, EncoderOnlyConv3DProposalNet, TOSConvNet, TZSConvNet, DynamicTOSConvNet
from lib.models.profile_utils import count_parameters, estimate_module_flops, estimate_sigmoid_flops, extract_feature_tensor
from lib.utils1.show_one_img import show_one_img
import torch

class Img2PointsSmallObjectDetection(nn.Module):
    def __init__(self, heads, image_size = [512,512], img_num = 20, layers = 3, thresh=None, input_channels=1,
                 feat_channels=[16,32,64,128], T_pooling=False,groups=1,downsample_mode='stride',
                 net1name='UNet3DWithNormalConv3D', opt=None):
        super().__init__()
        self.print = 0
        self.use_runtime = getattr(opt, 'use_runtime', False)
        self.runtime_component_names = (
            'I2PNet',
            'ACS',
            'BackboneNoMFE',
            'MFE',
            'DetHead',
        )
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
        # ==================================================
        # points generate net
        self.net1name=net1name
        temporal_mode = getattr(opt, 'use_tzsconv', 'tzsconv')
        self.net1_feature_channels = feat_channels[0]
        if net1name=='UNet3DWithNormalConv3D':
            self.I2PNet = UNet3DWithNormalConv3D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode, use_final_conv=True,TConvOnly=False)
        elif net1name=='UNet2DWithNormalConv2D':
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

            self.tau = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.tau.data.fill_(1)
            self.conv_std = nn.Sequential(
                nn.AdaptiveAvgPool2d([1, 1]),
                nn.Conv2d(img_num, img_num, 1),
                nn.ReLU(inplace=True)
            )

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
    def _compute_and_print_stage1_metrics(self, binary_mask, bboxes, b, t, h, w, device):
        """
        计算并实时打印三项指标，同时累计数据集总平均。

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

        # ---- 数据集累计均值 ----
        ds_recall = self._stat_recall_sum / self._stat_recall_cnt if self._stat_recall_cnt > 0 else 0.0
        ds_hit    = self._stat_hit_num    / self._stat_hit_den    if self._stat_hit_den    > 0 else 0.0
        ds_fa     = self._stat_fa_sum     / self._stat_fa_cnt     if self._stat_fa_cnt     > 0 else 0.0

        # ---- 打印 ----
        print(
            f"[Stage1 Metrics] Batch {self._stat_batch_idx:04d} | "
            f"当前: 命中率={cur_hit*100:.2f}%  命中覆盖率={cur_recall*100:.2f}%  虚警率={cur_fa*100:.4f}% | "
            f"数据集累计: 命中率={ds_hit*100:.2f}%  命中覆盖率={ds_recall*100:.2f}%  虚警率={ds_fa*100:.4f}%"
        )

    def print_dataset_stage1_summary(self):
        """推理结束后主动调用，打印整个数据集的汇总指标。"""
        ds_recall = self._stat_recall_sum / self._stat_recall_cnt if self._stat_recall_cnt > 0 else 0.0
        ds_hit    = self._stat_hit_num    / self._stat_hit_den    if self._stat_hit_den    > 0 else 0.0
        ds_fa     = self._stat_fa_sum     / self._stat_fa_cnt     if self._stat_fa_cnt     > 0 else 0.0
        print("=" * 80)
        print(f"[Stage1 Dataset Summary]  共处理 {self._stat_batch_idx} 个 batch，"
              f"有效框 {self._stat_hit_den} 个，帧 {self._stat_fa_cnt} 帧")
        print(f"  第一阶段命中率（hit_rate）        : {ds_hit*100:.4f}%")
        print(f"  第一阶段命中覆盖率（coverage_rate）: {ds_recall*100:.4f}%")
        print(f"  点级虚警率（false_alarm）         : {ds_fa*100:.6f}%")
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

    def get_component_stats(self, per_frame_divisor=1):
        runtime_stats = self.get_runtime_stats(per_frame_divisor=per_frame_divisor)
        flops_stats = self.get_flops_stats(per_frame_divisor=per_frame_divisor)
        param_stats = self.get_component_param_stats()
        stats = {}
        for name in self.runtime_component_names:
            stats[name] = {
                'runtime': float(runtime_stats.get(name, 0.0)),
                'flops': float(flops_stats.get(name, 0.0)),
                'params': int(param_stats.get(name, 0)),
            }
        stats['Total'] = {
            'runtime': sum(stats[name]['runtime'] for name in self.runtime_component_names),
            'flops': sum(stats[name]['flops'] for name in self.runtime_component_names),
            'params': self.get_total_param_count(),
        }
        return stats

    def print_component_stats_summary(self, per_frame_divisor=None, mark_printed=True):
        if self._runtime_count == 0:
            return
        if per_frame_divisor is None:
            per_frame_divisor = self._runtime_frame_divisor
        stats = self.get_component_stats(per_frame_divisor=per_frame_divisor)
        print('=' * 80)
        print('[Component Profiling Summary]')
        for component_name in self.runtime_component_names:
            item = stats[component_name]
            print(
                f"  {component_name}: runtime={item['runtime']:.6f}s  "
                f"FLOPs={item['flops']:.6f}  Params={item['params']}"
            )
        total = stats['Total']
        print(
            f"  Total: runtime={total['runtime']:.6f}s  "
            f"FLOPs={total['flops']:.6f}  Params={total['params']}"
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

    def _flops_enabled(self):
        return self.use_runtime and (not self.training)

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

    def _start_flops_capture(self):
        if not self._flops_enabled():
            return None, None
        flops_dict = self._new_runtime_dict()
        handles = []

        def _make_hook(component_name):
            def _hook(module, inputs, output):
                flops_dict[component_name] += self._estimate_module_flops(module, inputs, output)
            return _hook

        for module_name, module in self.named_modules():
            if module_name == '':
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
        self._runtime_frame_divisor = max(float(t), 1.0)
        runtime_dict = self._new_runtime_dict() if self._runtime_enabled() else None
        flops_dict, flops_handles = self._start_flops_capture()

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
            batch_dict['voxel_coords'] = coords.to(device)
            batch_dict['batch_size'] = b
            if runtime_dict is not None:
                runtime_dict['ACS'] += self._runtime_stop(time_start, device)

            ################################运行Net2##################################
            if runtime_dict is not None:
                time_start = self._runtime_start(device)
            sp_backbone_out = self.sp_backbone(batch_dict)
            if runtime_dict is not None:
                stage2_total_runtime = self._runtime_stop(time_start, device)
                mfe_runtime = float(sp_backbone_out.get('mfe_runtime', 0.0)) if isinstance(sp_backbone_out, dict) else 0.0
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
            z['binary_mask'] = binary_mask

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
            if 'bboxes' in batch:
                self._compute_and_print_stage1_metrics(
                    binary_mask=binary_mask_full,
                    bboxes=batch['bboxes'],
                    b=b, t=t, h=h, w=w,
                    device=device,
                )
            # ==============================================================

            runtime_merged = self._merge_runtime_dicts(runtime_left, runtime_right)
            flops_merged = self._merge_runtime_dicts(flops_left, flops_right)
            self._update_runtime_stats(runtime_merged, flops_merged)

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
                )
            # ==============================================================

            self._update_runtime_stats(runtime_dict, flops_dict)

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
        
