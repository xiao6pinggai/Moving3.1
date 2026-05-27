# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from lib.utils1.utils import _transpose_and_gather_feat
import torch.nn.functional as F


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _standard_focal_loss(pred, gt, alpha=0.25, gamma=2.0):
    ''' 标准Focal Loss（来自《Focal Loss for Dense Object Detection》）
        输入输出格式与原CornerNet修改版完全一致
      Arguments:
        pred (batch x c x h x w): 预测概率图（需保证0≤pred≤1）
        gt (batch x c x h x w): 标签图（标准FL默认是0/1二值标签，兼容原代码0~1热图）
        alpha (float): 正负样本平衡因子，默认0.25（正样本权重更高）
        gamma (float): 聚焦参数，默认2.0（调节难分样本权重）
    '''
    # 数值稳定性：避免log(0)或log(1)导致的无穷大/NaN
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1.0 - eps)

    # 区分正负样本（兼容原代码gt的0~1范围，正样本=1，负样本=≠1）
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()  # 与原代码负样本定义完全对齐

    # 核心：计算p_t（正样本p_t=pred，负样本p_t=1-pred）
    p_t = pos_inds * pred + neg_inds * (1 - pred) ### 这里仅针对gt==1的像素点使用这个正样本损失
    # 计算alpha_t（正样本=alpha，负样本=1-alpha，平衡正负样本数量）
    alpha_t = pos_inds * alpha + neg_inds * (1 - alpha) ###这里生成了一个和pred同样shape的tensor，正样本位置是alpha，负样本位置是1-alpha
    # 聚焦权重：(1-p_t)^gamma，难分样本（p_t≈0.5）权重更高
    focal_weight = torch.pow(1 - p_t, gamma)

    # 标准Focal Loss公式：-α_t * (1-p_t)^γ * log(p_t)
    loss = -alpha_t * focal_weight * torch.log(p_t)

    # 与原代码完全一致的归一化逻辑
    num_pos = pos_inds.float().sum()
    loss_sum = loss.sum()

    if num_pos == 0:
        # 无正样本时仅计算负样本损失（保持原代码逻辑）
        loss = loss_sum
    else:
        # 有正样本时按正样本数归一化
        loss = loss_sum / num_pos

    return loss




def _hard_mining_adaptive_loss(pred, gt, bbox_mask, alpha=1.0, tk=3.0):
    '''
    Arguments:
        pred (B, C, H, W): 预测热图，经过 sigmoid
        gt (B, C, H, W): 真值热图 (椭圆高斯, 中心为1, 边缘为0-1之间)
        bbox_mask: 这里其实主要用到 gt 即可，bbox_mask 可用于辅助定义纯背景
        alpha (float): 额外惩罚项的权重系数 (默认为1.0，即 hard example 权重变2倍)
        tk (float): 自适应阈值系数 (mean + tk * std)
    '''
    
    # ==================================================================
    # Part 1: 标准 CenterNet Loss (Base Loss) - 检查确认无误
    # ==================================================================
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    # 负样本权重：(1-Y)^4，用于减少高斯斑附近的惩罚，保护高斯形状
    neg_weights = torch.pow(1 - gt, 4)

    # 基础正样本损失: -(1-p)^2 * log(p)
    pos_loss_term = torch.log(pred + 1e-4) * torch.pow(1 - pred, 2) * pos_inds
    
    # 基础负样本损失: -(1-y)^4 * p^2 * log(1-p)
    neg_loss_term = torch.log(1 - pred + 1e-4) * torch.pow(pred, 2) * neg_weights * neg_inds

    # 计算 Base Sum
    # 注意：这里先不除以 num_pos，方便后面统一归一化
    base_pos_loss_sum = -pos_loss_term.sum()
    base_neg_loss_sum = -neg_loss_term.sum()
    
    # ==================================================================
    # Part 2: 动态阈值计算
    # ==================================================================
    with torch.no_grad():
        # 计算单图统计量 (B, C, 1, 1)
        instance_mean = torch.mean(pred.detach(), dim=[-2, -1], keepdim=True)
        instance_std  = torch.std(pred.detach(), dim=[-2, -1], keepdim=True)
        
        # 动态阈值 T
        adaptive_thresh = instance_mean + tk * instance_std

    # ==================================================================
    # Part 3: Recall Loss (针对漏检的正样本)
    # ==================================================================
    # 定义：是中心点(gt==1) 且 预测值 <= 阈值
    # 这种点属于"没学好"的正样本
    hard_pos_mask = pos_inds * (pred <= adaptive_thresh).float() # B1HW
    
    # 计算额外的 Recall Loss
    # 【关键修改】：直接复用 pos_loss_term！
    # 这意味着我们用完全相同的公式再罚一遍，保证梯度方向一致，只是加倍惩罚
    extra_recall_loss = -pos_loss_term * hard_pos_mask
    extra_recall_loss_sum = extra_recall_loss.sum()

    # ==================================================================
    # Part 4: Precision Loss (针对虚警的负样本)
    # ==================================================================
    # 定义：是背景(gt<1) 且 预测值 > 阈值
    # 这种点属于"太嚣张"的负样本
    # 注意：这里依然乘了 neg_inds，所以不会把中心点算进去
    hard_neg_mask = neg_inds * (pred > adaptive_thresh).float()
    
    # 计算额外的 Precision Loss
    # 【关键修改】：直接复用 neg_loss_term！
    # 这一点非常重要：neg_loss_term 包含了 neg_weights (1-gt)^4
    # 这意味着如果这个"虚警"是在高斯斑边缘 (gt=0.8)，惩罚会自动很小，不会破坏高斯形状
    # 如果是在纯背景 (gt=0)，惩罚会很大
    extra_prec_loss = -neg_loss_term * hard_neg_mask
    extra_prec_loss_sum = extra_prec_loss.sum()

    # ==================================================================
    # Part 5: 总损失与归一化
    # ==================================================================
    # 归一化因子：通常只除以正样本数量 N
    num_pos = pos_inds.float().sum()
    
    if num_pos == 0:
        # 只有负样本损失
        loss = base_neg_loss_sum + alpha * extra_prec_loss_sum
    else:
        # 总损失 = (基础正 + 基础负 + alpha*额外正 + alpha*额外负) / N
        # 当 alpha=1.0 时，对于困难样本，相当于权重变成了 2.0
        total_sum = (base_pos_loss_sum + base_neg_loss_sum) + \
                    alpha * (extra_recall_loss_sum + extra_prec_loss_sum)
        loss = total_sum / num_pos
        
    return loss


def _relaxed_soft_focal_loss(pred, gt, alpha=0.25, gamma=2.0):
    ''' 
    宽容版 Soft Focal Loss (适配平顶高斯 + 2x2 容差)
    
    Arguments:
        pred (batch x c x h x w): 预测概率图
        gt (batch x c x h x w): 平顶高斯标签图 (0 <= gt <= 1)
        alpha (float): 正负样本平衡因子
        gamma (float): 聚焦参数
    '''
    # 1. 数值稳定性
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1.0 - eps)

    # 2. 生成“宽容”的预测图 (Tolerance Mechanism)
    # 使用 2x2 MaxPooling 提取局部最大值
    # padding=1 + 切片操作是为了保证输出尺寸与输入 (h, w) 严格一致
    # 逻辑：只要 2x2 邻域内有一个点预测准了，pred_relaxed 就会高
    pred_relaxed = F.max_pool2d(pred, kernel_size=2, stride=1, padding=1)[:, :, :-1, :-1]
    pred_relaxed = torch.clamp(pred_relaxed, eps, 1.0 - eps)

    # 3. 定义正负样本区域 (基于 Soft Label)
    # gt > 0 的地方都算有目标信号 (正样本区域)
    # gt == 0 的地方是纯背景 (负样本区域)
    pos_mask = gt.gt(0).float() 
    neg_mask = gt.eq(0).float()

    # ---------------------------------------------------------------------
    # 4. 正样本损失 (Recall 优化) - 使用 "宽容预测" pred_relaxed
    # ---------------------------------------------------------------------
    # 权重设计 (QFL风格): |gt - pred_relaxed|^gamma
    # 如果 gt=0.9, pred_relaxed=0.9 (即使原位置是0.1，只要旁边有0.9)，权重 -> 0，Loss -> 0
    pos_weight = torch.abs(gt - pred_relaxed).pow(gamma)
    
    # Loss公式: - alpha * gt * weight * log(pred_relaxed)
    # 这里的 gt 作为系数，意味着平顶高斯中心(1.0)比边缘(0.5)更重要
    pos_loss = -alpha * gt * pos_weight * torch.log(pred_relaxed)
    
    # 只保留 gt > 0 区域的损失
    pos_loss = pos_loss * pos_mask

    # ---------------------------------------------------------------------
    # 5. 负样本损失 (Precision 优化) - 使用 "严格预测" pred
    # ---------------------------------------------------------------------
    # 背景必须严格抑制，不能宽容，所以这里用原始 pred
    neg_weight = torch.pow(pred, gamma)
    
    # Loss公式: - (1-alpha) * weight * log(1-pred)
    neg_loss = -(1 - alpha) * neg_weight * torch.log(1 - pred)
    
    # 只保留 gt == 0 区域的损失
    neg_loss = neg_loss * neg_mask

    # 6. 归一化 (使用 gt 的总强度，而不是像素个数，以适应软标签)
    # 对应数学公式中的 N_pos
    num_pos = gt.sum() 
    
    loss = pos_loss.sum() + neg_loss.sum()

    if num_pos == 0:
        return loss
    else:
        return loss / num_pos
def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)
    
class FocalLoss_for_points(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss_for_points, self).__init__()
        self.standard_focal_loss = _standard_focal_loss

    def forward(self, out, target):
        return self.standard_focal_loss(out, target)
class RecallHmLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(RecallHmLoss, self).__init__()
        self.instance_adaptive_recall_loss = _hard_mining_adaptive_loss

    def forward(self, out, target, bbox_mask):
        return self.instance_adaptive_recall_loss(out, target, bbox_mask)
    
class TolerantSoftFocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(TolerantSoftFocalLoss, self).__init__()
        self.relaxed_soft_focal_loss = _relaxed_soft_focal_loss

    def forward(self, out, target):
        return self.relaxed_soft_focal_loss(out, target)
    
class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
