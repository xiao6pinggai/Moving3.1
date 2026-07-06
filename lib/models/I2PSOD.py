import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import os, sys

# 向上查找项目根目录并加入 sys.path（支持 autodl/本地 Windows 双环境）
_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_cur, 'path_setup.py')):
    _cur = os.path.dirname(_cur)
if _cur not in sys.path:
    sys.path.insert(0, _cur)

from lib.utils1.enlarge_mask import dilate_mask_fast
from lib.utils1.bbox2binarymask import bboxes_to_binary_mask

# 验证是否生效
# print("修正后的sys.path:", sys.path[:3])
from lib.models.spconv_unet import UNetV2, UNetV2_3, UNetV2_2, UNetV2_3_32, UNetV2_3_T_nodown, UNetV2_3_T_nodown_maxpool, UNetV2_3_T_nodown_v2, UNetV2_3_T_nodown_v3
from lib.models.spconv_utils import replace_feature, spconv

from lib.models.noramlconv_unet3d2_1 import UNet2DWithNormalConv2D, UNet3DWithNormalConv3D, LightWeightedConv3D, EncoderOnlyConv3DProposalNet, TOSConvNet, TPConvNet, TZSConvNet, DynamicTOSConvNet



from lib.utils1.show_one_img import show_one_img
import torch

class Img2PointsSmallObjectDetection(nn.Module):
    def __init__(self, heads, image_size = [512,512], img_num = 20, layers = 3, thresh=None, input_channels=1, 
                 feat_channels=[16,32,64,128], T_pooling=False,groups=1,downsample_mode='stride',
                 net1name='UNet3DWithNormalConv3D', opt=None):
        super().__init__()
        self.print = 0
        # points generate net
        self.net1name=net1name
        temporal_mode = getattr(opt, 'use_tzsconv', 'tzsconv')
        
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
        self.net1_feature_channels = feat_channels[0]
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

    def forward(self, batch):
        device = batch['input'].device
        b, c, t, h, w = batch['input'].shape
        
        net1_output = self.I2PNet(batch['input'])  # B 3 T H W --> B C T H W
        # 注意：不使用 EncoderOnlyConv3DProposalNet，其 forward 返回 dict 而非 tensor

        if net1_output.shape[1] > 1:
            voxel_score_logits = net1_output.mean(dim=1, keepdim=True)
        else:
            voxel_score_logits = net1_output

        soft_mask = self.sigmoid(voxel_score_logits)  # B 1 T H W

        # 核心：基于均值+方差卡阈值
        binary_mask = self.get_mask_by_mean_std( # b 1 T h w -->0/1.0
            soft_mask=soft_mask,
            var_coeff=self.thresh,    # 可根据数据集调整（0.3宽松，1.0严格）
        ) 
        
        ################形态学扩张####################
        # binary_mask = dilate_mask_fast(binary_mask)
        #############################################
        #############输入改为bbox2mask################
        # binary_mask = bboxes_to_binary_mask(batch['bboxes'], h, w)
        # voxel_features = binary_mask
        # voxel_features_ori = binary_mask
        #############################################
        # # 可选：打印日志，监控有效点数分布（调试用）
        # valid_pts = torch.sum(binary_mask.view(b, -1), dim=-1)
        # print(f"每帧有效点数：{valid_pts.cpu().numpy()}")
        
        coords = torch.nonzero(binary_mask.squeeze(1)).contiguous()
        # ====================== 新增：计算采样率 ======================
        total_points = b * t * h * w  # 理论上的所有点数
        sampled_points = coords.shape[0] # 经过阈值筛选后的有效点数
        sampling_rate = sampled_points / total_points if total_points > 0 else 0
        # del binary_mask, soft_mask

        ####反向传播报错，特征第一维度和坐标第一维度不匹配
        # 将 net1_features 展平为 [B*T*H*W, 1]，然后通过坐标筛选有效特征
        # coords 的格式为 [batch_idx, t, h, w]，需转换为展平后的索引
        batch_idx = coords[:, 0]  # 批次索引
        t_idx = coords[:, 1]      # 时间维度索引
        h_idx = coords[:, 2]      # 高度索引
        w_idx = coords[:, 3]      # 宽度索引
        # 计算展平后的索引（对应 B*T*H*W 中的位置）
        flattened_indices = batch_idx * t * h * w + t_idx * h * w + h_idx * w + w_idx
        # 筛选有效特征（仅保留 coords 对应的特征）
        # voxel_features = voxel_features.reshape(b*t*h*w, 1)[flattened_indices]
        ########################################################

        batch_dict = {}
        # lhg 20251216新增输入包括原图取点云和net1特征取点云拼接
        # ====================== 新增：拼接voxel_features和原始输入 ======================
        # 通道维度（dim=1）拼接，得到 B 4 T H W 的特征

        # voxel_features = torch.cat([batch['input'], voxel_features], dim=1)  # 核心修改1

        # ====================== 修改：索引拼接后的4通道特征 ======================
        # 展平为 [B*T*H*W, 4]，再通过flattened_indices索引，得到 [N,4] 的点云特征（N是有效点数）
        voxel_feature_channels = net1_output.shape[1]
        voxel_features_flat = net1_output.permute(0, 2, 3, 4, 1).reshape(b * t * h * w, voxel_feature_channels)
        batch_dict['voxel_features'] = voxel_features_flat[flattened_indices]  # 核心修改2
        batch_dict['voxel_coords'] = coords.to(device)
        batch_dict['batch_size'] = b
        if coords.shape[0] == 0:
            print("Warning: No points generated from I2PNet!")
        if batch_dict['voxel_features'].shape[0] == 0:
            print("Warning: No voxel features selected for SPConvNet!")
        '''# -------------------------- 关键：记录net2运行前的显存基线 --------------------------
        # 1. 清空缓存（避免临时碎片干扰）
        torch.cuda.empty_cache()
        # 2. 记录net2运行前的当前已分配显存（基线）
        net2_before_allocated = torch.cuda.memory_allocated(device)
        # 3. 重置峰值显存统计（只统计net2运行阶段的峰值）
        torch.cuda.reset_peak_memory_stats(device)
        print(f"net2将要运行：\n\t",torch.cuda.memory_summary(device, abbreviated=True))  # abbreviated=True 简化输出'''
        ################################运行Net2##################################
        sp_backbone_out = self.sp_backbone(batch_dict)
        ##########################################################################
        z = {}
        for head in self.heads:
            input_sp_tensor = sp_backbone_out['encoded_spconv_tensor']
            # print("input_sp_tensor shape:", input_sp_tensor.spatial_shape)
            # print("input_sp_tensor indices max:", input_sp_tensor.indices.max())
            # print("input_sp_tensor indices min:", input_sp_tensor.indices.min())
            out_h = self.__getattr__(head)(input_sp_tensor)

            if 'hm' in head:
                out_h = replace_feature(out_h, self.sigmoid(out_h.features))
                spatial_features = out_h.dense()
                spatial_features = torch.clamp(spatial_features, min=1e-4, max=1 - 1e-4)
            else:
                spatial_features = out_h.dense()
            z[head] = spatial_features
        z['hm_large_heatmap'] = net1_output
        z['voxel_coords'] = batch_dict['voxel_coords']
        z['soft_mask'] = soft_mask # b 1 t h w
        # z['lasso'] = torch.sum(mask_all, dim=[-1,-2]) / (h * w)
        '''# -------------------------- 计算net2新增的显存 --------------------------
        # 1. net2运行后的当前已分配显存
        net2_after_allocated = torch.cuda.memory_allocated(device)
        # 2. net2运行过程中的峰值显存（仅net2阶段）
        net2_peak_allocated = torch.cuda.max_memory_allocated(device)
        # 3. 计算新增显存（当前增量/峰值增量）
        net2_add_current = (net2_after_allocated - net2_before_allocated) / 1024**2  # 转MB
        net2_add_peak = (net2_peak_allocated - net2_before_allocated) / 1024**2      # 转MB

        # 打印net2专属的新增显存
        print(f"\n=== net2 新增显存统计 ===")
        print(f"net2运行前基线显存：{net2_before_allocated/1024**2:.2f} MB")
        print(f"net2当前新增显存：{net2_add_current:.2f} MB")
        print(f"net2峰值新增显存：{net2_add_peak:.2f} MB")
        print(f"net2运行完毕\n\t",torch.cuda.memory_summary(device, abbreviated=True))  # abbreviated=True 简化输出'''
        # ====================== 新增：输出采样率 ======================
        # 转成 Tensor 方便在外部与其他特征统一处理或收集
        z['sampling_rate'] = torch.tensor(sampling_rate, device=device) 
        # ==============================================================
        return [z]


def I2PSOD(heads, image_size = [512,512], img_num = 20, layers=4, thresh=None,input_channels=1,feat_channels=[16,32,64],T_pooling=False,groups=2,downsample_mode='maxpool',net1name='UNet3D', opt=None):
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
        model = I2PSOD(
            heads=heads, image_size = [512, 512], img_num = 10, layers=3.61, thresh=3,net1name='UNet3DWithNormalConv3D',
            feat_channels=[8,16,32,64],
        ).to(device).eval()
        # print("\n=== 网络结构概要 ===")
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
        
