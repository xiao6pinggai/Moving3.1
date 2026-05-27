import torch
import torch.nn as nn
from functools import partial
import numpy as np
import os, sys








ROOT_DIR = "/root/autodl-tmp/Moving3.1"
# 确保根目录在sys.path首位（覆盖默认的子目录）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lib.models.noramlconv_unet3d13 import UNet3DWithTSC3D
from lib.models.noramlconv_unet3d14 import UNet3DWithTSC3D_CUDA

from lib.models.noramlconv_unet3d10 import UNet3DCatATDC
from lib.models.noramlconv_unet3d9 import UNet3DAddATDC
from lib.utils1.enlarge_mask import dilate_mask_fast
from lib.utils1.bbox2binarymask import bboxes_to_binary_mask

# 验证是否生效
# print("修正后的sys.path:", sys.path[:3])
from lib.models.spconv_unet import UNetV2, UNetV2_3, UNetV2_2, UNetV2_3_32, UNetV2_3_T_nodown, UNetV2_3_T_nodown_maxpool
from lib.models.spconv_utils import replace_feature, spconv
from lib.models.noramlconv_unet3d2 import UNet3D, UNet3DATDC
from lib.models.noramlconv_unet3d0_1 import UNet3DGroupDilation4Branch
from lib.models.noramlconv_unet3d3 import UNet3DATDCDilation
from lib.models.noramlconv_unet3d4 import UNet3DZSTA
from lib.models.noramlconv_unet3d5 import UNet3DTAM
from lib.models.noramlconv_unet3d6 import UNet3DGroupATDCDilation
from lib.models.noramlconv_unet3d8 import UNet3DGroupATDCDilation4Branch
from lib.models.noramlconv_unet3d9 import UNet3DAddATDC
from lib.models.noramlconv_unet3d11 import UNet3DwithZZLB
from lib.models.noramlconv_unet3d12 import UNet3DWithATDCDilation2
from lib.models.noramlconv_unet3d13 import UNet3DWithTSC3D
from lib.models.noramlconv_unet3d2_1 import UNet3DWithNormalConv3D
from lib.models.noramlconv_unet3d_3branch import UNet3DWithNormalConv3D3Branch
from lib.models.noramlconv_unet3d_3branchDilation import UNet3DWithGroupedMultiBranch
from lib.models.noramlconv_unet3dcsam import UNet3DWithNormalConv3DCSAM
from lib.models.noramlconv_unet3patdc import UNet3DWithNormalConv3DPATDC

from lib.models.noramlconv_unet3patdc_split import UNet3DWithNormalConv3DPATDCSplit
from lib.utils1.show_one_img import show_one_img
import torch

class Img2PointsSmallObjectDetection(nn.Module):
    def __init__(self, heads, image_size = [512,512], img_num = 20, layers = 3, thresh=None, input_channels=1, 
                 feat_channels=[16,32,64,128], T_pooling=False,groups=1,downsample_mode='stride',
                 net1name='UNet3DWithNormalConv3D'):
        super().__init__()
        self.print = 0
        # points generate net
        self.net1name=net1name
        if net1name=='UNet3DATDC':
                self.I2PNet = UNet3DATDC(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DATDCDilation':
                self.I2PNet = UNet3DATDCDilation(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3D': 
            self.I2PNet = UNet3D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name == 'UNet3DZSTA':
            self.I2PNet = UNet3DZSTA(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name == 'UNet3DTAM':
            self.I2PNet = UNet3DTAM(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name == 'UNet3DGroupATDCDilation':
            self.I2PNet = UNet3DGroupATDCDilation(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name == 'UNet3DGroupATDCDilation4Branch':
            self.I2PNet = UNet3DGroupATDCDilation4Branch(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name == 'UNet3DGroupDilation4Branch': # 验证时域金字塔的作用
            self.I2PNet = UNet3DGroupDilation4Branch(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DAddATDC': 
            self.I2PNet = UNet3DAddATDC(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DCatATDC': 
            self.I2PNet = UNet3DCatATDC(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)  
        elif net1name=='UNet3DwithZZLB':
            self.I2PNet = UNet3DwithZZLB(num_channels=4, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DWithATDCDilation2':
            self.I2PNet = UNet3DWithATDCDilation2(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DWithNormalConv3D':
            self.I2PNet = UNet3DWithNormalConv3D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DWithTSC3D':
            self.I2PNet = UNet3DWithTSC3D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DWithTSC3D_CUDA':
            self.I2PNet = UNet3DWithTSC3D_CUDA(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DWithNormalConv3D3Branch':
            self.I2PNet = UNet3DWithNormalConv3D3Branch(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DWithGroupedMultiBranch':
            self.I2PNet = UNet3DWithGroupedMultiBranch(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,downsample_mode=downsample_mode)
        elif net1name == 'UNet3DWithNormalConv3DCSAM':
            self.I2PNet = UNet3DWithNormalConv3DCSAM(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,downsample_mode=downsample_mode)
        elif net1name=='UNet3DWithNormalConv3DPATDC':
            self.I2PNet = UNet3DWithNormalConv3DPATDC(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        elif net1name=='UNet3DWithNormalConv3DPATDCSplit':
            self.I2PNet = UNet3DWithNormalConv3DPATDCSplit(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None, 
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode)
        else:
            print('net1name 错误！')
        self.sigmoid = nn.Sigmoid()
        # 稀疏
        self.thresh=thresh
        head_conv=128
        grid_size = np.array([image_size[1], image_size[0], img_num - 1])
        self.points_all = img_num*image_size[0]*image_size[1]
        if  layers==4:
            self.sp_backbone = UNetV2(input_channels, grid_size)
        elif layers==3:
            self.sp_backbone = UNetV2_3(input_channels, grid_size)
        elif layers == 2:
            self.sp_backbone = UNetV2_2(input_channels, grid_size)
        elif layers == 3.5:
            self.sp_backbone = UNetV2_3_32(input_channels, grid_size)
        elif layers == 3.6:
            self.sp_backbone = UNetV2_3_T_nodown(input_channels, grid_size)
        elif layers == 3.7:
            self.sp_backbone = UNetV2_3_T_nodown_maxpool(input_channels, grid_size)
        else:
            raise Exception('Not a valid mode!!!!!')
        head_input_channel = self.sp_backbone.num_point_features
        ###get head conv
        # self.heads = heads
        # for head in self.heads:
        #     classes = self.heads[head]
        #     name_1 = 'subm1'+head
        #     name_2 = 'subm2'+head
        #     if head_conv > 0:
        #         if 'hm' in head:
        #             fc = spconv.SparseSequential(
        #                     spconv.SubMConv3d(head_input_channel, head_conv, 3, padding=1, bias=False, indice_key=name_1),
        #                 nn.ReLU(),
        #                 spconv.SubMConv3d(head_conv, classes, 3, padding=1, bias=True, indice_key=name_2),
        #                 )
        #         else:
        #             fc = spconv.SparseSequential(
        #                 spconv.SubMConv3d(head_input_channel, head_conv, 3, padding=1, bias=False, indice_key=name_1),
        #                 nn.ReLU(),
        #                 spconv.SubMConv3d(head_conv, classes, 3, padding=1, bias=False, indice_key=name_2),
        #                 )
        #     else:
        #         fc = spconv.SubMConv3d(head_input_channel, classes, 3, padding=1, bias=True, indice_key=name_1)
        #     ###
        #     if 'hm' in head:
        #         fc[-1].bias.data.fill_(-2.19)
        import torch.nn as nn

        # 使用 ModuleDict 管理头部
        self.heads_dict = nn.ModuleDict()

        for head in self.heads:
            classes = self.heads[head]
            
            # 构建 3D 密集卷积头
            if head_conv > 0:
                if 'hm' in head:
                    fc = nn.Sequential(
                        # 第一层：3D 卷积
                        nn.Conv3d(head_input_channel, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.BatchNorm3d(head_conv), # 可选：密集卷积通常配合 BN 使用
                        nn.ReLU(),
                        # 第二层：输出层 (保持 3D)
                        nn.Conv3d(head_conv, classes, kernel_size=3, padding=1, bias=True)
                    )
                else:
                    fc = nn.Sequential(
                        nn.Conv3d(head_input_channel, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.BatchNorm3d(head_conv),
                        nn.ReLU(),
                        nn.Conv3d(head_conv, classes, kernel_size=3, padding=1, bias=True)
                    )
            else:
                # 无中间层，直接映射
                fc = nn.Conv3d(head_input_channel, classes, kernel_size=3, padding=1, bias=True)

            # 针对 Heatmap 的偏置初始化
            if 'hm' in head:
                if isinstance(fc, nn.Sequential):
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fc.bias.data.fill_(-2.19)
                    
            self.heads_dict[head] = fc
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
    def get_mask_by_mean_std(self, soft_mask, var_coeff=3, min_thresh=0.01):
        """
        基于soft_mask的空间维度(HW)均值+方差计算动态阈值
        保证每一帧(T)的有效点数≥50（不足则取该帧的Top50）
        """
        var_coeff = var_coeff if var_coeff is not None else 3
        B, C, T, H, W = soft_mask.shape
        assert C == 1, "soft_mask通道数必须为1"
        device = soft_mask.device
        
        # =========================================================================
        # Part 1: 计算均值方差 (完全对齐 Reference Snippet)
        # -------------------------------------------------------------------------
        # 输入: [B, 1, T, H, W]
        # dim=[-2, -1]: 对 H, W 求统计量 -> 结果 [B, 1, T]
        # unsqueeze: 恢复为 [B, 1, T, 1, 1] 以便广播
        # =========================================================================
        
        # : Show B,C,T,H,W collapsing H and W to 1,1
        mask_mean = torch.mean(soft_mask, dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1)
        mask_std = torch.std(soft_mask, dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1)
        
        # 2. 计算动态阈值
        # 利用广播机制: [B,1,T,1,1] 作用于 [B,1,T,H,W]
        dynamic_thresh = mask_mean + var_coeff * mask_std
        dynamic_thresh = torch.clamp(dynamic_thresh, min=min_thresh, max=1.0)
        
        # 3. 初始生成二值掩码 [B, 1, T, H, W]
        binary_mask = (soft_mask > dynamic_thresh).float()
        
        # =========================================================================
        # Part 2: Top-50 兜底逻辑
        # 为了方便遍历每一帧，这里再进行 View 操作
        # =========================================================================
        
        # 将 [B, 1, T, H, W] 展平为 [B*T, H*W] 以便循环处理
        # 注意：这里 view 出来的是原 tensor 的视窗，修改它会影响 binary_mask
        binary_mask_flat = binary_mask.view(B * T, -1)
        soft_mask_flat = soft_mask.view(B * T, -1)
        
        # 统计每一帧的有效点数
        valid_pts_count = torch.sum(binary_mask_flat, dim=-1) # [B*T]
        
        num_samples = B * T
        for i in range(num_samples):
            if valid_pts_count[i] < 50:
                # 该帧有效点不足50，取该帧的Top50
                single_soft_mask = soft_mask_flat[i, :] # [H*W]
                
                # 这里的 topk 是在 H*W 范围内找
                k = min(50, single_soft_mask.shape[0])
                _, topk_idx = torch.topk(single_soft_mask, k=k, dim=0, largest=True)
                
                # 重置该帧的 mask
                topk_mask = torch.zeros_like(single_soft_mask, device=device)
                topk_mask[topk_idx] = 1.0
                
                # 修改 flat 视图，原 binary_mask 也会被修改
                binary_mask_flat[i, :] = topk_mask

        self.print += 1
        
        # binary_mask 在 loop 中已被原地修改，直接返回即可
        # 形状依然是 [B, 1, T, H, W]
        return binary_mask

    def forward(self, batch):
        device = batch['input'].device
        b, c, t, h, w = batch['input'].shape
        
        ################################运行Net1##################################
        if self.net1name == 'UNet3DwithZZLB':
            voxel_features = self.I2PNet(batch) # B 3 T H W --> B 1 T H W
        else:
            voxel_features = self.I2PNet(batch['input'],) # B 3 T H W --> B 1 T H W
        ##########################################################################
        
        ##########################################################################
        z = {}
        for head in self.heads:
            # --- 关键修改 2: 输入改为 dense_input_3d ---
            # 这里的 self.__getattr__(head) 应该是您之前修改好的 nn.Conv3d 模块
            out_dense = self.__getattr__(head)(voxel_features)

            if 'hm' in head:
                # --- 关键修改 3: 激活函数处理 ---
                # 原代码是处理 sparse features，现在直接处理 dense tensor
                # 假设 self.sigmoid 是 nn.Sigmoid()，或者使用 torch.sigmoid
                out_dense = torch.sigmoid(out_dense) 
                
                # Clamp 操作
                # out_dense = torch.clamp(out_dense, min=0, max=1)
            
            # else 分支不需要特殊处理，out_dense 已经是结果了
            
            # 存入结果
            z[head] = out_dense
        

        return [z]


def I2PSOD(heads, image_size = [512,512], img_num = 20, layers=4, thresh=None,input_channels=1,feat_channels=[16,32,64],T_pooling=False,groups=2,downsample_mode='maxpool',net1name='UNet3D'):
    model =Img2PointsSmallObjectDetection(heads,  image_size = image_size, img_num = img_num, 
                                          layers=layers, thresh=thresh, input_channels=input_channels,
                                          feat_channels=feat_channels,T_pooling=T_pooling,
                                          groups=groups,downsample_mode=downsample_mode,net1name=net1name)
    return model



if __name__ == '__main__':
    import time
    import torch
    from thop import profile
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
            heads=heads, image_size = [512, 512], img_num = 10, layers=3.6, thresh=None,net1name='UNet3DWithGroupedMultiBranch'
        ).to(device).eval()
        # print("\n=== 网络结构概要 ===")
        # print(model)  # 打印完整模型结构
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

        # # 参数量/计算量计算（thop）
        if profile is not None:
            flops, params = profile(model, inputs=(batch,), verbose=False)
            # Params 通常以 M (Million) 为单位
            print(f"Total Parameters: {params / 1e6:.8f} M") 
            # FLOPs 通常以 G (Billion) 为单位
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
        