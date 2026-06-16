import torch
import torch.nn as nn
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

from lib.models.spconv_unet import UNetV2, UNetV2_3, UNetV2_2, UNetV2_3_32, UNetV2_3_T_nodown, UNetV2_3_T_nodown_maxpool, UNetV2_3_T_nodown_v2, UNetV2_3_T_nodown_v3
from lib.models.spconv_utils import replace_feature, spconv
from lib.models.noramlconv_unet3d2_1 import UNet2DWithNormalConv2D, UNet3DWithNormalConv3D, LightWeightedConv3D, TOSConvNet, TZSConvNet, DynamicTOSConvNet
from lib.utils1.show_one_img import show_one_img
import torch

class Net1Only(nn.Module):
    def __init__(self, heads, image_size = [512,512], img_num = 20, layers = 3, thresh=None, input_channels=1, 
                 feat_channels=[16,32,64,128], T_pooling=False,groups=1,downsample_mode='stride',
                 net1name='UNet3DWithNormalConv3D', opt=None):
        super().__init__()
        # points generate net
        self.net1name=net1name
        if net1name=='UNet3DWithNormalConv3D': # 实际使用
            self.I2PNet = UNet3DWithNormalConv3D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode, use_final_conv=False,TConvOnly=False)
        elif net1name=='UNet2DWithNormalConv2D':
            self.I2PNet = UNet2DWithNormalConv2D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode, use_final_conv=False, TConvOnly=False)
        elif net1name=='LightWeightedConv3D':
            self.I2PNet = LightWeightedConv3D(num_channels=3, num_classes=1, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode, use_final_conv=False)
        elif net1name=='TOSConvNet':
            self.I2PNet = TOSConvNet(num_channels=3, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode,
                                 use_final_conv=False, use_tzsconv=getattr(opt, 'use_tzsconv', True),
                                 seq_len=img_num)
        elif net1name in ('DynamicTOSConvNet', 'DynamicTOSconvNet'):
            self.I2PNet = DynamicTOSConvNet(num_channels=3, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode,
                                 use_final_conv=False, use_tzsconv=getattr(opt, 'use_tzsconv', True),
                                 seq_len=img_num)
        elif net1name in ('TZSConvNet', 'TZSconvNet'):
            self.I2PNet = TZSConvNet(num_channels=3, feat_channels=feat_channels, residual=None,
                                 upsample_mode="trilinear", activation=None,T_pooling=T_pooling,groups=groups,downsample_mode=downsample_mode,
                                 use_final_conv=False, use_tzsconv=getattr(opt, 'use_tzsconv', True),
                                 seq_len=img_num)
        else:
            print('net1name 错误！')
        self.sigmoid = nn.Sigmoid()
        
        head_conv=128
        
        head_input_channel = feat_channels[0]
        # 32 1 可以训练
        ###get head conv
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            # 原indice_key是稀疏卷积特有的，稠密卷积不需要，注释掉即可
            # name_1 = 'subm1'+head
            # name_2 = 'subm2'+head
            
            if head_conv > 0:
                if 'hm' in head:
                    # 替换为nn.Sequential + nn.Conv3d，移除indice_key，保留其他参数
                    fc = nn.Sequential(
                        nn.Conv3d(head_input_channel, head_conv, 3, padding=1, bias=False),
                        # nn.BatchNorm3d(head_conv),
                        nn.ReLU(),
                        nn.Conv3d(head_conv, classes, 3, padding=1, bias=True),
                    )
                else:
                    fc = nn.Sequential(
                        nn.Conv3d(head_input_channel, head_conv, 3, padding=1, bias=False),
                        # nn.BatchNorm3d(head_conv),
                        nn.ReLU(),
                        nn.Conv3d(head_conv, classes, 3, padding=1, bias=False),
                    )
            else:
                # 单卷积层场景，同样替换为nn.Conv3d，移除indice_key
                fc = nn.Conv3d(head_input_channel, classes, 3, padding=1, bias=True)
            
            # 保持hm头的bias初始化逻辑不变
            if 'hm' in head:
                # 无论单层/双层卷积，最后一层的bias都初始化为-2.19
                fc[-1].bias.data.fill_(-2.19)
            
            # 保持属性赋值逻辑不变
            self.__setattr__(head, fc)
        
    # def forward(self, batch):
    #     ################################运行Net1##################################
    #     if self.net1name == 'UNet3DwithZZLB':
    #         voxel_features = self.I2PNet(batch) # B 3 T H W --> B 1 T H W
    #     else:
    #         voxel_features = self.I2PNet(batch['input'],) # B 3 T H W --> B 1 T H W
    #     ##########################################################################
        
        
    #     z = {}
    #     for head in self.heads:
    #         spatial_features = self.__getattr__(head)(voxel_features)

    #         if 'hm' in head:
    #             spatial_features = self.sigmoid(spatial_features)
    #             spatial_features = torch.clamp(spatial_features, min=1e-4, max=1 - 1e-4)
    #         z[head] = spatial_features
    #     return [z]

    ###############################################滑窗推理######################################################
    def _forward_patch(self, batch, patch_w):
        """
        处理单个图像块的核心前向逻辑
        patch_w: 当前块的真实宽度 (用于确保展平和 SPConv 恢复时的尺寸正确)
        """
        device = batch['input'].device
        b, c, t, h, w = batch['input'].shape

        ################################运行Net1##################################
        voxel_features = self.I2PNet(batch['input'])
        ##########################################################################
        
        
        
        z = {}
        for head in self.heads:
            spatial_features = self.__getattr__(head)(voxel_features)
            if 'hm' in head:
                spatial_features = self.sigmoid(spatial_features)
                spatial_features = torch.clamp(spatial_features, min=1e-4, max=1 - 1e-4)
            z[head] = spatial_features
            
            # 关键：由于 sp_backbone 初始化时 grid_size 是全图尺寸，
            # dense() 还原出的特征图是全图宽度。我们需要将其截取为当前的 patch_w
            z[head] = spatial_features[..., :patch_w]
            
        return z

    def forward(self, batch):
        b, c, t, h, w = batch['input'].shape
        
        # 触发条件：非训练模式，且宽度足够大 (例如 1920)
        if not self.training and w >= 1920:
            w_half = w // 2
            
            # ========== 1. 处理左半部分 ==========
            batch_left = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_left['input'] = batch['input'][..., :w_half]
            
            with torch.no_grad():
                z_left = self._forward_patch(batch_left, patch_w=w_half)
            
            del batch_left
            torch.cuda.empty_cache()
            
            # ========== 2. 处理右半部分 ==========
            batch_right = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_right['input'] = batch['input'][..., w_half:]
            
            with torch.no_grad():
                # 注意右半部分的宽度是 w - w_half (处理奇数宽度的严谨写法)
                z_right = self._forward_patch(batch_right, patch_w=w - w_half)
                
            del batch_right
            torch.cuda.empty_cache()

            # ========== 3. 完美拼接输出 ==========
            z_merged = {}
            
            # (1) 拼接所有的密集特征图 (沿 Width 维度 dim=-1)
            for head in self.heads:
                z_merged[head] = torch.cat([z_left[head], z_right[head]], dim=-1)
                
            # z_merged['hm_large_heatmap'] = torch.cat([z_left['hm_large_heatmap'], z_right['hm_large_heatmap']], dim=-1)
            # z_merged['soft_mask'] = torch.cat([z_left['soft_mask'], z_right['soft_mask']], dim=-1)
            
            # # (2) 拼接稀疏点云坐标 (核心操作：右半图坐标的 W 需要加上偏移量)
            # coords_left = z_left['voxel_coords']
            # coords_right = z_right['voxel_coords']
            # # coords 形状为 [N, 4]，对应 [Batch, T, H, W]，因此索引 3 就是宽度 W
            # coords_right[:, 3] += w_half 
            # z_merged['voxel_coords'] = torch.cat([coords_left, coords_right], dim=0)
            # # ====================== 新增：计算采样率 ======================
            # total_points = b * t * h * w  # 理论上的所有点数
            # sampled_points = z_merged['voxel_coords'].shape[0] # 经过阈值筛选后的有效点数
            # sampling_rate = sampled_points / total_points if total_points > 0 else 0
            # # ====================== 新增：输出采样率 ======================
            # # 转成 Tensor 方便在外部与其他特征统一处理或收集
            # z_merged['sampling_rate'] = torch.tensor(sampling_rate) 
            # # ==============================================================
            return [z_merged]
            
        else:
            # ========== 正常模式 (训练时或小图) ==========
            z = self._forward_patch(batch, patch_w=w)
            # ====================== 新增：计算采样率 ======================
            # total_points = b * t * h * w  # 理论上的所有点数
            # sampled_points = z['voxel_coords'].shape[0] # 经过阈值筛选后的有效点数
            # sampling_rate = sampled_points / total_points if total_points > 0 else 0
            # # ====================== 新增：输出采样率 ======================
            # # 转成 Tensor 方便在外部与其他特征统一处理或收集
            # z['sampling_rate'] = torch.tensor(sampling_rate) 
            # ==============================================================
            return [z]
    ###############################################滑窗推理######################################################


def Net1(heads, image_size = [512,512], img_num = 20, layers=4, thresh=None,input_channels=1,feat_channels=[16,32,64],T_pooling=False,groups=2,downsample_mode='maxpool',net1name='UNet3D', opt=None):
    model =Net1Only(heads,  image_size = image_size, img_num = img_num, 
                                          layers=layers, thresh=thresh,
                                          feat_channels=feat_channels,T_pooling=T_pooling,
                                          groups=groups,downsample_mode=downsample_mode,net1name=net1name, opt=opt)
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
        model = Net1(
            heads=heads, image_size = [512, 512], img_num = 10, layers=3.61, thresh=3,net1name='UNet3DWithNormalConv3D',
            feat_channels=[8,16,32,64],
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
        
