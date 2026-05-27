import torch
import torch.nn as nn
from functools import partial
import numpy as np
import sys
ROOT_DIR = "/root/autodl-tmp/Moving3.1"
# 确保根目录在sys.path首位（覆盖默认的子目录）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from lib.models.spconv_unet import UNetV2, UNetV2_3, UNetV2_2
from lib.models.spconv_utils import replace_feature, spconv

class spcenterDet(nn.Module):
    def __init__(self, heads, image_size = [512,512], img_num = 20, layers = 4, thresh=None):
        super().__init__()
        self.thresh=thresh
        input_channels = 4
        head_conv=128
        grid_size = np.array([image_size[1], image_size[0], img_num - 1])
        self.points_all = img_num*image_size[0]*image_size[1]
        if  layers==4:
            self.sp_backbone = UNetV2(input_channels, grid_size)
        elif layers==3:
            self.sp_backbone = UNetV2_3(input_channels, grid_size)
        elif layers == 2:
            self.sp_backbone = UNetV2_2(input_channels, grid_size)
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

    def preprocess(self, img, img_gray):
        mask_all = torch.zeros_like(img_gray)
        diff = img_gray - torch.median(img_gray[:,:,::3], 2)[0].unsqueeze(2)
        diff = abs(diff)
        # if diff.dtype != torch.float32: # lhg
        #     diff = diff.float()
        #####
        diff0 = diff.clone()
        std = torch.std(diff, [-2, -1]).unsqueeze(-1).unsqueeze(-1)
        mean = torch.mean(diff, [-2, -1]).unsqueeze(-1).unsqueeze(-1)
        if self.thresh is not None:
            lr_th = mean + self.thresh * std
        else:
            lr_th = mean + 3 * std
        diff = self.relu(diff-lr_th)
        coords = torch.nonzero(diff.squeeze(1))
        img1 = torch.cat([img, diff], 1)
        features = img1[coords[:,0],:,coords[:,1], coords[:,2], coords[:,3]]
        # print(features.shape[0]/1024/1024/20*100)
        coords = coords.contiguous()
        batch_dict = {}
        batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = coords.to(features.device)
        batch_dict['batch_size'] = img_gray.shape[0]
        del img, img_gray
        return batch_dict, diff0, mask_all

    def forward(self, batch):
        # print(self.tau1)
        _, _, _, h, w = batch['input'].shape
        batch_dict, diff0, mask_all = self.preprocess(batch['input'], batch['input_gray'])
        sp_backbone_out = self.sp_backbone(batch_dict)
        z = {}
        for head in self.heads:# self.heads={'hm': 1, 'wh': 2, 'reg': 2}    head='hm' / 'wh' / 'reg'
            input_sp_tensor = sp_backbone_out['encoded_spconv_tensor']
            # print("input_sp_tensor shape:", input_sp_tensor.spatial_shape)
            # print("input_sp_tensor indices max:", input_sp_tensor.indices.max())
            # print("input_sp_tensor indices min:", input_sp_tensor.indices.min())
            out_h = self.__getattr__(head)(input_sp_tensor) # 魔法函数之一，动态获取当前类中预先定义的 hm、wh、reg 检测头模块（这些模块是网络层，比如卷积层 / 卷积块），进而调用这些模块完成特征预测。
            if 'hm' in head:
                out_h = replace_feature(out_h, self.sigmoid(out_h.features)) # 激活为概率
                spatial_features = out_h.dense() # 稀疏张量转换成密集特征图
                spatial_features = torch.clamp(spatial_features, min=1e-4, max=1 - 1e-4) # 数值截断，保证热力图的数值合理性，为后续损失计算（如 Focal Loss）提供稳定输入，防止log(0)出现极值。
            else:
                spatial_features = out_h.dense()
            z[head] = spatial_features # z['hm']存放激活的概率特征 z['wh'] z['reg']存放转换成dense()的特征
        z['mask_all'] = diff0
        z['voxel_coords'] = batch_dict['voxel_coords']
        z['lasso'] = torch.sum(mask_all, dim=[-1,-2]) / (h * w)
        return [z] # z: {'hm': hm, 'wh': wh, 'reg': reg, 'mask_all': diff0, 'voxel_coords': voxel_coords, 'lasso': lasso}

def sp_centerDet_minus(heads, image_size = [512,512], img_num = 20, layers=4, thresh=None):
    model = spcenterDet(heads,  image_size = image_size, img_num = img_num, layers=layers, thresh=thresh)
    return model

if __name__ == '__main__':
    import time
    import torch
    try:
        from thop import profile, clever_format
    except ImportError:
        print("提示：未安装thop（pip install thop），跳过参数量/计算量计算")
        profile = None
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")


    # 初始化模型
    model = sp_centerDet_minus({'hm':1, 'wh':2, 'reg': 2}, image_size = [512,512], img_num = 20, layers=3, thresh=3).to(device).eval()
    # print("\n=== 网络结构概要 ===")
    # print(model)  # 打印完整模型结构
    # 测试输入：[B, C, D, H, W] = [1, 3, 5, 512, 512]（D=5为时间维度）
    # test_input = torch.randn(1, 3, 5, 512, 512).to(device)  # 移除过时的Variable
    batch={}
    batch['input'] = torch.randn(1, 3, 20, 512, 512).to(device)
    batch['input_gray'] = torch.randn(1, 1, 20, 512, 512).to(device)
    # 前向传播 & 推理耗时
    start_time = time.time()
    # with torch.no_grad():
    #     output = model(test_input)
    model.train() # 切换到训练模式以启用梯度计算检查显存占用
    output = model(batch)
    infer_time = time.time() - start_time

    # 基础信息打印
    # print(f"输入尺寸: {test_input.shape}")
    # print(f"输出尺寸: {output.shape}")
    print(f"推理耗时: {infer_time:.4f}s")
    # assert output.shape[2] == test_input.shape[2], "时间维度（D）尺寸被错误修改！"

    # 参数量/计算量计算（thop）
    if profile is not None:
        flops, params = profile(model, inputs=(batch,), verbose=False)
        # Params 通常以 M (Million) 为单位
        print(f"Total Parameters: {params / 1e6:.8f} M") 
        # FLOPs 通常以 G (Billion) 为单位
        print(f"Total FLOPs (MACs): {flops / 1e9:.8f} G")
        
        # 网络结构打印（简易版，替代torchsummaryX）
        
        # 可选：仅打印层结构统计
        # total_layers = sum(1 for _ in model.named_modules() if not isinstance(getattr(model, _.split('.')[0]), Module) or _.count('.') == 0)
        # print(f"模型总层数（顶层）: {total_layers}")
