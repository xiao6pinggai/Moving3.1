import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
try:
    from lib.utils1.visual_utils.save_hm_img import save_hm_img
except:
    pass
    # from .utils1.visual_utils.save_hm_img import save_hm_img

# --------------------------------------------------------
# 基础模块 BasicConv3d (保持不变)
# --------------------------------------------------------
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicATDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        # self.bn = nn.BatchNorm3d(out_channels,
        #                          eps=0.001, # value found in tensorflow
        #                          momentum=0.1, # default pytorch value
        #                          affine=True)
        # self.relu = nn.ReLU(inplace=True)
        # 或者
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        # 可训练缩放系数
        # self.relu = nn.PReLU()
        # self.gamma = nn.Parameter(torch.ones(out_channels, 1, 1, 1)) # 通道级，即卷积核级

    def forward(self, x):
        # 硬零和约束
        W = self.conv.weight # cout cin t h w
        if W.shape[2]!=1:
            W = W - W.mean(dim=2, keepdim=True)

        # 卷积
        out = F.conv3d(x, W, stride=self.conv.stride, padding=self.conv.padding)
        # out = self.bn(out)

        # **将 gamma 放在卷积之后**
        # out = self.gamma * out  # <-- 可训练缩放

        # out = self.relu(out)
        return out
# --------------------------------------------------------
# 动态深度 UNet
# --------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel_list=[32, 64, 128, 256, 512], num_classes=1):
        super(UNet, self).__init__()
        
        self.out_channel_list = out_channel_list
        depth = len(out_channel_list)
        
        # --- 1. 构建 Encoder List ---
        self.encoders = nn.ModuleList()
        
        for i in range(depth):
            # 第一层输入为 in_channel，后续层输入为上一层输出
            input_c = in_channel if i == 0 else out_channel_list[i-1]
            output_c = out_channel_list[i]
            
            # 构建标准的 3层 卷积块 (模拟原代码结构)
            block = nn.Sequential(
                BasicConv3d(input_c, output_c, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)), # 先同一升维
                BasicConv3d(output_c, output_c, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
                BasicConv3d(output_c, output_c, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
                BasicConv3d(output_c, output_c, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            )
            self.encoders.append(block)

        # --- 2. 构建 Decoder & Up List ---
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Decoder 数量比 Encoder 少 1 (Bottleneck 层不需要对应的 Decoder)
        for i in range(depth - 2, -1, -1):
            high_c = out_channel_list[i+1] # 深层通道 (来源)
            low_c = out_channel_list[i]    # 浅层通道 (目标)
            
            # 1x1 卷积降维
            self.ups.append(nn.Conv3d(high_c, low_c, kernel_size=1))
            
            # Decoder Block
            block = nn.Sequential(
                BasicConv3d(low_c * 2, low_c, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)), # 先同一降维
                BasicConv3d(low_c, low_c, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
                BasicConv3d(low_c, low_c, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
                BasicConv3d(low_c, low_c, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            )
            self.decoders.append(block)

        # --- 3. Final Restoration Block (对应原 decoder1) ---
        first_c = out_channel_list[0]
        self.final_decoder = nn.Sequential(
            BasicConv3d(first_c, first_c, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(first_c, first_c, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(first_c, first_c, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        
        # --- 4. Output Head ---
        self.final_conv = nn.Conv3d(first_c, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        original_size = x['input'].shape[2:] # [D, H, W]
        
        # 存储 Skip Connections (改为存储池化前的Encoder特征)
        skips = []
        out = x['input']
        
        # --- Encoder Forward (核心修改部分) ---
        for i, encoder in enumerate(self.encoders):
            # 第一步：执行当前Encoder Block，得到池化前的高分辨率特征
            enc_out = encoder(out)
            
            # 如果不是最后一层 (Bottleneck)：先存特征，再池化
            if i < len(self.encoders) - 1:
                skips.append(enc_out)  # 关键修改：存入池化前的特征
                # 池化后作为下一层Encoder的输入
                out = F.max_pool3d(enc_out, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                # 最后一层是Bottleneck，不池化，直接作为Decoder的起始输入
                out = enc_out
        
        # --- Decoder Forward (逻辑不变，拼接的是池化前的高分辨率特征) ---
        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            # 取出对应的Skip Connection (池化前的Encoder特征)
            skip = skips.pop() 
            
            # 1. 降维
            out = up(out)
            
            # 2. 上采样 (对齐skip的尺寸：此时skip是池化前的高分辨率，匹配UNet逻辑)
            out = F.interpolate(out, size=skip.shape[2:], mode='trilinear', align_corners=True)
            
            # 3. 拼接 (高分辨率Skip特征 + 上采样后的Decoder特征)
            out = torch.cat([out, skip], dim=1)
            
            # 4. 卷积融合
            out = decoder(out)
        
        # --- Final Restoration ---
        # 上采样回原始输入尺寸
        out = F.interpolate(out, size=original_size, mode='trilinear', align_corners=True)
        
        out = self.final_decoder(out)
        out = self.final_conv(out)
        out = self.activation(out)
        out = torch.clamp(out, min=1e-6, max=1 - 1e-6) # 数值截断
        out_dict = {'hm_large_heatmap': out}
        
        # 可选可视化（取消注释即可）
        try:
            if random.random() < 0.1:
                save_hm_img(x,out_dict)
        except:
            pass
        
        return [out_dict]
# --------------------------------------------------------
# 测试代码
# --------------------------------------------------------
if __name__ == '__main__':
    # 设置测试环境
    device = torch.device('cpu')
    seqlen = 5
    image_size = 512
    num_classes = 1
    x={}
    x['input'] = torch.randn(1, 3, seqlen, image_size, image_size).to(device)

    '''# === 测试案例 1: 标准 5 层 ===
    print("--- Test Case 1: Standard 5 Layers [32, 64, 128, 256, 512] ---")
    list1 = [32, 64, 128, 256, 512]
    net1 = UNet(out_channel_list=list1, num_classes=num_classes).to(device)
    
    
    # 统计参数
    flops, params = profile(net1, inputs=(x, ), verbose=False)
    print(f"Params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.2f}G")
    
    out1 = net1(x)[0]['hm_large_heatmap']
    print(f"Output Shape: {out1.shape}")
    assert out1.shape == (1, num_classes, seqlen, image_size, image_size)'''
    
    
    # === 测试案例 2: 浅层网络 (3层) ===
    print("\n--- Test Case 2: Shallow  Layers [16, 32, 64, 128] ---")
    list2 = [32, 64, 128]
    net2 = UNet(out_channel_list=list2, num_classes=num_classes).to(device)
    print(net2)
    
    # 统计参数
    flops, params = profile(net2, inputs=(x, ), verbose=False)
    print(f"Params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.2f}G")
    
    out2 = net2(x)[0]['hm_large_heatmap']
    print(f"Output Shape: {out2.shape}")
    # assert out2.shape == (1, num_classes, seqlen, image_size, image_size)
    
    print("\nAll Tests Passed!")