import torch
import torch.nn as nn
import torch.nn.functional as F

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

class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel_list=[32, 64, 128, 256, 512], num_classes=1):
        super(UNet, self).__init__()
        c = out_channel_list # [32, 64, 128, 256, 512]
        
        # --- Encoder (Contracting Path) ---
        self.encoder1 = nn.Sequential(
            BasicConv3d(in_channel, c[0], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[0], c[0], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[0], c[0], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.encoder2 = nn.Sequential(
            BasicConv3d(c[0], c[1], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[1], c[1], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[1], c[1], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.encoder3 = nn.Sequential(
            BasicConv3d(c[1], c[2], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[2], c[2], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[2], c[2], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.encoder4 = nn.Sequential(
            BasicConv3d(c[2], c[3], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[3], c[3], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[3], c[3], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.encoder5 = nn.Sequential(
            BasicConv3d(c[3], c[4], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[4], c[4], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[4], c[4], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        
        # --- Up Modules (1x1 conv to reduce channels) ---
        self.up5 = nn.Conv3d(c[4], c[3], kernel_size=1) # 512 -> 256
        self.up4 = nn.Conv3d(c[3], c[2], kernel_size=1) # 256 -> 128
        self.up3 = nn.Conv3d(c[2], c[1], kernel_size=1) # 128 -> 64
        self.up2 = nn.Conv3d(c[1], c[0], kernel_size=1) # 64 -> 32
        
        # --- Decoder (Expanding Path) ---
        self.decoder5 = nn.Sequential(
            BasicConv3d(c[3] * 2, c[3], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[3], c[3], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[3], c[3], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.decoder4 = nn.Sequential(
            BasicConv3d(c[2] * 2, c[2], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[2], c[2], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[2], c[2], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.decoder3 = nn.Sequential(
            BasicConv3d(c[1] * 2, c[1], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[1], c[1], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[1], c[1], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.decoder2 = nn.Sequential(
            BasicConv3d(c[0] * 2, c[0], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[0], c[0], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(c[0], c[0], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.decoder1 = nn.Sequential(
            BasicConv3d(c[0], c[0], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(c[0], c[0], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)), 
        )

        self.final_conv = nn.Conv3d(c[0], num_classes, kernel_size=1)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)


    def forward(self, x):
        # 假设输入 x: [N, 3, D, 512, 512], 其中 D = seqlen
        original_size = x.shape[2:] 
        
        # ----------------------
        # ENCODER PATH
        # ----------------------
        
        # --- Stage 1 ---
        e1 = self.encoder1(x) 
        # e1: [N, 32, D, 512, 512]
        out = F.max_pool3d(e1, kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        # out: [N, 32, D, 256, 256] (注意 D 维度不变)
        t1 = out
        
        # --- Stage 2 ---
        e2 = self.encoder2(out)
        # e2: [N, 64, D, 256, 256]
        out = F.max_pool3d(e2, kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        # out: [N, 64, D, 128, 128]
        t2 = out
        
        # --- Stage 3 ---
        e3 = self.encoder3(out)
        # e3: [N, 128, D, 128, 128]
        out = F.max_pool3d(e3, kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        # out: [N, 128, D, 64, 64]
        t3 = out
        
        # --- Stage 4 ---
        e4 = self.encoder4(out)
        # e4: [N, 256, D, 64, 64]
        out = F.max_pool3d(e4, kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        # out: [N, 256, D, 32, 32]
        t4 = out
        
        # --- Stage 5 (Bottleneck) ---
        out = self.encoder5(out) 
        # out: [N, 512, D, 32, 32]
        
        
        # ----------------------
        # DECODER PATH
        # ----------------------
        
        # --- Block 5 -> 4 ---
        # 1. 降维: 512 -> 256
        u5 = self.up5(out)  
        # u5: [N, 256, D, 32, 32]
        # 2. 上采样: 32 -> 32 (此处与 t4 尺寸一致，interpolate 实际上是 identity 或为了对齐)
        u5 = F.interpolate(u5, size=t4.shape[2:], mode='trilinear', align_corners=True)
        # 3. 拼接: [N, 256, ...] + [N, 256, ...] -> [N, 512, D, 32, 32]
        out = torch.cat([u5, t4], dim=1)
        # 4. 解码卷积: [N, 512, ...] -> [N, 256, D, 32, 32]
        out = self.decoder5(out)

        # --- Block 4 -> 3 ---
        # 1. 降维: 256 -> 128
        u4 = self.up4(out)
        # u4: [N, 128, D, 32, 32]
        # 2. 上采样: 32 -> 64 (size match t3)
        u4 = F.interpolate(u4, size=t3.shape[2:], mode='trilinear', align_corners=True)
        # u4: [N, 128, D, 64, 64]
        # 3. 拼接: [N, 128, ...] + [N, 128, ...] -> [N, 256, D, 64, 64]
        out = torch.cat([u4, t3], dim=1)
        # 4. 解码卷积: [N, 256, ...] -> [N, 128, D, 64, 64]
        out = self.decoder4(out)
        
        # --- Block 3 -> 2 ---
        # 1. 降维: 128 -> 64
        u3 = self.up3(out)
        # u3: [N, 64, D, 64, 64]
        # 2. 上采样: 64 -> 128 (size match t2)
        u3 = F.interpolate(u3, size=t2.shape[2:], mode='trilinear', align_corners=True)
        # u3: [N, 64, D, 128, 128]
        # 3. 拼接: [N, 64, ...] + [N, 64, ...] -> [N, 128, D, 128, 128]
        out = torch.cat([u3, t2], dim=1)
        # 4. 解码卷积: [N, 128, ...] -> [N, 64, D, 128, 128]
        out = self.decoder3(out)

        # --- Block 2 -> 1 ---
        # 1. 降维: 64 -> 32
        u2 = self.up2(out)
        # u2: [N, 32, D, 128, 128]
        # 2. 上采样: 128 -> 256 (size match t1)
        u2 = F.interpolate(u2, size=t1.shape[2:], mode='trilinear', align_corners=True)
        # u2: [N, 32, D, 256, 256]
        # 3. 拼接: [N, 32, ...] + [N, 32, ...] -> [N, 64, D, 256, 256]
        out = torch.cat([u2, t1], dim=1)
        # 4. 解码卷积: [N, 64, ...] -> [N, 32, D, 256, 256]
        out = self.decoder2(out)
        
        # --- Final Restoration ---
        # 直接上采样至原始输入尺寸 (256 -> 512)
        u1 = F.interpolate(out, size=original_size, mode='trilinear', align_corners=True)
        # u1: [N, 32, D, 512, 512]
        
        # 最终特征提取
        out = self.decoder1(u1)
        # out: [N, 32, D, 512, 512]
        
        # --- Output Head ---
        out = self.final_conv(out)
        # out: [N, 1, D, 512, 512]
        
        # out = self.activation(out)
        
        return out
    
if __name__ == '__main__':
    # 引入 thop 库用于统计
    from thop import profile
    # 简单的测试代码
    channel_list = [32, 64, 128, 256, 512]
    num_classes = 1
    seqlen = 5
    image_size = 512
    
    net = UNet(out_channel_list=channel_list, num_classes=num_classes)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0') # 测试用 cpu 即可
    net.to(device)
    
    x = torch.randn(1, 3, seqlen, image_size, image_size).to(device)
    
    # --- 核心修改：增加统计代码 ---
    print("Start analysing model complexity...")
    # thop.profile 会打印详细的层级信息，如果不想看可以重定向输出或忽略
    # inputs 需要是一个 tuple
    flops, params = profile(net, inputs=(x, ))
    
    print("\n" + "="*30)
    print(f"Input Shape: {x.shape}")
    # Params 通常以 M (Million) 为单位
    print(f"Total Parameters: {params / 1e6:.2f} M") 
    # FLOPs 通常以 G (Billion) 为单位
    print(f"Total FLOPs (MACs): {flops / 1e9:.2f} G")
    print("="*30 + "\n")
    # ---------------------------

    # 无论是 training 还是 inference，现在都只返回一个输出
    out = net(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
