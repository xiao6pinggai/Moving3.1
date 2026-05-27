import torch
import torch.nn as nn

def reset_and_freeze_model(model):
    """
    1. 冻结 I2PNet 部分的参数。
    2. 重置除 I2PNet 外所有模块的参数（包含稀疏卷积）。
    3. 激活除 I2PNet 外所有模块的梯度。
    """
    
    # --- 第一步：定义权重初始化函数 (用于重置) ---
    def init_weights(m):
        # 尝试调用模块自带的 reset_parameters 方法 (最原生的重置)
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        
        # 或者使用自定义初始化 (更常用，特别是对于ReLU网络)
        # 这里使用 Kaiming Initialization，适用于 Conv, Linear, 以及大多数 SparseConv
        elif hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # 针对 Normalization 层 (BN, GN, LN)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    print("正在处理模型参数...")

    # --- 第二步：遍历模块进行 重置 (Reset) ---
    # 注意：我们遍历 named_modules 来重置层，而不是遍历 parameters
    for name, module in model.named_modules():
        # 跳过根节点
        if name == "": 
            continue
            
        # 逻辑：如果 "I2PNet" 不在名字里，说明这是需要重置的部分
        # 我们只重置 "叶子节点" (即包含具体权重的层，而不是容器)
        if "I2PNet" not in name:
            # 只有当模块有参数时才重置，避免对 Sequential 等容器报错
            if len(list(module.parameters(recurse=False))) > 0:
                init_weights(module)
                # 可选：打印一下确认重置了哪些关键模块
                # print(f"  [Reset] 重置模块: {name} ({type(module).__name__})")

    # --- 第三步：遍历参数进行 冻结/解冻 (Freeze/Unfreeze) ---
    frozen_count = 0
    active_count = 0
    
    for name, param in model.named_parameters():
        if "I2PNet" in name:
            # 属于 I2PNet -> 冻结，保留预训练权重
            param.requires_grad = False
            frozen_count += 1
        else:
            # 不属于 I2PNet -> 解冻，参与训练
            param.requires_grad = True
            active_count += 1

    print(f"处理完成:\n"
          f"  - 冻结参数组 (I2PNet): {frozen_count} 个 (保留预训练值)\n"
          f"  - 重置并激活参数组 (其他): {active_count} 个 (已重新初始化)")
    return model

# --- 调用 ---
# 假设 model 是你的网络实例
# reset_and_freeze_model(model)