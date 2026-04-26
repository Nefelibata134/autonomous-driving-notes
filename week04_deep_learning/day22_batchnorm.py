import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== BatchNorm 与 Dropout ===")

# 1. BatchNorm2d（加速训练）
bn = nn.BatchNorm2d(32).to(device)
x = torch.randn(16, 32, 28, 28).to(device)
out = bn(x)
print(f"BatchNorm:")
print(f"  输入均值: {x.mean():.3f}, 输出均值: {out.mean():.3f}（接近0）")
print(f"  输入标准差: {x.std():.3f}, 输出标准差: {out.std():.3f}（接近1）")

# 2. Dropout（防止过拟合）
dropout = nn.Dropout(0.5).to(device)
x = torch.ones(2, 10).to(device)
dropout.train()
out_train = dropout(x)
print(f"\nDropout训练模式（部分为0）: {out_train[0][:5]}")

dropout.eval()
out_eval = dropout(x)
print(f"Dropout评估模式（全1）: {out_eval[0][:5]}")

# 3. 全局平均池化（GAP）替代全连接
gap = nn.AdaptiveAvgPool2d(1)
x = torch.randn(4, 512, 7, 7).to(device)
out = gap(x)
print(f"\nGAP: {x.shape} -> {out.shape}")  # (4, 512, 1, 1)
out_flat = out.view(4, -1)
print(f"展平后: {out_flat.shape}")  # (4, 512)