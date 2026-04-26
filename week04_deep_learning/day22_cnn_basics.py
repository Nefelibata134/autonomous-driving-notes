import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

print("=== CNN 基础 ===")

# 1. Conv2d 参数详解
print("\n1. Conv2d")
conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1).to(device)
#                                                    加上 .to(device) ↑
print(f"Conv2d(1,32,3,1,1): 权重 {conv1.weight.shape}")

x = torch.randn(1, 1, 28, 28).to(device)
out = conv1(x)
print(f"输入: {x.shape} -> 输出: {out.shape}")

# MNIST 输入: (batch=1, channel=1, H=28, W=28)
x = torch.randn(1, 1, 28, 28).to(device)
out = conv1(x)
print(f"输入: {x.shape} -> 输出: {out.shape}")  # (1, 32, 28, 28) padding=1保持尺寸

# 2. 输出尺寸公式（必背！）
print("\n2. 尺寸计算公式")
# Output = (Input - Kernel + 2*Padding) / Stride + 1
print(f"28x28, k=3, p=1, s=1 -> {(28 - 3 + 2 * 1) // 1 + 1}")  # 28
print(f"28x28, k=3, p=0, s=2 -> {(28 - 3 + 0) // 2 + 1}")  # 13
print(f"14x14, k=3, p=1, s=1 -> {(14 - 3 + 2 * 1) // 1 + 1}")  # 14

# 3. MaxPool2d（降维）
print("\n3. MaxPool2d")
pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2池化，尺寸减半
out_pool = pool(out)
print(f"Pool前: {out.shape} -> 后: {out_pool.shape}")  # (1, 32, 14, 14)


# 4. 标准 CNN 块（Conv + BN + ReLU + Pool）
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


block = ConvBlock(1, 32).to(device)
test = block(x)
print(f"\nConvBlock(1->32): {x.shape} -> {test.shape}")  # (1, 32, 14, 14)