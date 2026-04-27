import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== CNN 架构演进 ===")

# 练习 1：LeNet-5（1998，CNN 鼻祖）
print("\n1. LeNet-5（5层，手写数字识别）")


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # 1x32x32 -> 6x28x28
            nn.Tanh(),
            nn.AvgPool2d(2),  # 6x14x14
            nn.Conv2d(6, 16, kernel_size=5),  # 16x10x10
            nn.Tanh(),
            nn.AvgPool2d(2),  # 16x5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


lenet = LeNet5().to(device)
print(f"LeNet-5 参数量: {sum(p.numel() for p in lenet.parameters()):,}")

# 练习 2：AlexNet（2012，ImageNet 冠军，深度学习爆发点）
print("\n2. AlexNet（8层，ReLU+Dropout+GPU并行）")
alexnet = models.alexnet(weights=None).to(device)  # 不加载预训练权重
print(f"AlexNet 参数量: {sum(p.numel() for p in alexnet.parameters()):,}")

# 练习 3：VGG（2014，小卷积核 3x3 堆叠，16-19 层）
print("\n3. VGG-16（16层，3x3卷积堆叠）")
vgg16 = models.vgg16(weights=None).to(device)
print(f"VGG-16 参数量: {sum(p.numel() for p in vgg16.parameters()):,}")
print("VGG 特点：层数深，但参数量巨大（138M），计算慢")

# 练习 4：ResNet（2015，Skip Connection，152+层）
print("\n4. ResNet-18（18层，残差连接）")
resnet18 = models.resnet18(weights=None).to(device)
print(f"ResNet-18 参数量: {sum(p.numel() for p in resnet18.parameters()):,}")
print("ResNet 特点：Skip Connection 解决梯度消失，可以训练非常深的网络")

# 参数量对比
print("\n=== 参数量对比 ===")
print(f"LeNet-5:  ~60K")
print(f"AlexNet:  ~60M")
print(f"VGG-16:   ~138M")
print(f"ResNet-18: ~11M（比VGG小但更深！）")
