import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 1. 数据加载（CIFAR-10，3通道32x32，ResNet可以处理）
print("\n1. 加载 CIFAR-10")
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet 期望 224x224，放大（或保持32用AdaptivePool）
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 均值
])

# 为加速学习，只用部分数据（演示用）
train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# 子集：每类只取 500 张（共 5000 张），模拟小数据集场景
from torch.utils.data import Subset
import numpy as np


def get_subset(dataset, num_per_class=500):
    targets = np.array(dataset.targets)
    indices = []
    for i in range(10):
        idx = np.where(targets == i)[0][:num_per_class]
        indices.extend(idx.tolist())
    return Subset(dataset, indices)


train_subset = get_subset(train_data, 500)
test_subset = get_subset(test_data, 100)

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=100, shuffle=False)

print(f"训练子集: {len(train_subset)} 张, 测试子集: {len(test_subset)} 张")

# 2. 创建模型（迁移学习版）
print("\n2. 创建预训练 ResNet18")
model = models.resnet18(weights='DEFAULT')

# 冻结特征层
for param in model.parameters():
    param.requires_grad = False

# 替换分类头（CIFAR-10 有 10 类）
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model = model.to(device)

# 3. 训练配置
criterion = nn.CrossEntropyLoss()
# 只优化可训练参数（最后一层）
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


# 4. 训练函数
def train_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
    return total_loss / len(train_loader), 100. * correct / total


def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return 100. * correct / total


# 5. 训练（小数据集 + 预训练，5 个 epoch 就够了）
print("\n3. 开始训练（只训练最后一层）")
epochs = 5
for epoch in range(1, epochs + 1):
    loss, acc = train_epoch()
    test_acc = evaluate()
    print(f"Epoch {epoch}: Loss={loss:.4f}, Train={acc:.1f}%, Test={test_acc:.1f}%")

# 6. 对比：从头训练的 ResNet（可选，慢很多）
print("\n4. 保存迁移学习模型")
torch.save(model.state_dict(), 'resnet18_transfer.pth')
print("模型已保存")

# 7. 推理测试
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
model.eval()
sample, label = test_data[0]
with torch.no_grad():
    pred = model(sample.unsqueeze(0).to(device)).argmax(1).item()
print(f"\n样本0: 真实={classes[label]}, 预测={classes[pred]}")