import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 1. 数据加载（MNIST 是 torchvision 内置数据集）
print("\n1. 加载 MNIST 数据")

transform = transforms.Compose([
    transforms.ToTensor(),  # PIL -> Tensor, [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准归一化
])

# 自动下载到 ./data 文件夹
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"训练集: {len(train_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")

# 2. 定义模型
print("\n2. 定义模型")


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()  # (1,28,28) -> (784,)
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 10)  # 10 类输出

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


model = MNISTNet().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# 3. 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 4. 训练循环
print("\n3. 开始训练")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # 前向
        output = model(data)
        loss = criterion(output, target)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = output.max(1)  # 取最大值的索引作为预测
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")

    acc = 100. * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度，节省内存
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


# 训练 5 个 epoch
epochs = 5
for epoch in range(1, epochs + 1):
    print(f"\n--- Epoch {epoch}/{epochs} ---")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

# 5. 保存模型
print("\n4. 保存模型")
torch.save(model.state_dict(), 'mnist_model.pth')
print("模型已保存为 mnist_model.pth")

# 6. 推理测试
print("\n5. 推理测试")
model.eval()
sample_data, sample_target = test_dataset[0]
sample_input = sample_data.unsqueeze(0).to(device)  # 增加 batch 维度
with torch.no_grad():
    output = model(sample_input)
    pred = output.argmax(1).item()
print(f"第0张图真实标签: {sample_target}, 预测: {pred}")