import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 1. 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)


# 2. CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # 第1层: 1x28x28 -> 32x28x28 -> 32x14x14
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 第2层: 32x14x14 -> 64x14x14 -> 64x7x7
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 第3层: 64x7x7 -> 128x7x7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 128*7*7 = 6272
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 3. 损失与优化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


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
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return total_loss / len(test_loader), 100. * correct / total


# 5. 训练 5 个 epoch
print("\n开始训练...")
for epoch in range(1, 6):
    train_loss, train_acc = train_epoch()
    test_loss, test_acc = evaluate()
    scheduler.step()
    print(f"Epoch {epoch}: Train={train_acc:.2f}% | Test={test_acc:.2f}%")

# 6. 保存模型
torch.save(model.state_dict(), 'cnn_mnist.pth')
print("\n模型已保存: cnn_mnist.pth")

# 7. 推理测试
model.eval()
sample, label = test_data[0]
with torch.no_grad():
    pred = model(sample.unsqueeze(0).to(device)).argmax(1).item()
print(f"样本0: 真实={label}, 预测={pred}")