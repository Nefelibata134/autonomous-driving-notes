import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 练习 1：理解 nn.Linear（全连接层）
print("\n=== 1. nn.Linear ===")
# nn.Linear(in_features, out_features)
# 数学: y = x @ W^T + b
linear = nn.Linear(in_features=10, out_features=5)
print(f"权重形状: {linear.weight.shape}")   # (5, 10)
print(f"偏置形状: {linear.bias.shape}")     # (5,)

# 输入一个样本
x = torch.randn(1, 10)  # (batch_size=1, features=10)
out = linear(x)
print(f"输入: {x.shape} -> 输出: {out.shape}")  # (1, 5)

# 输入一个 batch
x_batch = torch.randn(32, 10)  # (32, 10)
out_batch = linear(x_batch)
print(f"Batch 输入: {x_batch.shape} -> 输出: {out_batch.shape}")  # (32, 5)

# 练习 2：激活函数（非线性变换）
print("\n=== 2. 激活函数 ===")
x = torch.linspace(-3, 3, 100)

# ReLU: max(0, x) - 最常用，缓解梯度消失
relu = nn.ReLU()
print(f"ReLU(-1) = {relu(torch.tensor(-1.0))}, ReLU(2) = {relu(torch.tensor(2.0))}")

# Sigmoid: 1/(1+e^(-x)) - 二分类输出
sigmoid = nn.Sigmoid()
print(f"Sigmoid(0) = {sigmoid(torch.tensor(0.0))}")

# Tanh: (e^x - e^(-x))/(e^x + e^(-x))
tanh = nn.Tanh()

# Softmax: 多分类概率（输出和为1）
softmax = nn.Softmax(dim=1)
logits = torch.tensor([[1.0, 2.0, 3.0]])
probs = softmax(logits)
print(f"Softmax 输出: {probs}, 和: {probs.sum().item()}")

# 练习 3：nn.Sequential（层堆叠）
print("\n=== 3. nn.Sequential ===")
simple_net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.randn(64, 784)  # 64张 flattened 28x28 图像
out = simple_net(x)
print(f"Sequential 输出: {out.shape}")  # (64, 10)

# 练习 4：自定义网络（继承 nn.Module）
print("\n=== 4. 自定义网络 ===")


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        # 定义层
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 防止过拟合
        self.layer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 前向传播（必须实现！）
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


model = SimpleClassifier(input_dim=784, hidden_dim=256, num_classes=10).to(device)
print(f"模型结构:\n{model}")

# 测试前向传播
x = torch.randn(32, 784).to(device)
out = model(x)
print(f"模型输出: {out.shape}")  # (32, 10)

# 查看模型参数
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")