import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== 损失函数与优化器 ===")

# 练习 1：损失函数
print("\n1. 损失函数")

# 回归任务：MSE（均方误差）
mse_loss = nn.MSELoss()
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 2.5])
loss = mse_loss(pred, target)
print(f"MSE: {loss.item():.4f}")  # (0.5² + 0.5² + 0.5²)/3 = 0.25

# 分类任务：CrossEntropyLoss（多分类）
# 注意：输入是 logits（未归一化），内部会自动 softmax
ce_loss = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 1.0, 0.1]])  # 模型输出（3类）
target = torch.tensor([0])  # 真实标签是第0类
loss = ce_loss(logits, target)
print(f"CrossEntropy: {loss.item():.4f}")

# 二分类：BCEWithLogitsLoss（更稳定）
bce_loss = nn.BCEWithLogitsLoss()
pred = torch.tensor([0.5, -0.5, 2.0])  # logits
target = torch.tensor([1.0, 0.0, 1.0])  # 0或1
loss = bce_loss(pred, target)
print(f"BCE: {loss.item():.4f}")

# 练习 2：优化器
print("\n2. 优化器")

# 创建一个简单模型
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
).to(device)

# SGD（随机梯度下降）：最基础
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print("SGD: 基础，需调学习率")

# Adam（自适应矩估计）：最常用，默认首选
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
print("Adam: 自适应学习率，收敛快")

# AdamW（Adam + 权重衰减）：Transformer 时代标配
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print("AdamW: 更好的正则化")

# 练习 3：训练步骤（一个 batch 的完整流程）
print("\n3. 一个 batch 的训练步骤")

# 假数据
batch_size = 16
x = torch.randn(batch_size, 10).to(device)
y = torch.randint(0, 2, (batch_size,)).float().to(device)  # 二分类标签

# 1. 前向传播
model.train()  # 训练模式（启用 Dropout/BatchNorm）
pred = model(x).squeeze()  # (16,) - 二分类输出

# 2. 计算损失
criterion = nn.BCEWithLogitsLoss()
loss = criterion(pred, y)

# 3. 清零旧梯度（必须！）
optimizer_adam.zero_grad()

# 4. 反向传播（计算梯度）
loss.backward()

# 5. 更新参数（走一步）
optimizer_adam.step()

print(f"损失: {loss.item():.4f}")
print("梯度更新完成！")

# 练习 4：学习率调度器（进阶）
print("\n4. 学习率调度")
scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=10, gamma=0.1)
# 每 10 个 epoch，学习率乘以 0.1
print("StepLR: 阶梯式衰减学习率")