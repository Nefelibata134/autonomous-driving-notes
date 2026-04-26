import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== Autograd 自动求导 ===")

# 练习 1：标量求导
print("\n1. 标量求导")
x = torch.tensor(3.0, requires_grad=True, device=device)
y = x ** 2 + 2 * x + 1  # y = x² + 2x + 1

print(f"x = {x.item()}")
y.backward()  # 反向传播
print(f"dy/dx = {x.grad.item()}")  # 2*3 + 2 = 8

# 练习 2：向 量/矩阵求导（神经网络场景）
print("\n2. 矩阵求导（模拟一层神经网络）")
# 输入 (3, 1)
x = torch.randn(3, 1, device=device)
# 权重 (3, 3)，需要梯度！
w = torch.randn(3, 3, requires_grad=True, device=device)
# 偏置 (3, 1)
b = torch.randn(3, 1, requires_grad=True, device=device)

# 前向：y = w @ x + b
y = torch.matmul(w, x) + b
loss = y.sum()  # 简化损失

print(f"loss = {loss.item()}")
loss.backward()

print(f"w.grad 形状: {w.grad.shape}")  # 与w相同 (3,3)
print(f"b.grad 形状: {b.grad.shape}")  # 与b相同 (3,1)

# 练习 3：理解计算图
print("\n3. 计算图与链式法则")
a = torch.tensor(2.0, requires_grad=True, device=device)
b = torch.tensor(3.0, requires_grad=True, device=device)
c = a * b      # c = 6
d = c + 1      # d = 7
e = d ** 2     # e = 49

e.backward()
# e = (a*b + 1)²
# de/da = 2*(a*b+1)*b = 2*7*3 = 42
# de/db = 2*(a*b+1)*a = 2*7*2 = 28
print(f"de/da = {a.grad.item()} (应为42)")
print(f"de/db = {b.grad.item()} (应为28)")

# 练习 4：清零梯度（关键！）
print("\n4. 梯度清零")
w.grad.zero_()
b.grad.zero_()
print(f"清零后 w.grad: {w.grad.sum().item()}")

# 练习 5：禁用梯度追踪（推理时节省内存）
print("\n5. 禁用梯度")
with torch.no_grad():
    y_pred = torch.matmul(w, x) + b
    print(f"no_grad中 requires_grad: {y_pred.requires_grad}")  # False