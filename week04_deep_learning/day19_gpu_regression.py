import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 生成数据：y = 3x + 2 + 噪声
torch.manual_seed(42)
x = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)  # (100, 1)
true_w = 3.0
true_b = 2.0
y = true_w * x + true_b + 0.1 * torch.randn(100, 1, device=device)

# 初始化参数（需要梯度，放在GPU）
w = torch.randn(1, requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)

print(f"初始: w={w.item():.3f}, b={b.item():.3f}")

# 超参数
lr = 0.1
epochs = 100

# 训练循环
for epoch in range(epochs):
    # 前向传播
    y_pred = x * w + b

    # 计算损失（MSE）
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播（自动计算梯度）
    loss.backward()

    # 手动更新参数（后续用 optimizer 替代）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        # 清零梯度（必须！）
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss={loss.item():.6f}, w={w.item():.3f}, b={b.item():.3f}")

print(f"\n最终: w={w.item():.3f} (目标{true_w}), b={b.item():.3f} (目标{true_b})")

# 扩展：对比 CPU vs GPU 速度（大数据）
print("\n=== CPU vs GPU 速度对比 ===")
size = 5000

# CPU
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start

# GPU
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)
torch.cuda.synchronize()  # 等待GPU就绪
start = time.time()
c_gpu = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()  # 等待计算完成
gpu_time = time.time() - start

print(f"CPU: {cpu_time:.3f}s")
print(f"GPU: {gpu_time:.3f}s")
print(f"加速比: {cpu_time / gpu_time:.1f}x")