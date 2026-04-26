import torch
import numpy as np

print("=== PyTorch Tensor 基础 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 设备对象（后续所有Tensor都发到这个设备）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 创建Tensor（对比NumPy）
print("\n1. 创建Tensor")
t1 = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
t2 = torch.randn(2, 3, device=device)          # 正态分布
t3 = torch.zeros(3, 3, device=device)
t4 = torch.arange(0, 10, 2, device=device)     # [0, 2, 4, 6, 8]

print(f"arange: {t4}, 设备: {t4.device}")

# 2. 属性
print("\n2. Tensor属性")
t = torch.randn(2, 3, 4, device=device)
print(f"形状: {t.shape}, 维度: {t.ndim}, 元素数: {t.numel()}")
print(f"数据类型: {t.dtype}, 设备: {t.device}")

# 3. 与NumPy互转（注意：CPU Tensor才能转NumPy）
print("\n3. NumPy互转")
np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
t_from_np = torch.from_numpy(np_arr).to(device)  # 先转Tensor，再发GPU
print(f"From NumPy: {t_from_np}")

# GPU Tensor转NumPy：必须先回CPU
t_cpu = t_from_np.cpu()
np_from_t = t_cpu.numpy()
print(f"To NumPy: {type(np_from_t)}")

# 4. 形状操作（与NumPy几乎一样，但支持GPU）
print("\n4. 形状操作")
t = torch.arange(12, device=device)
t_2d = t.reshape(3, 4)
t_t = t_2d.t()  # 转置
t_perm = t_2d.permute(1, 0)  # 同上
print(f"reshape(3,4): {t_2d.shape}")
print(f"transpose: {t_perm.shape}")

# 5. 索引与切片（同NumPy）
print("\n5. 索引切片")
print(f"第0行: {t_2d[0]}")
print(f"第1列: {t_2d[:, 1]}")

# 6. 广播（同NumPy，但自动在GPU上并行）
print("\n6. 广播运算")
a = torch.randn(3, 1, device=device)  # (3,1)
b = torch.randn(1, 4, device=device)  # (1,4)
c = a + b  # 广播为 (3,4)
print(f"广播结果: {c.shape}")