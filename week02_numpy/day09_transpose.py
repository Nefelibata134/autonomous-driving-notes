import numpy as np

# 练习 1：transpose 基础（维度重排）
print("=== transpose 基础 ===")
arr = np.arange(24).reshape(2, 3, 4)  # 2个3x4的"图"
print(f"原形状：{arr.shape}")  # (2, 3, 4)
print(f"原数组[0]：\n{arr[0]}")

# 完全转置：(2,3,4) → (4,3,2)
transposed = arr.transpose(2, 1, 0)
print(f"\ntranspose(2,1,0) 后：{transposed.shape}")  # (4, 3, 2)

# 部分转置：交换最后两个维度（类似图像 HW → WH）
swapped = arr.transpose(0, 2, 1)  # (2, 3, 4) → (2, 4, 3)
print(f"transpose(0,2,1) 后：{swapped.shape}")  # (2, 4, 3)

# 练习 2：图像维度转换（核心！）
print("\n=== 图像维度转换（HWC ↔ CHW） ===")

# 模拟一张 2x3 的 RGB 图 (H=2, W=3, C=3)
img_hwc = np.arange(18).reshape(2, 3, 3)
print(f"HWC 格式：{img_hwc.shape}")
print(f"像素(0,0)的RGB：{img_hwc[0, 0]}")  # [0, 1, 2]

# 转为 CHW（PyTorch 需要）
img_chw = img_hwc.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
print(f"\nCHW 格式：{img_chw.shape}")
print(f"R通道（第0通道）：\n{img_chw[0]}")  # 所有像素的 R 值

# 练习 3：内存连续性（性能关键！）
print("\n=== 内存连续性（C_CONTIGUOUS） ===")

arr = np.arange(12).reshape(3, 4)
print(f"原数组 C_CONTIGUOUS：{arr.flags['C_CONTIGUOUS']}")  # True（行优先，连续）

# transpose 后通常不连续
transposed = arr.T  # 或 arr.transpose(1, 0)
print(f"transpose 后 C_CONTIGUOUS：{transposed.flags['C_CONTIGUOUS']}")  # False！

# 强制连续（深度学习需要，否则 PyTorch 会警告）
contiguous = np.ascontiguousarray(transposed)
print(f"ascontiguousarray 后：{contiguous.flags['C_CONTIGUOUS']}")  # True

# 性能对比（大数组）
print("\n=== 性能对比 ===")
big_arr = np.random.rand(1000, 1000)

# 连续内存求和
import time
start = time.time()
for _ in range(100):
    big_arr.sum()
t1 = time.time() - start

# 非连续（转置后）求和
big_transposed = big_arr.T
start = time.time()
for _ in range(100):
    big_transposed.sum()
t2 = time.time() - start

print(f"连续内存耗时：{t1:.4f}s")
print(f"非连续内存耗时：{t2:.4f}s")
print(f"性能差距：{t2/t1:.2f}x")

# 练习 5：permute 与 swapaxes（更灵活的维度交换）
print("\n=== swapaxes（交换两个轴） ===")
arr = np.arange(24).reshape(2, 3, 4)

# 交换轴 0 和 2：(2,3,4) ↔ (4,3,2)
swapped = np.swapaxes(arr, 0, 2)
print(f"swapaxes(0,2)：{swapped.shape}")

# 等价于 transpose
transposed = arr.transpose(2, 1, 0)
print(f"transpose(2,1,0)：{transposed.shape}")
print(f"结果相同：{np.array_equal(swapped, transposed)}")