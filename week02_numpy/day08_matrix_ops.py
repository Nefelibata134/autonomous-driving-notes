import numpy as np

# 练习 1：矩阵乘法（@ 运算符）
print("=== 矩阵乘法 ===")
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(f"A @ B（矩阵乘）：\n{A @ B}")
# 手工验证：[1*5+2*7, 1*6+2*8] = [19, 22]

# 点乘（逐元素乘）：A * B
print(f"\nA * B（逐元素）：\n{A * B}")

# 练习 2：Batch 矩阵乘法（深度学习常用）
print("\n=== Batch 矩阵乘法 ===")
# 模拟 4 个样本，每个是 2x3 的矩阵
batch_A = np.random.rand(4, 2, 3)  # (B, M, K)
batch_B = np.random.rand(4, 3, 2)  # (B, K, N)

# Batch 乘法：4 个矩阵各自相乘，结果 (4, 2, 2)
batch_C = batch_A @ batch_B
print(f"Batch A 形状：{batch_A.shape}")
print(f"Batch B 形状：{batch_B.shape}")
print(f"Batch C 形状：{batch_C.shape}")

# 练习 3：Batch 图像处理（核心！）
print("\n=== Batch 图像处理 ===")

# 单张图：(H=64, W=64, C=3)
single_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
print(f"单张图形状：{single_image.shape}")

# Batch 图：(B=8, H=64, W=64, C=3) - 8张图一起处理
batch_images = np.random.randint(0, 256, (8, 64, 64, 3), dtype=np.uint8)
print(f"Batch 图形状：{batch_images.shape}")

# Batch 归一化（关键！）
mean = np.array([123.675, 116.28, 103.53])  # (3,)
std = np.array([58.395, 57.12, 57.375])      # (3,)

# 广播：(8, 64, 64, 3) - (3,) → 在 B,H,W 维度复制 mean
# 每张图的每个像素都进行 (pixel - mean) / std
batch_normalized = (batch_images.astype(np.float32) - mean) / std
print(f"Batch 归一化后形状：{batch_normalized.shape}")
print(f"数据范围：[{batch_normalized.min():.2f}, {batch_normalized.max():.2f}]")

# 练习 4：轴（axis）操作（Batch 统计）
print("\n=== 轴操作（理解 axis 是关键） ===")
batch = np.random.rand(4, 3, 2)  # 4个样本，每个3行2列

print(f"原数组形状：{batch.shape}")
print(f"axis=0 求和（对所有样本）：{batch.sum(axis=0).shape}")  # (3, 2)
print(f"axis=1 求和（对行）：{batch.sum(axis=1).shape}")        # (4, 2)
print(f"axis=2 求和（对列）：{batch.sum(axis=2).shape}")        # (4, 3)
print(f"全部求和：{batch.sum().shape}")  # 标量 ()

# 练习 5：keepdims 保持维度（广播时需要）
print("\n=== keepdims 保持维度 ===")
batch_mean = batch.mean(axis=0)           # 形状 (3, 2)
batch_mean_keep = batch.mean(axis=0, keepdims=True)  # 形状 (1, 3, 2)

print(f"无 keepdims：{batch_mean.shape}，减法时可能无法广播")
print(f"有 keepdims：{batch_mean_keep.shape}，可与原数组 (4,3,2) 广播")

# 实际应用：Batch 标准化（减去 batch 均值）
centered = batch - batch_mean_keep
print(f"中心化后形状：{centered.shape}")