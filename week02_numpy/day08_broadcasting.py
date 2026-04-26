import numpy as np

# 练习 1：标量广播（最简单）
print("=== 标量广播 ===")
arr = np.array([1, 2, 3])
print(f"原数组：{arr}")
print(f"加 10：{arr + 10}")      # [11 12 13]，10 被广播到 [10 10 10]
print(f"乘 2：{arr * 2}")       # [2 4 6]

# 练习 2：一维与二维广播（图像减均值场景）
print("\n=== 一维 vs 二维广播 ===")
# 模拟图像：(3, 4) 的灰度图
image = np.array([[10, 20, 30, 40],
                  [50, 60, 70, 80],
                  [90, 100, 110, 120]])
print(f"图像形状：{image.shape}")

# 每行减去该行的平均值（行方向广播）
row_means = image.mean(axis=1)  # 每行平均值，形状 (3,)
print(f"行平均值：{row_means}, 形状：{row_means.shape}")

# 关键：row_means 是 (3,)，image 是 (3,4)
# 广播规则：从后向前对齐，缺失的维度视为 1
# (3, 4) - (3,) → (3, 4) - (3, 1) → 自动扩展为 (3, 4)
centered = image - row_means.reshape(-1, 1)  # reshape 显式表明意图
print(f"行中心化后：\n{centered}")
print(f"每行和（应为0）：{centered.sum(axis=1)}")

# 练习 3：三维数组广播（彩色图像归一化 - 核心！）
print("\n=== 三维广播（图像归一化） ===")
# 模拟 2x2 的 RGB 图像 (H=2, W=2, C=3)
rgb_image = np.array([[[255, 0, 0],   [0, 255, 0]],
                      [[0, 0, 255],   [255, 255, 255]]])
print(f"RGB 图像形状：{rgb_image.shape}")  # (2, 2, 3)

# ImageNet 均值 [R, G, B]
mean = np.array([123.675, 116.28, 103.53])  # 形状 (3,)
std = np.array([58.395, 57.12, 57.375])      # 形状 (3,)

# 广播魔法：(2, 2, 3) - (3,) → 自动在 H,W 维度复制 mean
# 实际运算：每个像素的 [R,G,B] 都减去 [123.675, 116.28, 103.53]
normalized = (rgb_image.astype(np.float32) - mean) / std
print(f"归一化后形状：{normalized.shape}")
print(f"第一个像素归一化值：{normalized[0, 0]}")

# 练习 4：广播规则验证（失败案例）
print("\n=== 广播规则（从后向前对齐） ===")
# 规则：维度大小要么相等，要么其中一个为 1（或缺失）

a = np.ones((3, 4))      # 3行4列
b = np.ones((4,))        # 4个元素
c = np.ones((3,))        # 3个元素

print(f"(3,4) + (4,)：成功 → {(a + b).shape}")  # (3,4)，4匹配，3扩展
# print(f"(3,4) + (3,)：失败 → ")  # 会报错！因为 4 != 3 且 3 != 1

# 解决：reshape 显式添加维度
b_fixed = c.reshape(-1, 1)  # (3,) → (3, 1)
print(f"(3,4) + (3,1)：成功 → {(a + b_fixed).shape}")  # (3,4)

# 练习 5：np.newaxis 增加维度（常用技巧）
print("\n=== np.newaxis 增加维度 ===")
arr = np.array([1, 2, 3])  # 形状 (3,)

# 变成列向量 (3, 1)
col = arr[:, np.newaxis]
print(f"列向量形状：{col.shape}")
# 变成行向量 (1, 3)
row = arr[np.newaxis, :]
print(f"行向量形状：{row.shape}")

# 外积（outer product）：(3,1) * (1,3) → (3,3)
outer = col * row
print(f"外积结果：\n{outer}")