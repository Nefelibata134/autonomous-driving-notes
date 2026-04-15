import numpy as np

# 练习 1：创建数组（对比 Python 列表）
print("=== 创建数组 ===")

# 从列表创建
list_data = [1, 2, 3, 4, 5]
arr1 = np.array(list_data)
print(f"一维数组：{arr1}, 类型：{type(arr1)}")
print(f"数据类型：{arr1.dtype}")  # int64 或 int32

# 创建二维数组（矩阵）
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n二维数组：\n{arr2d}")
print(f"形状（shape）：{arr2d.shape}")  # (3, 3) - 3行3列

# 练习 2：特殊数组创建（图像处理常用）
print("\n=== 特殊数组创建 ===")

# zeros - 创建黑图（全0）
black_img = np.zeros((3, 3))  # 3x3 全0
print(f"zeros:\n{black_img}")

# ones - 创建白图（全1）
white_img = np.ones((2, 4))
print(f"\nones:\n{white_img}")

# full - 填充特定值（如全255表示白色）
solid_img = np.full((2, 2), 255)
print(f"\nfull 255:\n{solid_img}")

# eye - 单位矩阵（对角线为1）
identity = np.eye(3)
print(f"\n单位矩阵:\n{identity}")

# 练习 3：序列创建（arange 和 linspace）
print("\n=== 序列创建 ===")

# arange - 类似 range，但返回数组
arr_seq = np.arange(0, 10, 2)  # 0到10，步长2
print(f"arange: {arr_seq}")  # [0 2 4 6 8]

# linspace - 等间隔分割（起点，终点，数量）
arr_lin = np.linspace(0, 1, 5)  # 0到1分成5份
print(f"linspace: {arr_lin}")  # [0.   0.25 0.5  0.75 1.  ]

# 练习 4：随机数组（图像噪声、数据增强用）
print("\n=== 随机数组 ===")

# random.rand - 0-1均匀分布
rand_arr = np.random.rand(3, 3)
print(f"随机0-1:\n{rand_arr}")

# random.randint - 整数随机（如生成随机像素）
rand_int = np.random.randint(0, 256, (2, 2))  # 0-255，2x2（像灰度图）
print(f"\n随机像素(0-255):\n{rand_int}")

# random.randn - 标准正态分布（深度学习权重初始化用）
rand_norm = np.random.randn(2, 3)
print(f"\n标准正态分布:\n{rand_norm}")

# 练习 5：数组属性（关键！）
print("\n=== 数组属性 ===")
img = np.random.randint(0, 256, (100, 200, 3))  # 模拟 100x200 的彩色图

print(f"形状 (shape): {img.shape}")      # (100, 200, 3) -> (高, 宽, 通道)
print(f"维度数 (ndim): {img.ndim}")      # 3 - 三维数组
print(f"总元素数 (size): {img.size}")    # 100*200*3 = 60000
print(f"数据类型 (dtype): {img.dtype}")  # int64
print(f"每个元素大小 (itemsize): {img.itemsize} 字节")  # 8 字节(int64)
print(f"总内存: {img.nbytes / 1024:.2f} KB")  # 约 468.75 KB