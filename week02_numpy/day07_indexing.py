import numpy as np

# 练习 1：一维数组索引（基础）
print("=== 一维索引 ===")
arr = np.arange(10)
print(f"数组：{arr}")
print(f"第1个元素：{arr[0]}")
print(f"最后一个：{arr[-1]}")
print(f"前3个：{arr[:3]}")      # 切片 [0:3]
print(f"第3-5个：{arr[2:5]}")  # 索引2到4（不含5）
print(f"每隔2个：{arr[::2]}")  # 步长为2
print(f"反转：{arr[::-1]}")    # 步长-1，反转数组

# 练习 2：二维数组索引（图像行列操作）
print("\n=== 二维索引（图像思维） ===")
# 创建 5x5 "图像"（模拟5x5像素的灰度图）
img_gray = np.arange(25).reshape(5, 5)
print(f"5x5 图像：\n{img_gray}")

# 取特定像素（行, 列）
print(f"第2行第3列的值：{img_gray[1, 2]}")  # 行索引1，列索引2 -> 7

# 取整行（第2行）
print(f"\n第2行：{img_gray[1, :]}")  # 或简写 img_gray[1]

# 取整列（第3列）
print(f"第3列：{img_gray[:, 2]}")

# 取区域（ROI - Region of Interest，图像处理核心！）
roi = img_gray[1:4, 1:4]  # 第2-4行，第2-4列（3x3区域）
print(f"\n中心3x3区域：\n{roi}")

# 练习 3：三维数组索引（彩色图像 HWC）
print("\n=== 三维索引（彩色图像） ===")
# 模拟 4x5 的彩色图 (H=4, W=5, C=3)
color_img = np.random.randint(0, 256, (4, 5, 3))
print(f"彩色图形状：{color_img.shape}")  # (4, 5, 3)

# 取特定像素的颜色值（BGR或RGB）
pixel = color_img[2, 3]  # 第3行第4列的像素
print(f"位置(2,3)的像素值：[R:{pixel[0]}, G:{pixel[1]}, B:{pixel[2]}]")

# 分离通道（图像处理常用！）
r_channel = color_img[:, :, 0]  # 所有行，所有列，第0通道（R）
g_channel = color_img[:, :, 1]  # G通道
b_channel = color_img[:, :, 2]  # B通道

print(f"\nR通道形状：{r_channel.shape}")  # (4, 5) - 变成二维

# 修改特定区域（比如在图像中间画个红方块）
color_img_copy = color_img.copy()
color_img_copy[1:3, 2:4] = [255, 0, 0]  # 红色（假设第0通道是R）
print(f"\n修改后的中心区域：\n{color_img_copy[1:3, 2:4]}")

# 练习 4：布尔索引（条件筛选）
print("\n=== 布尔索引（筛选像素） ===")
img = np.array([[10, 50, 100], [150, 200, 255], [30, 80, 120]])

# 找出所有大于100的像素（亮区）
bright_mask = img > 100
print(f"亮区掩码：\n{bright_mask}")

bright_pixels = img[bright_mask]
print(f"亮区像素值：{bright_pixels}")  # [150 200 255 120]

# 把所有暗区（<50）设为0（黑色）
img_dark_removed = img.copy()
img_dark_removed[img < 50] = 0
print(f"\n去除暗区后：\n{img_dark_removed}")

# 练习 5：花式索引（整数数组索引）
print("\n=== 花式索引 ===")
arr = np.arange(10)
indices = [1, 3, 5, 7]
print(f"取特定索引：{arr[indices]}")  # [1 3 5 7]

# 二维花式索引
img = np.arange(16).reshape(4, 4)
print(f"\n原图：\n{img}")
rows = [0, 2]
cols = [1, 3]
print(f"取(0,1)和(2,3)：{img[rows, cols]}")  # [1 11]