import cv2
import numpy as np
import os


# 练习 1：读取图像并检查属性
print("=== OpenCV 图像读取 ===")

# 读取图像（注意：OpenCV 默认读取为 BGR 格式！）
img = cv2.imread('test_image.jpg')  # 确保文件存在

if img is None:
    print("错误：无法读取图像！请确保 test_image.jpg 存在")
    exit()

print(f"类型：{type(img)}")          # <class 'numpy.ndarray'>
print(f"形状：{img.shape}")          # (H, W, 3) - 高度、宽度、通道
print(f"数据类型：{img.dtype}")       # uint8（0-255）
print(f"维度：{img.ndim}")            # 3（三维数组）

# 计算图像内存占用
h, w, c = img.shape
memory_kb = (h * w * c) / 1024
print(f"分辨率：{w}x{h}，通道数：{c}")
print(f"内存占用：{memory_kb:.2f} KB（无压缩）")

# 练习 2：访问像素值（NumPy 索引）
print("\n=== 像素值访问 ===")
# 左上角像素（蓝色通道排第一！）
pixel = img[0, 0]
print(f"左上角像素 BGR 值：{pixel}")  # [B, G, R]，如 [45, 67, 123]

# 访问单个通道
blue = img[0, 0, 0]   # B 通道
green = img[0, 0, 1]  # G 通道
red = img[0, 0, 2]    # R 通道
print(f"B={blue}, G={green}, R={red}")

# 修改像素（会变蓝！）
img_copy = img.copy()
img_copy[100:200, 100:200] = [255, 0, 0]  # BGR 蓝色
print("\n已修改中心区域为蓝色（BGR: [255, 0, 0]）")

# 练习 3：图像保存（验证 BGR）
print("\n=== 图像保存 ===")
cv2.imwrite('output_bgr.jpg', img)  # OpenCV 直接保存 BGR，无需转换

# 练习 4：读取灰度图（单通道）
print("\n=== 灰度图读取 ===")
gray = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
print(f"灰度图形状：{gray.shape}")    # (H, W) - 没有通道维度！
print(f"灰度图维度：{gray.ndim}")     # 2
print(f"第一个像素值：{gray[0, 0]}")  # 0-255 的整数

# 练习 5：读取不同尺寸（缩小以加速处理）
print("\n=== 读取时缩小尺寸 ===")
img_small = cv2.imread('test_image.jpg')
img_small = cv2.resize(img_small, (224, 224))  # 调整为 224x224
print(f"缩小后形状：{img_small.shape}")

# 练习 6：图像属性统计（结合 NumPy）
print("\n=== 图像统计 ===")
print(f"最大值：{img.max()}（应该是 255）")
print(f"最小值：{img.min()}（应该是 0）")
print(f"B 通道平均值：{img[:,:,0].mean():.2f}")
print(f"G 通道平均值：{img[:,:,1].mean():.2f}")
print(f"R 通道平均值：{img[:,:,2].mean():.2f}")

# 找出最亮像素的坐标
max_val = img.max()
brightest = np.where(img == max_val)
print(f"最亮像素位置（前3个）：")
for i in range(min(3, len(brightest[0]))):
    y, x = brightest[0][i], brightest[1][i]
    print(f"  ({y}, {x}): {img[y, x]}")