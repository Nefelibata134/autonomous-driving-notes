import cv2
import numpy as np
import matplotlib.pyplot as plt  # 用于显示图像（可选）

# 读取图像
img = cv2.imread('test_image.jpg')
if img is None:
    print("请放置 test_image.jpg")
    exit()

print("=== 颜色空间转换 ===")

# 练习 1：BGR ↔ RGB（最关键！）
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"RGB 转换后形状：{rgb.shape}")
print(f"原图 BGR[0,0]：{img[0,0]}")
print(f"转换后 RGB[0,0]：{rgb[0,0]}（B和R交换了）")

# 注意：如果直接保存 RGB 用 OpenCV，颜色会错乱！
cv2.imwrite('wrong_rgb.jpg', rgb)  # OpenCV 期望 BGR，所以保存后会偏色

# 正确做法：转回 BGR 再保存，或用其他库
bgr_again = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite('correct_bgr.jpg', bgr_again)

# 练习 2：BGR ↔ 灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"\n灰度图形状：{gray.shape}")  # (H, W)，无通道维度

# 灰度转回 3 通道（复制3份）
gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
print(f"灰度转3通道：{gray_3channel.shape}")  # (H, W, 3)，但看起来像灰度

# 练习 3：通道分离与合并（理解 BGR 结构）
print("\n=== 通道分离与合并 ===")
b, g, r = cv2.split(img)  # 分离为三个 2D 数组
print(f"B 通道形状：{b.shape}, G 通道形状：{g.shape}, R 通道形状：{r.shape}")

# 单独显示某通道（ NumPy 操作）
zeros = np.zeros_like(b)  # 创建同形状的零数组

# 只保留红色通道（R 在 BGR 的第 2 个位置）
red_only = cv2.merge([zeros, zeros, r])
cv2.imwrite('red_channel.jpg', red_only)

# 只保留蓝色通道
blue_only = cv2.merge([b, zeros, zeros])
cv2.imwrite('blue_channel.jpg', blue_only)

# 用 NumPy 方式合并（效果同 cv2.merge）
merged = np.stack([b, g, r], axis=2)  # 在通道维度堆叠
print(f"NumPy 合并形状：{merged.shape}")
print(f"与原图相同：{np.array_equal(img, merged)}")

# 练习 4：HSV 颜色空间（用于颜色筛选，如提取肤色、车道线）
print("\n=== HSV 颜色空间 ===")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
print(f"H 通道范围：[{h.min()}, {h.max()}]")  # 0-179（OpenCV 中 H 是 0-179）
print(f"S 通道范围：[{s.min()}, {s.max()}]")  # 0-255
print(f"V 通道范围：[{v.min()}, {v.max()}]")  # 0-255

# 提取蓝色物体（H 约 100-130）
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imwrite('blue_mask.jpg', blue_mask)  # 白色是蓝色区域

# 练习 5：图像算术运算（NumPy 广播 + OpenCV）
print("\n=== 图像算术运算 ===")

# 亮度调整（加法）
brighter = cv2.add(img, 50)  # 更安全的加法（防止溢出）
# 等价于：np.clip(img.astype(int) + 50, 0, 255).astype(np.uint8)

# 对比度调整（乘法）
contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)  # alpha=对比度, beta=亮度

# 最小修复：在文件开头添加
img_small = None  # 或者读取第二张图

# 混合两张图（需要尺寸相同）
if img_small is not None:
    blended = cv2.addWeighted(img, 0.7, img_small, 0.3, 0)
    cv2.imwrite('blended.jpg', blended)

# 练习 6：图像与 NumPy 掩码（高级）
print("\n=== 图像掩码操作 ===")
# 创建圆形掩码
mask = np.zeros(img.shape[:2], dtype=np.uint8)
center = (img.shape[1] // 2, img.shape[0] // 2)
radius = min(center[0], center[1]) // 2
cv2.circle(mask, center, radius, 255, -1)  # 白色圆

# 只保留圆形区域（其余变黑）
masked_img = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite('circular_mask.jpg', masked_img)