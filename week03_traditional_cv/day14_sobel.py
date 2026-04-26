import cv2
import numpy as np

# 练习 1：手动 Sobel 卷积（理解原理）
print("=== 手动 Sobel 边缘检测 ===")

# Sobel 水平核（检测垂直边缘）
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Sobel 垂直核（检测水平边缘）
sobel_y = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
], dtype=np.float32)

# 创建测试图（白底黑方块，有明显边缘）
test_img01 = np.zeros((100, 100), dtype=np.float32)
test_img01[30:70, 30:70] = 255  # 中间白色方块

# 手动卷积
#from scipy.signal import convolve2d
# 如果没有 scipy，用 cv2.filter2D
gx_manual = cv2.filter2D(test_img01, -1, sobel_x)
gy_manual = cv2.filter2D(test_img01, -1, sobel_y)

# 梯度幅值（边缘强度）
gradient_magnitude = np.sqrt(gx_manual**2 + gy_manual**2)
# 梯度方向（边缘朝向）
gradient_direction = np.arctan2(gy_manual, gx_manual) * (180 / np.pi)

print(f"Gx（水平梯度）范围：[{gx_manual.min():.1f}, {gx_manual.max():.1f}]")
print(f"Gy（垂直梯度）范围：[{gy_manual.min():.1f}, {gy_manual.max():.1f}]")
print(f"梯度幅值范围：[{gradient_magnitude.min():.1f}, {gradient_magnitude.max():.1f}]")

# 保存可视化
cv2.imwrite('sobel_test_original.jpg', test_img01.astype(np.uint8))
cv2.imwrite('sobel_test_gx.jpg', np.abs(gx_manual).astype(np.uint8))
cv2.imwrite('sobel_test_gy.jpg', np.abs(gy_manual).astype(np.uint8))
cv2.imwrite('sobel_test_magnitude.jpg', gradient_magnitude.astype(np.uint8))

# 练习 2：OpenCV Sobel 函数（实际使用）
print("\n=== OpenCV Sobel 函数 ===")

img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("请放置 test_image.jpg")
    exit()

# Sobel 参数：cv2.Sobel(src, ddepth, dx, dy, ksize)
# ddepth: 输出深度，-1 表示和输入相同，但通常用 cv2.CV_64F 防止溢出
# dx, dy: 求导方向（dx=1,dy=0 检测垂直边缘；dx=0,dy=1 检测水平边缘）
# ksize: 核大小（1, 3, 5, 7）

# 检测垂直边缘（水平梯度）
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# 检测水平边缘（垂直梯度）
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 取绝对值并转回 uint8（因为梯度有正负）
sobelx_abs = cv2.convertScaleAbs(sobelx)
sobely_abs = cv2.convertScaleAbs(sobely)

# 合并梯度（近似幅值）
sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

cv2.imwrite('sobel_vertical_edges.jpg', sobelx_abs)   # 垂直边缘（横向线条）
cv2.imwrite('sobel_horizontal_edges.jpg', sobely_abs)  # 水平边缘（纵向线条）
cv2.imwrite('sobel_combined.jpg', sobel_combined)

print("""已保存：
- sobel_vertical_edges.jpg: 检测到的垂直边缘（如树干、建筑竖线）
- sobel_horizontal_edges.jpg: 检测到的水平边缘（如地平线、桌面）
- sobel_combined.jpg: 合并结果""")

# 练习 3：Scharr 算子（Sobel 的增强版，3x3 核的最优解）
print("\n=== Scharr 算子（更精确的边缘） ===")
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharr_combined = cv2.addWeighted(
    cv2.convertScaleAbs(scharrx), 0.5,
    cv2.convertScaleAbs(scharry), 0.5, 0
)
cv2.imwrite('scharr_combined.jpg', scharr_combined)
print("Scharr 对弱边缘更敏感，但噪声也更明显")

# 练习 4：不同核大小对比
print("\n=== 核大小对比 ===")
for ksize in [1, 3, 5, 7]:
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    cv2.imwrite(f'sobel_ksize_{ksize}.jpg', cv2.convertScaleAbs(sx))
    print(f"  ksize={ksize}: 核越大，边缘越粗，抗噪越强")