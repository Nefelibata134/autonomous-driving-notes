import cv2
import numpy as np

img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("请放置 test_image.jpg")
    exit()

print("=== Canny 边缘检测 ===")

# 练习 1：基础 Canny
print("\n1. 基础 Canny")
# cv2.Canny(image, threshold1, threshold2)
# threshold1: 低阈值（低于此值的不是边缘）
# threshold2: 高阈值（高于此值的肯定是边缘）
# 推荐比例：1:2 或 1:3

edges = cv2.Canny(img, 50, 150)
cv2.imwrite('canny_basic.jpg', edges)
print("Canny 输出是二值图：白色=边缘，黑色=背景")

# 练习 2：参数影响（关键！）
print("\n2. 双阈值参数影响（核心理解）")

# 低阈值过低 → 噪声被误判为边缘
edges_low = cv2.Canny(img, 10, 20)
cv2.imwrite('canny_threshold_low.jpg', edges_low)
print("阈值过低：大量噪点被当作边缘")

# 高阈值过高 → 真实边缘被漏掉
edges_high = cv2.Canny(img, 200, 300)
cv2.imwrite('canny_threshold_high.jpg', edges_high)
print("阈值过高：边缘断裂，细节丢失")

# 推荐参数
edges_good = cv2.Canny(img, 50, 150)
cv2.imwrite('canny_threshold_good.jpg', edges_good)
print("推荐阈值：低=50，高=150（1:3比例）")

# 练习 3：Canny 前置高斯核大小影响
print("\n3. 前置平滑的影响")
# Canny 内部会先进行高斯平滑，核大小影响去噪程度
# apertureSize: Sobel 核大小（默认 3）

for aperture in [3, 5, 7]:
    edges_ap = cv2.Canny(img, 50, 150, apertureSize=aperture)
    cv2.imwrite(f'canny_aperture_{aperture}.jpg', edges_ap)
    print(f"  apertureSize={aperture}: Sobel核越大，边缘越平滑，细节越少")

# 练习 4：Canny 步骤拆解（理解原理）
print("\n4. Canny 步骤拆解（手动模拟）")

# Step 1: 高斯平滑（Canny 内部已做，这里演示）
blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

# Step 2: 计算梯度（Sobel）
gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(gx**2 + gy**2)
direction = np.arctan2(gy, gx) * (180 / np.pi)

# Step 3: 非极大值抑制（NMS，Canny 核心）
# 简化版：在梯度方向上，只保留局部最大值
# 这里用 OpenCV 的 Canny 直接得到最终结果

# Step 4: 双阈值（手动模拟思想）
# 强边缘：magnitude > high_threshold
# 弱边缘：low_threshold < magnitude < high_threshold
# 抑制：magnitude < low_threshold

high = 150
low = 50
strong_edges = (magnitude > high).astype(np.uint8) * 255
weak_edges = ((magnitude > low) & (magnitude <= high)).astype(np.uint8) * 128

cv2.imwrite('canny_step_magnitude.jpg', np.clip(magnitude, 0, 255).astype(np.uint8))
cv2.imwrite('canny_step_strong.jpg', strong_edges)
cv2.imwrite('canny_step_weak.jpg', weak_edges)

print("""已保存 Canny 中间步骤：
- canny_step_magnitude.jpg: 梯度幅值
- canny_step_strong.jpg: 强边缘（>150）
- canny_step_weak.jpg: 弱边缘（50-150）""")

# 练习 5：彩色图 Canny（分别处理各通道）
print("\n5. 彩色图边缘检测")
color_img = cv2.imread('test_image.jpg')

# 方法 A：转灰度再 Canny（推荐，速度快）
edges_gray = cv2.Canny(cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY), 50, 150)

# 方法 B：各通道分别 Canny，再合并（保留更多颜色边缘信息）
edges_b = cv2.Canny(color_img[:,:,0], 50, 150)
edges_g = cv2.Canny(color_img[:,:,1], 50, 150)
edges_r = cv2.Canny(color_img[:,:,2], 50, 150)
edges_color = cv2.bitwise_or(cv2.bitwise_or(edges_b, edges_g), edges_r)

cv2.imwrite('canny_from_gray.jpg', edges_gray)
cv2.imwrite('canny_from_color.jpg', edges_color)
print("彩色通道合并通常比灰度检测到更多边缘，但也更多噪点")