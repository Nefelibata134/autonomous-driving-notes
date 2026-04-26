import cv2
import numpy as np

img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("请放置 test_image.jpg")
    exit()

# 先二值化（形态学操作通常对二值图进行）
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

print("=== 形态学操作 ===")

# 1. 腐蚀（Erode）- 缩小白色区域
print("\n1. 腐蚀（去噪点）")
kernel = np.ones((5, 5), np.uint8)  # 5x5 全1核
eroded = cv2.erode(binary, kernel, iterations=1)
cv2.imwrite('morph_erode.jpg', eroded)
print("腐蚀：白色区域缩小，小噪点消失")

# 2. 膨胀（Dilate）- 扩大白色区域
print("\n2. 膨胀（连接断裂）")
dilated = cv2.dilate(binary, kernel, iterations=1)
cv2.imwrite('morph_dilate.jpg', dilated)
print("膨胀：白色区域扩大，断裂处连接")

# 3. 核大小影响
print("\n3. 核大小对比")
for k in [3, 5, 7]:
    kernel_k = np.ones((k, k), np.uint8)
    eroded_k = cv2.erode(binary, kernel_k, iterations=1)
    dilated_k = cv2.dilate(binary, kernel_k, iterations=1)
    cv2.imwrite(f'morph_erode_k{k}.jpg', eroded_k)
    cv2.imwrite(f'morph_dilate_k{k}.jpg', dilated_k)
    print(f"  核{k}x{k}：腐蚀/膨胀效果更强")

# 4. 迭代次数影响
print("\n4. 迭代次数")
eroded_3iter = cv2.erode(binary, kernel, iterations=3)
cv2.imwrite('morph_erode_3iter.jpg', eroded_3iter)
print("迭代3次 = 连续腐蚀3次，效果更强")

# 5. 开运算（Opening）- 去噪点
print("\n5. 开运算（Opening = 腐蚀 + 膨胀）")
# 添加椒盐噪声测试
noisy = binary.copy()
salt = np.random.rand(*binary.shape) < 0.02
pepper = np.random.rand(*binary.shape) < 0.02
noisy[salt] = 255
noisy[pepper] = 0

opening = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
cv2.imwrite('morph_noisy.jpg', noisy)
cv2.imwrite('morph_opening.jpg', opening)
print("开运算完美去除椒盐噪点，同时保留原图形")

# 6. 闭运算（Closing）- 填空洞、连断裂
print("\n6. 闭运算（Closing = 膨胀 + 腐蚀）")
# 模拟断裂边缘
broken = binary.copy()
broken[50:55, 100:200] = 0  # 人为制造断裂

closing = cv2.morphologyEx(broken, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('morph_broken.jpg', broken)
cv2.imwrite('morph_closing.jpg', closing)
print("闭运算连接断裂处，填充小空洞")

# 7. 梯度（Gradient）- 提取轮廓
print("\n7. 形态学梯度（轮廓提取）")
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite('morph_gradient.jpg', gradient)
print("梯度 = 膨胀 - 腐蚀，得到物体轮廓")

# 8. 顶帽与黑帽（高级）
print("\n8. 顶帽（Top Hat）- 提取亮细节")
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imwrite('morph_tophat.jpg', tophat)
print("顶帽 = 原图 - 开运算，提取比周围亮的细小区域")
