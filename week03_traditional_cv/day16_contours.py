import cv2
import numpy as np

img = cv2.imread('test_image.jpg')
if img is None:
    print("请放置 test_image.jpg")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=== 轮廓检测 ===")

# 1. 二值化/边缘准备
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 或者用 Canny 边缘（更常用）
edges = cv2.Canny(gray, 50, 150)

# 2. 查找轮廓
print("\n1. findContours 基础")
# cv2.findContours(image, mode, method)
# mode: RETR_EXTERNAL（只外层）, RETR_LIST（所有）, RETR_TREE（层级）
# method: CHAIN_APPROX_SIMPLE（压缩点）, CHAIN_APPROX_NONE（保留所有）
contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(f"找到 {len(contours)} 个轮廓")

# 3. 绘制轮廓
print("\n2. 绘制轮廓")
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # -1=画所有，2=线宽
cv2.imwrite('contours_all.jpg', contour_img)

# 4. 轮廓面积与周长（筛选大物体）
print("\n3. 轮廓筛选（面积过滤）")
filtered_contours = []
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 100:  # 过滤小噪点
        filtered_contours.append(cnt)
        print(f"  轮廓 {i}: 面积={area:.1f}, 周长={cv2.arcLength(cnt, True):.1f}")

# 绘制筛选后的
filtered_img = img.copy()
cv2.drawContours(filtered_img, filtered_contours, -1, (0, 0, 255), 2)
cv2.imwrite('contours_filtered.jpg', filtered_img)

# 5. 外接矩形与最小外接圆
print("\n4. 外接形状")
bbox_img = img.copy()
for cnt in filtered_contours[:10]:  # 只看前10个
    # 外接矩形
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 最小外接圆
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(bbox_img, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)

cv2.imwrite('contours_bbox.jpg', bbox_img)

# 6. 轮廓近似（多边形拟合）
print("\n5. 多边形近似")
approx_img = img.copy()
for cnt in filtered_contours[:5]:
    epsilon = 0.02 * cv2.arcLength(cnt, True)  # 精度参数
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(approx_img, [approx], -1, (255, 255, 0), 3)
    print(f"  原轮廓 {len(cnt)} 点 → 近似后 {len(approx)} 点")

cv2.imwrite('contours_approx.jpg', approx_img)