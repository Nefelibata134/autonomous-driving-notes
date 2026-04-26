import cv2
import numpy as np

img = cv2.imread('test_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

print("=== 霍夫变换 ===")

# 1. 标准霍夫直线（HoughLines）
print("\n1. 标准霍夫直线（不推荐，输出极坐标）")
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)
# rho: 距离精度（像素）
# theta: 角度精度（弧度）
# threshold: 交点累加数阈值（越高越严格）

hough_img = img.copy()
if lines is not None:
    for line in lines[:10]:  # 只看前10条
        rho, theta = line[0]
        # 极坐标转直角坐标
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('hough_standard.jpg', hough_img)

# 2. 概率霍夫直线（HoughLinesP - 实际使用！）
print("\n2. 概率霍夫直线 HoughLinesP（推荐）")
# 输出的是线段端点，不是极坐标
lines_p = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi/180,
    threshold=50,           # 阈值降低，检测更多线
    minLineLength=50,       # 最小线段长度（过滤短噪线）
    maxLineGap=10           # 最大间隙（断裂处连接）
)

houghp_img = img.copy()
if lines_p is not None:
    print(f"检测到 {len(lines_p)} 条线段")
    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        cv2.line(houghp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
else:
    print("未检测到线段")

cv2.imwrite('hough_probabilistic.jpg', houghp_img)

# 3. 霍夫圆检测（交通标志、车轮）
print("\n3. 霍夫圆检测 HoughCircles")
# 需要先做中值模糊去噪
blurred = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,               # 分辨率反比
    minDist=50,         # 圆心最小间距
    param1=100,         # Canny 高阈值
    param2=30,          # 累加器阈值（越小检测越多假圆）
    minRadius=10,
    maxRadius=100
)

circle_img = img.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"检测到 {len(circles[0])} 个圆")
    for i in circles[0, :]:
        cv2.circle(circle_img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 外圆
        cv2.circle(circle_img, (i[0], i[1]), 2, (0, 255, 255), 3)   # 圆心
else:
    print("未检测到圆")

cv2.imwrite('hough_circles.jpg', circle_img)