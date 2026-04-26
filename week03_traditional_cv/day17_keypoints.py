import cv2
import numpy as np

img = cv2.imread('test_image.jpg')
if img is None:
    print("请放置 test_image.jpg")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=== 特征点检测 ===")

# 练习 1：SIFT（Scale-Invariant Feature Transform）
print("\n1. SIFT 检测")
# SIFT 专利已过期（2020年），OpenCV 4.4+ 可直接使用
sift = cv2.SIFT_create()

# 检测关键点和计算描述子
keypoints, descriptors = sift.detectAndCompute(gray, None)
print(f"SIFT 检测到 {len(keypoints)} 个特征点")
print(f"描述子形状：{descriptors.shape}")  # (N, 128) - 每个点 128 维向量

# 绘制关键点（带尺度圆圈和方向）
sift_img = cv2.drawKeypoints(
    img, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 显示大小和方向
)
cv2.imwrite('keypoints_sift.jpg', sift_img)

# 练习 2：ORB（Oriented FAST and Rotated BRIEF）
print("\n2. ORB 检测（免费、快速）")
orb = cv2.ORB_create(nfeatures=500)  # 最多检测 500 个

keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
print(f"ORB 检测到 {len(keypoints_orb)} 个特征点")
print(f"描述子形状：{descriptors_orb.shape}")  # (N, 32) - 每个点 32 维，二进制

orb_img = cv2.drawKeypoints(img, keypoints_orb, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('keypoints_orb.jpg', orb_img)

# 练习 3：对比不同算法
print("\n3. 算法对比")
algorithms = {
    'ORB': cv2.ORB_create(nfeatures=1000),
    'FAST': cv2.FastFeatureDetector_create(),  # 只检测，无描述子
    'AKAZE': cv2.AKAZE_create()  # 免费替代 SIFT，无需额外模块
}

# 尝试加入 SIFT（如果当前环境支持）
try:
    algorithms['SIFT'] = cv2.SIFT_create()
except Exception:
    pass

for name, detector in algorithms.items():
    try:
        kp, desc = detector.detectAndCompute(gray, None)
        if desc is not None:
            print(f"  {name}: {len(kp)} 点, 描述子 {desc.shape}, 类型 {desc.dtype}")
        else:
            print(f"  {name}: {len(kp)} 点, 无描述子")
    except cv2.error as e:
        print(f"  {name}: 检测失败（{e}）")

# 练习 4：理解描述子（Descriptor）
print("\n4. 描述子匹配原理")
# 描述子是特征点周围区域的"指纹"
# SIFT: 128 维浮点向量（梯度直方图）
# ORB: 32 维二进制向量（BRIEF）

# 计算两个描述子的欧氏距离（SIFT）
if len(keypoints) >= 2:
    d1 = descriptors[0]
    d2 = descriptors[1]
    distance = np.linalg.norm(d1 - d2)
    print(f"第1和第2个 SIFT 描述子距离：{distance:.2f}")
    print(f"  距离越小，越可能是同一个地方")

# 练习 5：多尺度检测（图像缩放后仍能检测到同一点）
print("\n5. 尺度不变性验证")
small_img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
small_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

kp_small, desc_small = sift.detectAndCompute(small_gray, None)
print(f"原图：{len(keypoints)} 点，缩小后：{len(kp_small)} 点")
print("SIFT/ORB 能在不同尺度下找到相似位置（尺度不变性）")