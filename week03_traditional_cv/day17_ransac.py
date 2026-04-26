import cv2
import numpy as np

# 读取两张图（模拟不同视角拍摄的同一场景）
img1 = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
if img1 is None:
    print("请放置 test_image.jpg")
    exit()

# 模拟第二张图：旋转+缩放
(h, w) = img1.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 15, 0.9)  # 旋转15度，缩放0.9
img2 = cv2.warpAffine(img1, M, (w, h))

print("=== 特征匹配 ===")

# 1. 检测特征点
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print(f"图1: {len(kp1)} 点, 图2: {len(kp2)} 点")

# 2. 暴力匹配（Brute-Force Matcher）
print("\n1. BFMatcher（暴力匹配）")
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 汉明距离（适合 ORB）
matches = bf.match(des1, des2)

# 按距离排序
matches = sorted(matches, key=lambda x: x.distance)
print(f"匹配对数：{len(matches)}")
print(f"最近距离：{matches[0].distance}, 最远距离：{matches[-1].distance}")

# 绘制前 20 个匹配
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('matches_bf.jpg', match_img)

# 3. KNN 匹配 + 比率测试（更鲁棒）
print("\n2. KNN 匹配 + Lowe 比率测试")
bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING)
knn_matches = bf_knn.knnMatch(des1, des2, k=2)  # 每个点找 2 个最近邻

good_matches = []
for m, n in knn_matches:
    # Lowe 比率测试：最近距离 / 次近距离 < 0.75 才保留
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"KNN 匹配总数：{len(knn_matches)}, 筛选后：{len(good_matches)}")

match_knn_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:30], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('matches_knn_ratio.jpg', match_knn_img)

# 4. FLANN 匹配（快速近似，大数据集用）
print("\n3. FLANN 匹配（快速）")
# ORB 是二进制描述子，FLANN 需要特定参数
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
try:
    flann_matches = flann.knnMatch(des1, des2, k=2)
    flann_good = []
    for m, n in flann_matches:
        if m.distance < 0.7 * n.distance:
            flann_good.append(m)
    print(f"FLANN 筛选后：{len(flann_good)} 对")
except:
    print("FLANN 对 ORB 不稳定，建议 BFMatcher")

# 5. RANSAC 几何验证（剔除错误匹配）
print("\n4. RANSAC 单应性矩阵估计")
if len(good_matches) >= 4:
    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵（Homography），用 RANSAC 剔除外点
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # H: 3x3 变换矩阵，mask: 1=内点，0=外点

    inliers = mask.ravel().sum()
    print(f"RANSAC 内点：{inliers}/{len(good_matches)} ({100 * inliers / len(good_matches):.1f}%)")

    # 只绘制内点
    inlier_matches = [m for m, inlier in zip(good_matches, mask.ravel()) if inlier]
    ransac_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:30], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('matches_ransac.jpg', ransac_img)
else:
    print("匹配点太少，无法 RANSAC")