import cv2
import numpy as np


class ImageStitcher:
    """
    图像拼接器（Panorama）
    基于 ORB + BFMatcher + RANSAC + 透视变换
    """

    def __init__(self, detector='ORB'):
        if detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect_and_match(self, img1, img2):
        """检测特征点并匹配"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return None, None, None

        # KNN 匹配
        knn_matches = self.matcher.knnMatch(des1, des2, k=2)

        # Lowe 比率筛选
        good = []
        for m_n in knn_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        return kp1, kp2, good

    def estimate_homography(self, kp1, kp2, matches):
        """用 RANSAC 估计单应性矩阵"""
        if len(matches) < 4:
            return None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H, mask

    def stitch(self, img1, img2):
        """
        拼接两张图（将 img1 变换到 img2 的坐标系）
        """
        kp1, kp2, matches = self.detect_and_match(img1, img2)
        if matches is None or len(matches) < 4:
            print("匹配失败，无法拼接")
            return None

        H, mask = self.estimate_homography(kp1, kp2, matches)
        if H is None:
            print("单应性矩阵估计失败")
            return None

        # 计算输出画布大小
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 将 img1 的四个角变换到 img2 坐标系
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners1_transformed = cv2.perspectiveTransform(corners1, H)

        # 合并所有角点（img2 的角点 + 变换后的 img1 角点）
        all_corners = np.concatenate([
            corners1_transformed,
            np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        ])

        # 计算包围盒
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # 平移变换（让画布从 (0,0) 开始）
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        H_translated = translation @ H

        # 输出尺寸
        output_w = x_max - x_min
        output_h = y_max - y_min

        # 将 img1 变换到画布
        warped1 = cv2.warpPerspective(img1, H_translated, (output_w, output_h))

        # 将 img2 放到画布（简单平移）
        result = warped1.copy()
        result[-y_min:h2 - y_min, -x_min:w2 - x_min] = img2  # 直接覆盖（简化版）

        # 更好的做法：加权融合（Feathering）
        # 这里简化，实际应用需要处理重叠区域渐变

        return result


def test_stitcher():
    print("=== 图像拼接器测试 ===\n")

    # 读取两张图（需要是同一场景的不同视角）
    img1 = cv2.imread('test_image.jpg')
    if img1 is None:
        print("请放置 test_image.jpg")
        return

    # 模拟第二张图：裁剪右半部分并稍微旋转
    #h, w = img1.shape[:2]
    # 创建重叠图像（简单模拟：右移 20%）
    #M = np.float32([[1, 0, -int(w * 0.2)], [0, 1, 0]])
    #img2 = cv2.warpAffine(img1, M, (w, h))
    #img2 = img2[:, :int(w * 0.8)]  # 裁剪掉右边空白

    # 实际场景应该用两张真实照片，这里用变换模拟
    # 读取第二张真实图片（如果有）
    img2 = cv2.imread('test_image2.jpg')

    stitcher = ImageStitcher(detector='ORB')

    # 先显示匹配结果
    kp1, kp2, matches = stitcher.detect_and_match(img1, img2)
    if matches:
        match_vis = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('stitch_matches.jpg', match_vis)
        print(f"匹配点：{len(matches)}")

    # 拼接
    result = stitcher.stitch(img1, img2)
    if result is not None:
        cv2.imwrite('stitch_result.jpg', result)
        print("拼接完成，结果保存为 stitch_result.jpg")
    else:
        print("拼接失败（需要两张有明显重叠区域的图）")


if __name__ == "__main__":
    test_stitcher()