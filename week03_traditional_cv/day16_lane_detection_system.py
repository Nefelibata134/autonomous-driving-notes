import cv2
import numpy as np


class LaneDetectionSystem:
    """
    完整车道线检测系统（传统 CV 方法）
    Pipeline: 灰度 → 高斯 → Canny → ROI → 霍夫 → 左右分离 → 绘制
    """

    def __init__(self, roi_vertices=None):
        self.roi_vertices = roi_vertices

    def create_roi(self, shape):
        h, w = shape[:2]
        if self.roi_vertices is None:
            # 梯形 ROI
            return np.array([[(0, h), (w, h), (int(w * 0.85), int(h * 0.6)),
                              (int(w * 0.15), int(h * 0.6))]], dtype=np.int32)
        return self.roi_vertices

    def detect(self, image):
        h, w = image.shape[:2]
        original = image.copy()

        # 1. 灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Canny 边缘
        edges = cv2.Canny(blurred, 50, 150)

        # 4. ROI 掩码
        roi_mask = np.zeros_like(edges)
        vertices = self.create_roi((h, w))
        cv2.fillPoly(roi_mask, vertices, 255)
        roi_edges = cv2.bitwise_and(edges, roi_mask)

        # 5. 概率霍夫直线
        lines = cv2.HoughLinesP(
            roi_edges, 1, np.pi / 180,
            threshold=50, minLineLength=40, maxLineGap=20
        )

        # 6. 左右车道线分离与拟合
        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 过滤水平线（车道线应该是斜的）
                if abs(y2 - y1) < 10:
                    continue

                # 计算斜率
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)

                # 左车道线：斜率负（从左上到右下，但在图像坐标中 y 向下）
                # 注意：图像坐标系 y 向下，所以左车道线 slope < 0
                if slope < -0.3:
                    left_lines.append((x1, y1, x2, y2, slope))
                elif slope > 0.3:
                    right_lines.append((x1, y1, x2, y2, slope))

        # 7. 绘制结果
        result = original.copy()

        # 画 ROI 区域（半透明）
        overlay = original.copy()
        cv2.fillPoly(overlay, vertices, (0, 255, 0))
        result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
        cv2.polylines(result, vertices, True, (0, 255, 255), 2)

        # 画所有检测到的线（细灰线）
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # 画左车道线（红色粗线）
        for x1, y1, x2, y2, s in left_lines:
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 画右车道线（蓝色粗线）
        for x1, y1, x2, y2, s in right_lines:
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # 拟合一条平均车道线（更稳定）
        if len(left_lines) > 0:
            left_avg = self.fit_line(left_lines, h)
            if left_avg:
                cv2.line(result, left_avg[0], left_avg[1], (0, 0, 255), 5)

        if len(right_lines) > 0:
            right_avg = self.fit_line(right_lines, h)
            if right_avg:
                cv2.line(result, right_avg[0], right_avg[1], (255, 0, 0), 5)

        return {
            'original': original,
            'edges': edges,
            'roi_edges': roi_edges,
            'result': result,
            'left_count': len(left_lines),
            'right_count': len(right_lines)
        }

    def fit_line(self, lines, img_height):
        """
        用所有短线段拟合一条长线（最小二乘法思想，简化版）
        """
        all_x = []
        all_y = []
        for x1, y1, x2, y2, s in lines:
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        if len(all_x) < 2:
            return None

        # 简单平均斜率和截距
        # 从底部 (img_height-1) 延伸到 roi 顶部 (img_height*0.6)
        y_bottom = img_height - 1
        y_top = int(img_height * 0.6)

        # 计算平均 x 对应 y_bottom 和 y_top
        # 这里简化：用所有点的线性回归（可用 np.polyfit）
        z = np.polyfit(all_y, all_x, 1)  # x = a*y + b
        f = np.poly1d(z)

        x_bottom = int(f(y_bottom))
        x_top = int(f(y_top))

        return ((x_bottom, y_bottom), (x_top, y_top))


def test_system():
    print("=== 车道线检测系统测试 ===\n")

    detector = LaneDetectionSystem()

    # 处理单张图
    img = cv2.imread('test_image.jpg')
    if img is None:
        print("请放置 test_image.jpg（最好是路面照片）")
        return

    results = detector.detect(img)

    cv2.imwrite('lane_system_original.jpg', results['original'])
    cv2.imwrite('lane_system_edges.jpg', results['edges'])
    cv2.imwrite('lane_system_roi_edges.jpg', results['roi_edges'])
    cv2.imwrite('lane_system_result.jpg', results['result'])

    print(f"检测到左车道线片段：{results['left_count']}")
    print(f"检测到右车道线片段：{results['right_count']}")
    print("\n已保存：")
    print("- lane_system_result.jpg: 最终结果（绿色ROI + 红/蓝车道线）")


if __name__ == "__main__":
    test_system()