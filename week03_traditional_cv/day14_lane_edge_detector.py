import cv2
import numpy as np


class LaneEdgeDetector:
    """
    车道线边缘检测器（自动驾驶感知模块原型）
    输入：车载前视摄像头图像
    输出：路面区域内的边缘（用于后续霍夫变换检测直线）
    """

    def __init__(self, canny_low=50, canny_high=150,
                 roi_vertices=None, kernel_size=5):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.kernel_size = kernel_size

        # ROI 默认梯形（路面区域）
        # 四个点：左下、右下、右上、左上（顺时针）
        self.roi_vertices = roi_vertices

    def create_roi_mask(self, shape):
        """
        创建梯形 ROI 掩码
        只保留图像下方的梯形区域（路面）
        """
        h, w = shape[:2]

        if self.roi_vertices is None:
            # 默认梯形：底部全宽，顶部中间 60%
            # 适用于前视摄像头
            bottom_left = (0, h)
            bottom_right = (w, h)
            top_right = (int(w * 0.85), int(h * 0.6))
            top_left = (int(w * 0.15), int(h * 0.6))
            vertices = np.array([[bottom_left, bottom_right,
                                  top_right, top_left]], dtype=np.int32)
        else:
            vertices = self.roi_vertices

        # 创建黑色画布，填充白色多边形
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        return mask, vertices

    def detect(self, image_path):
        """
        完整检测流程
        """
        # 1. 读取
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")

        h, w = img.shape[:2]
        original = img.copy()

        # 2. 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. 高斯去噪（Canny 前置步骤，也可让 Canny 内部处理）
        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)

        # 4. Canny 边缘检测
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 5. 创建 ROI 掩码
        roi_mask, vertices = self.create_roi_mask((h, w))

        # 6. 应用掩码（只保留路面区域边缘）
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # 7. 可视化
        # 在原图上画 ROI 区域（半透明绿色）
        overlay = original.copy()
        cv2.fillPoly(overlay, vertices, (0, 255, 0))
        roi_visual = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

        # 画 ROI 边界线
        cv2.polylines(roi_visual, vertices, True, (0, 255, 255), 2)

        # 将边缘叠加到原图（红色边缘）
        edge_color = np.zeros_like(original)
        edge_color[masked_edges > 0] = [0, 0, 255]  # 红色边缘
        final_visual = cv2.addWeighted(roi_visual, 1.0, edge_color, 1.0, 0)

        return {
            'original': original,
            'gray': gray,
            'edges': edges,
            'roi_mask': roi_mask,
            'masked_edges': masked_edges,
            'visualization': final_visual,
            'roi_overlay': roi_visual
        }

    def auto_tune_thresholds(self, gray_image):
        """
        自动推荐 Canny 阈值（基于图像中值）
        公式：low = median * 0.33, high = median * 0.66
        """
        median = np.median(gray_image)
        low = int(max(0, 0.33 * median))
        high = int(min(255, 0.66 * median))
        return low, high

    def detect_auto(self, image_path):
        """使用自动阈值检测"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        low, high = self.auto_tune_thresholds(blurred)
        print(f"自动阈值：low={low}, high={high}")

        edges = cv2.Canny(blurred, low, high)

        # ROI 处理同上
        h, w = img.shape[:2]
        roi_mask, vertices = self.create_roi_mask((h, w))
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        return masked_edges


def test_lane_detector():
    """测试车道线检测器"""
    print("=== 车道线边缘检测器测试 ===\n")

    detector = LaneEdgeDetector(
        canny_low=50,
        canny_high=150,
        kernel_size=5
    )

    try:
        # 标准检测
        print("1. 标准参数检测")
        results = detector.detect('test_image.jpg')

        cv2.imwrite('lane_original.jpg', results['original'])
        cv2.imwrite('lane_gray.jpg', results['gray'])
        cv2.imwrite('lane_all_edges.jpg', results['edges'])
        cv2.imwrite('lane_roi_mask.jpg', results['roi_mask'])
        cv2.imwrite('lane_masked_edges.jpg', results['masked_edges'])
        cv2.imwrite('lane_roi_overlay.jpg', results['roi_overlay'])
        cv2.imwrite('lane_final_visual.jpg', results['visualization'])

        print("""已保存结果：
              - lane_gray.jpg: 灰度图
                               - lane_all_edges.jpg: 全图边缘
                                                     - lane_masked_edges.jpg: ROI
        内边缘（关键输出）
        - lane_final_visual.jpg: 最终可视化（绿色
        ROI + 红色边缘）""")

        # 自动阈值检测
        print("\n2. 自动阈值检测")
        auto_edges = detector.detect_auto('test_image.jpg')
        cv2.imwrite('lane_auto_edges.jpg', auto_edges)
        print("自动阈值结果已保存")

    except Exception as e:
        print(f"错误：{e}")
        print("请确保 test_image.jpg 存在，且包含路面场景")


if __name__ == "__main__":
    test_lane_detector()