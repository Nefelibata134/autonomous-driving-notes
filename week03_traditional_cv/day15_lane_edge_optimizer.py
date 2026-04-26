import cv2
import numpy as np


class LaneEdgeOptimizer:
    """
    车道线边缘优化器
    输入：Canny 边缘图（有断裂和噪点）
    输出：干净、连续的车道线边缘
    """

    def __init__(self):
        # 细长核（适合车道线这种细长结构）
        self.vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 15))  # 高而窄，适合竖直线
        self.horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (15, 3))  # 宽而扁，适合横线
        self.general_kernel = np.ones((5, 5), np.uint8)

    def remove_noise(self, edge_img):
        """开运算去除小噪点"""
        return cv2.morphologyEx(edge_img, cv2.MORPH_OPEN,
                                self.general_kernel, iterations=1)

    def connect_gaps(self, edge_img):
        """闭运算连接断裂"""
        return cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE,
                                self.vertical_kernel, iterations=2)

    def enhance_lines(self, edge_img):
        """用细长核增强直线"""
        # 膨胀增强竖直线
        enhanced = cv2.dilate(edge_img, self.vertical_kernel, iterations=1)
        # 腐蚀去除横向干扰
        cleaned = cv2.erode(enhanced, self.horizontal_kernel, iterations=1)
        return cleaned

    def full_pipeline(self, edge_img):
        """完整优化流程"""
        # Step 1: 开运算去噪
        step1 = self.remove_noise(edge_img)
        # Step 2: 闭运算连接
        step2 = self.connect_gaps(step1)
        # Step 3: 增强直线
        step3 = self.enhance_lines(step2)
        return step3

    def visualize_steps(self, original_edge):
        """可视化每一步"""
        step1 = self.remove_noise(original_edge)
        step2 = self.connect_gaps(step1)
        step3 = self.enhance_lines(step2)

        # 拼接对比图
        h, w = original_edge.shape
        canvas = np.zeros((h, w * 4), dtype=np.uint8)
        canvas[:, :w] = original_edge
        canvas[:, w:2 * w] = step1
        canvas[:, 2 * w:3 * w] = step2
        canvas[:, 3 * w:] = step3

        # 添加标签
        color_canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        labels = ['Original', 'Open(去噪)', 'Close(连接)', 'Enhance(增强)']
        for i, label in enumerate(labels):
            x = i * w + 10
            cv2.putText(color_canvas, label, (x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return color_canvas


def test_optimizer():
    print("=== 车道线边缘优化器 ===\n")

    # 读取 Day 14 的 Canny 结果（或重新生成）
    img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ROI 掩码（复用 Day 14 逻辑）
    h, w = edges.shape
    mask = np.zeros_like(edges)
    vertices = np.array([[(0, h), (w, h), (int(w * 0.85), int(h * 0.6)),
                          (int(w * 0.15), int(h * 0.6))]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    # 优化
    optimizer = LaneEdgeOptimizer()
    optimized = optimizer.full_pipeline(roi_edges)
    comparison = optimizer.visualize_steps(roi_edges)

    cv2.imwrite('lane_edge_original.jpg', roi_edges)
    cv2.imwrite('lane_edge_optimized.jpg', optimized)
    cv2.imwrite('lane_edge_comparison.jpg', comparison)

    print("""已保存：
          - lane_edge_original.jpg: 原始
    Canny + ROI
    - lane_edge_optimized.jpg: 形态学优化后
                               - lane_edge_comparison.jpg: 四步对比图
    """)

    if __name__ == "__main__":
        test_optimizer()