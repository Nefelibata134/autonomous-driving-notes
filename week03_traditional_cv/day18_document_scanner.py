import cv2
import numpy as np


class DocumentScanner:
    """
    智能文档扫描仪
    Pipeline: 读取 → 预处理 → Canny → 轮廓筛选 → 透视变换
    """

    def __init__(self):
        pass

    def preprocess(self, image):
        """预处理：缩小（加速）→ 灰度 → 高斯模糊"""
        max_dim = 1000
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
        else:
            resized = image.copy()
            scale = 1.0

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        return resized, gray, blurred, scale

    def detect_edges(self, blurred_img):
        """Canny 边缘检测 + 形态学闭运算（连接断裂边缘）"""
        edges = cv2.Canny(blurred_img, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        return edges, closed

    def find_document_contour(self, closed_edges):
        """从边缘图中找到最大的四边形轮廓（文档）"""
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # 放宽到 4~6 个点（圆角文档会有多点）
            if 4 <= len(approx) <= 6:
                return approx

        # 兜底：用最小外接矩形（一定能返回4个点）
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        return box.reshape(4, 1, 2).astype(np.int32)

    def order_points(self, pts):
        """将四个角点按顺序排列：左上、右上、右下、左下"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下

        return rect

    def perspective_transform(self, image, contour):
        """透视变换：将四边形区域拉正为矩形"""
        if contour is None:
            print("轮廓为 None，跳过透视变换")
            return None

        # 强制 reshape 为 Nx2
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 4:
            print(f"角点不足4个：{len(pts)}")
            return None

        # 如果超过4个点，取最小外接四边形
        if len(pts) > 4:
            rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
            box = cv2.boxPoints(rect)
            pts = box.astype(np.float32)

        rect = self.order_points(pts)

        (tl, tr, br, bl) = rect

        # 计算输出尺寸（取上下边、左右边的最大长度）
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # 目标角点：正矩形
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # 计算并执行透视变换
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped

    def full_pipeline(self, image_path):
        """
        完整扫描流程：读取 → 预处理 → 边缘 → 轮廓 → 透视变换
        返回拉正后的文档图像
        """
        # 1. 读取
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像：{image_path}")
            return None

        # 2. 预处理（缩小加速）
        resized, gray, blurred, scale = self.preprocess(img)

        # 3. 边缘检测
        edges, closed = self.detect_edges(blurred)

        # 4. 找文档轮廓
        contour = self.find_document_contour(closed)

        # 5. 透视变换（优先用原图做，分辨率更高）
        if contour is not None:
            if scale != 1.0:
                # 轮廓坐标还原到原图尺寸
                contour = (contour / scale).astype(np.int32)
                warped = self.perspective_transform(img, contour)
            else:
                warped = self.perspective_transform(resized, contour)
        else:
            print("未检测到四边形文档轮廓")
            return None

        return warped


def test_detection():
    """测试：检测文档轮廓并可视化中间步骤"""
    print("=== 文档边缘检测测试 ===\n")

    scanner = DocumentScanner()
    img = cv2.imread('test_image3.jpg')

    if img is None:
        print("请放置 test_image3.jpg（包含倾斜的纸张）")
        return

    resized, gray, blurred, scale = scanner.preprocess(img)
    edges, closed = scanner.detect_edges(blurred)
    contour = scanner.find_document_contour(closed)

    # 可视化
    vis = resized.copy()

    # 画所有轮廓（绿色）
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # 画检测到的文档轮廓（红色粗线）和角点（黄色）
    if contour is not None:
        cv2.polylines(vis, [contour], True, (0, 0, 255), 3)
        for point in contour:
            cv2.circle(vis, tuple(point[0]), 8, (0, 255, 255), -1)
        print(f"检测到文档轮廓：{len(contour)} 个角点")
    else:
        print("未检测到文档轮廓")

    # 保存中间步骤
    cv2.imwrite('scan_step1_original.jpg', resized)
    cv2.imwrite('scan_step2_edges.jpg', edges)
    cv2.imwrite('scan_step3_closed.jpg', closed)
    cv2.imwrite('scan_step4_contour.jpg', vis)


def test_transform():
    """测试：透视变换拉正文档"""
    print("\n=== 透视变换测试 ===")

    scanner = DocumentScanner()
    img = cv2.imread('test_image3.jpg')

    if img is None:
        print("请放置 test_image3.jpg")
        return

    resized, gray, blurred, scale = scanner.preprocess(img)
    edges, closed = scanner.detect_edges(blurred)
    contour = scanner.find_document_contour(closed)

    print(f"检测到的轮廓角点数：{len(contour) if contour is not None else 'None'}")

    if contour is not None:
        warped = scanner.perspective_transform(resized, contour)
        if warped is not None:
            cv2.imwrite('scan_step5_warped.jpg', warped)
            print(f"拉正后尺寸：{warped.shape}")
        else:
            print("perspective_transform 返回 None")
    else:
        print("没找到文档轮廓")


def test_full_pipeline():
    """测试：完整一键扫描流程"""
    print("\n=== 完整流程测试 ===")

    scanner = DocumentScanner()
    result = scanner.full_pipeline('test_image3.jpg')

    if result is not None:
        cv2.imwrite('scan_step6_final.jpg', result)
        print(f"扫描完成，保存为 scan_step6_final.jpg，尺寸：{result.shape}")
    else:
        print("扫描失败（请确保图片包含倾斜的纸张/文档）")


if __name__ == "__main__":
    test_detection()
    test_transform()
    test_full_pipeline()