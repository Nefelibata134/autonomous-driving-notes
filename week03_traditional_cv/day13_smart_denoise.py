import cv2
import numpy as np


class SmartFilter:
    """
    智能滤波器：自动分析噪声类型并选择合适的滤波算法
    """

    def __init__(self):
        self.methods = {
            'gaussian': self.denoise_gaussian,
            'median': self.denoise_median,
            'bilateral': self.denoise_bilateral,
            'nlm': self.denoise_nlm
        }

    def analyze_noise(self, image):
        """
        简单噪声分析（基于直方图和局部方差）
        返回建议的滤波方法
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 计算局部方差（使用拉普拉斯算子检测噪声水平）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 统计极值点比例（判断椒盐噪声）
        salt_pepper_ratio = np.sum((gray == 255) | (gray == 0)) / gray.size

        print(f"  拉普拉斯方差：{laplacian_var:.2f}")
        print(f"  极值点比例：{salt_pepper_ratio:.4f}")

        if salt_pepper_ratio > 0.01:
            return 'median', '检测到椒盐噪声'
        elif laplacian_var > 500:
            return 'nlm', '检测到高斯噪声（高强度）'
        elif laplacian_var > 200:
            return 'gaussian', '检测到轻度高斯噪声'
        else:
            return 'bilateral', '图像较干净，建议保边平滑'

    def denoise_gaussian(self, image, ksize=5):
        """高斯去噪"""
        if len(image.shape) == 3:
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def denoise_median(self, image, ksize=5):
        """中值去噪"""
        if len(image.shape) == 3:
            # 彩色图中值滤波需要对每个通道分别处理，或转换颜色空间
            return cv2.medianBlur(image, ksize)
        return cv2.medianBlur(image, ksize)

    def denoise_bilateral(self, image, d=9):
        """双边滤波"""
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, d, 75, 75)
        return cv2.bilateralFilter(image, d, 75, 75)

    def denoise_nlm(self, image):
        """非局部均值去噪（效果最好，最慢）"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def auto_denoise(self, image):
        """
        自动去噪主函数
        """
        method, reason = self.analyze_noise(image)
        print(f"建议方法：{method}（{reason}）")

        result = self.methods[method](image)
        return result, method

    def enhance_details(self, image):
        """
        细节增强：先双边滤波去噪，再锐化
        （自动驾驶中增强车道线对比度）
        """
        # 第一步：轻度双边滤波去噪（保边）
        smoothed = cv2.bilateralFilter(image, 5, 50, 50)

        # 第二步：Unsharp Mask 锐化
        blurred = cv2.GaussianBlur(smoothed, (0, 0), 3)
        sharpened = cv2.addWeighted(smoothed, 1.5, blurred, -0.5, 0)

        return sharpened


def test_smart_filter():
    """测试智能滤波器"""
    print("=== 智能滤波器测试 ===\n")

    filter_tool = SmartFilter()

    # 测试 1：带椒盐噪声的图
    print("1. 测试椒盐噪声图像")
    img = cv2.imread('test_image.jpg')
    if img is None:
        print("请放置 test_image.jpg")
        return

    noisy = img.copy()
    h, w = img.shape[:2]
    salt = np.random.rand(h, w) < 0.02
    pepper = np.random.rand(h, w) < 0.02
    noisy[salt] = [255, 255, 255]
    noisy[pepper] = [0, 0, 0]

    result, method = filter_tool.auto_denoise(noisy)
    cv2.imwrite('test_saltpepper_input.jpg', noisy)
    cv2.imwrite('test_saltpepper_output.jpg', result)
    print(f"  使用 {method} 去噪完成\n")

    # 测试 2：细节增强
    print("2. 测试细节增强（车道线增强模拟）")
    enhanced = filter_tool.enhance_details(img)
    cv2.imwrite('test_enhanced.jpg', enhanced)
    print("  细节增强完成")

    # 测试 3：不同方法的对比图
    print("\n3. 生成所有方法对比图")
    methods = ['gaussian', 'median', 'bilateral']
    for m in methods:
        result = filter_tool.methods[m](img)
        cv2.imwrite(f'compare_{m}.jpg', result)
        print(f"  {m} 处理完成")


if __name__ == "__main__":
    test_smart_filter()