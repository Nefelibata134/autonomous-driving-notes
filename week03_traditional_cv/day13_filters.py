import cv2
import numpy as np


def add_salt_pepper_noise(image, prob=0.05):
    """添加椒盐噪声（模拟传感器故障）"""
    noisy = image.copy()
    h, w = image.shape[:2]

    # 盐噪声（白点）
    salt = np.random.rand(h, w) < prob / 2
    noisy[salt] = 255

    # 椒噪声（黑点）
    pepper = np.random.rand(h, w) < prob / 2
    noisy[pepper] = 0

    return noisy

def add_gaussian_noise(image, mean=0, sigma=25):
    """添加高斯噪声（模拟低光环境）"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# 读取图像
img = cv2.imread('test_image.jpg')
if img is None:
    print("请放置 test_image.jpg")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=== OpenCV 滤波器实战 ===")

# 1. 高斯模糊（最常用，去噪+平滑）
print("\n1. 高斯模糊")
gaussian_blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.5)
# 参数：(src, ksize, sigmaX)
# ksize: 核大小，必须是正奇数
# sigmaX: X方向标准差，0 表示自动根据 ksize 计算
cv2.imwrite('filter_gaussian.jpg', gaussian_blur)

# 对比不同核大小
gaussian_3x3 = cv2.GaussianBlur(gray, (3, 3), 0)
gaussian_9x9 = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imwrite('filter_gaussian_3x3.jpg', gaussian_3x3)
cv2.imwrite('filter_gaussian_9x9.jpg', gaussian_9x9)
print("核越大越模糊：3x3 < 5x5 < 9x9")

# 2. 中值滤波（去椒盐噪声之王）
print("\n2. 中值滤波")
noisy_sp = add_salt_pepper_noise(gray, prob=0.05)
median_denoised = cv2.medianBlur(noisy_sp, 5)
# 参数：(src, ksize)，ksize 必须是大于1的奇数
cv2.imwrite('noise_salt_pepper.jpg', noisy_sp)
cv2.imwrite('filter_median.jpg', median_denoised)
print("中值滤波对椒盐噪声效果极好，且边缘保持较好")

# 3. 双边滤波（边缘保持平滑，人像美颜/卡通化）
print("\n3. 双边滤波")
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
# d: 邻域直径
# sigmaColor: 颜色空间滤波 sigma（越大越平滑）
# sigmaSpace: 坐标空间滤波 sigma
# 特点：在平坦区域平滑，在边缘处保持锐利
cv2.imwrite('filter_bilateral.jpg', bilateral)
print("双边滤波保持边缘，适合人像/后期处理")

# 4. 方框滤波（简单平均，速度最快）
print("\n4. 方框滤波")
box_blur = cv2.boxFilter(gray, -1, (5, 5), normalize=True)
cv2.imwrite('filter_box.jpg', box_blur)

# 5. 自定义卷积核（cv2.filter2D）
print("\n5. 自定义卷积（filter2D）")

# 锐化
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)
sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
cv2.imwrite('filter_sharpen.jpg', sharpened)

# 浮雕效果
emboss_kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
], dtype=np.float32)
emboss = cv2.filter2D(gray, -1, emboss_kernel)
cv2.imwrite('filter_emboss.jpg', emboss)

# 练习 6：去噪效果对比
print("\n=== 去噪效果对比 ===")
noisy_gaussian = add_gaussian_noise(gray, sigma=30)

# 高斯去噪
denoise_gaussian = cv2.GaussianBlur(noisy_gaussian, (5, 5), 0)

# 快速去噪算法（Non-local Means，计算慢但效果好）
denoise_nlm = cv2.fastNlMeansDenoising(noisy_gaussian, None, 10, 7, 21)
# 参数：(src, None, h, templateWindowSize, searchWindowSize)

cv2.imwrite('noise_gaussian.jpg', noisy_gaussian)
cv2.imwrite('denoise_gaussian.jpg', denoise_gaussian)
cv2.imwrite('denoise_nlm.jpg', denoise_nlm)

print("""
所有结果已保存，请查看对比：
- filter_gaussian_*.jpg: 高斯模糊不同核大小
- filter_median.jpg: 中值滤波去椒盐噪声
- filter_bilateral.jpg: 双边滤波保边
- filter_sharpen.jpg: 锐化
- denoise_*.jpg: 去噪对比
""")