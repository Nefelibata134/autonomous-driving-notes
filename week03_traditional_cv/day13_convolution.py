import numpy as np
import cv2

# 练习 1：手动实现 2D 卷积（理解原理）
print("=== 手动卷积实现 ===")


def manual_convolve2d(image, kernel):
    """
    手动实现灰度图卷积（无 padding，步长 1）
    image: 2D numpy array
    kernel: 2D numpy array（奇数尺寸）
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # 输出尺寸（无 padding 会缩小）
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # 滑动窗口
    for y in range(out_h):
        for x in range(out_w):
            # 提取对应区域
            region = image[y:y + kh, x:x + kw]
            # 逐元素乘，然后求和
            output[y, x] = np.sum(region * kernel)

    return output


# 测试：简单边缘检测核（检测垂直边缘）
vertical_edge_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)
print("Sobel 垂直边缘核：\n", vertical_edge_kernel)

# 创建测试图像（10x10，中间有一条竖线）
test_img = np.zeros((10, 10), dtype=np.float32)
test_img[:, 5:7] = 255  # 白色竖线
print(f"\n原图（中间有竖线）：\n{test_img}")

# 手动卷积
convolved = manual_convolve2d(test_img, vertical_edge_kernel)
print(f"\n卷积后（边缘被增强）：\n{convolved}")

# 练习 2：理解卷积核类型
print("\n=== 常见卷积核 ===")

# 均值模糊（平滑）
mean_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
print(f"均值核：\n{mean_kernel}")

# 高斯核（中心权重高，边缘权重低）
gaussian_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0
print(f"\n3x3 高斯核：\n{gaussian_kernel}")

# 锐化核
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)
print(f"\n锐化核：\n{sharpen_kernel}")

# 练习 3：NumPy 向量化实现（比双重 for 循环快 100 倍）
print("\n=== 向量化卷积（使用 NumPy 滑动窗口） ===")


def fast_convolve(image, kernel):
    """使用 as_strided 实现快速滑动窗口（了解即可）"""
    from numpy.lib.stride_tricks import as_strided

    h, w = image.shape
    kh, kw = kernel.shape
    out_h, out_w = h - kh + 1, w - kw + 1

    # 创建滑动窗口视图
    shape = (out_h, out_w, kh, kw)
    strides = image.strides + image.strides
    windows = as_strided(image, shape=shape, strides=strides)

    # 向量化计算
    return np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))


# 性能对比
import time

test_large = np.random.rand(100, 100).astype(np.float32)

t1 = time.time()
result_slow = manual_convolve2d(test_large, gaussian_kernel)
t2 = time.time()

t3 = time.time()
result_fast = fast_convolve(test_large, gaussian_kernel)
t4 = time.time()

print(f"手动 for 循环：{t2 - t1:.4f}s")
print(f"向量化实现：{t4 - t3:.4f}s")
print(f"加速比：{(t2 - t1) / (t4 - t3):.1f}x")