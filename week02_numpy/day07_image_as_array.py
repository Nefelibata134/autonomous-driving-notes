import numpy as np


def create_checkerboard(size=8, block_size=50):
    """
    创建棋盘格图像（黑白交替）
    用于相机标定、图像对齐测试
    """
    # 创建 8x8 的 0-1 矩阵
    board = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                board[i, j] = 255  # 白

    # 放大每个格子到 50x50 像素（图像上采样）
    # 用 repeat 实现最近邻插值
    board_img = np.repeat(np.repeat(board, block_size, axis=0), block_size, axis=1)
    return board_img


def create_gradient_image(height=100, width=200):
    """
    创建渐变图（从左到右由黑变白）
    用于测试图像显示和色彩映射
    """
    # 创建线性梯度
    gradient = np.linspace(0, 255, width)  # 1行，width列
    # 复制 height 行，形成渐变图
    img = np.tile(gradient, (height, 1)).astype(np.uint8)
    return img


def add_noise(img, noise_level=25):
    """
    给图像添加高斯噪声（模拟真实相机噪声）
    noise_level: 噪声强度（标准差）
    """
    noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    # 裁剪到 0-255 范围
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def crop_center(img, crop_height, crop_width):
    """
    中心裁剪（深度学习预处理常用）
    """
    h, w = img.shape[:2]
    start_h = (h - crop_height) // 2
    start_w = (w - crop_width) // 2
    return img[start_h:start_h + crop_height, start_w:start_w + crop_width]


def flip_image(img, direction='horizontal'):
    """
    图像翻转（数据增强）
    direction: 'horizontal'（水平）或 'vertical'（垂直）
    """
    if direction == 'horizontal':
        return img[:, ::-1]  # 列反转
    else:
        return img[::-1, :]  # 行反转


def main():
    print("=== 图像作为 NumPy 数组 ===\n")

    # 1. 创建棋盘格
    print("1. 创建棋盘格...")
    checker = create_checkerboard(size=8, block_size=50)
    print(f"棋盘格形状：{checker.shape}, 内存：{checker.nbytes} bytes")
    print(f"左上角像素值：{checker[0, 0]}（应该是255白色）")

    # 2. 创建彩色图像（RGB）
    print("\n2. 创建彩色图像...")
    # 创建 100x100 的 RGB 图
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # 红色区域（左上）
    color_img[:50, :50] = [255, 0, 0]
    # 绿色区域（右上）
    color_img[:50, 50:] = [0, 255, 0]
    # 蓝色区域（左下）
    color_img[50:, :50] = [0, 0, 255]
    # 白色区域（右下）
    color_img[50:, 50:] = [255, 255, 255]
    print(f"彩色图形状：{color_img.shape}（HWC格式）")
    print(f"中心像素值：{color_img[50, 50]}（应该是白色[255,255,255]）")

    # 3. 分离通道并统计
    print("\n3. 通道分离与统计...")
    r, g, b = color_img[:, :, 0], color_img[:, :, 1], color_img[:, :, 2]
    print(f"R通道平均值：{r.mean():.1f}（应该是127.5，一半255一半0）")
    print(f"G通道最大值：{g.max()}，最小值：{g.min()}")

    # 4. 添加噪声
    print("\n4. 添加噪声...")
    noisy_img = add_noise(checker, noise_level=30)
    print(f"噪声图形状：{noisy_img.shape}")
    print(f"原图[0,0]={checker[0, 0]}, 噪声图[0,0]={noisy_img[0, 0]}")

    # 5. 中心裁剪
    print("\n5. 中心裁剪...")
    cropped = crop_center(checker, crop_height=200, crop_width=200)
    print(f"裁剪后形状：{cropped.shape}（原图{checker.shape}）")

    # 6. 水平翻转（数据增强）
    print("\n6. 水平翻转...")
    flipped = flip_image(checker, 'horizontal')
    print(f"原图[0,0]={checker[0, 0]}, 翻转后[0,-1]={flipped[0, -1]}（应该相等）")

    # 7. 保存为原始数据（展示图像是数组）
    print("\n7. 保存为 numpy 数组文件...")
    np.save('checkerboard.npy', checker)  # 保存为 .npy 文件
    loaded = np.load('checkerboard.npy')  # 加载回来
    print(f"保存后加载验证：形状{loaded.shape}, 数据一致：{np.array_equal(checker, loaded)}")

    print("\n=== 所有操作完成！===")
    print("关键理解：图像就是 NumPy 数组，所有操作都是数组操作")


if __name__ == "__main__":
    main()