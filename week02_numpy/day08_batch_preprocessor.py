import numpy as np


class ImagePreprocessor:
    """
    图像预处理器（支持单张和 Batch 处理）
    用于深度学习数据准备
    """

    def __init__(self, target_size=(224, 224),
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        """
        初始化预处理器

        Args:
            target_size: (height, width)，输出尺寸
            mean: RGB 均值
            std: RGB 标准差
        """
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def resize(self, image):
        """
        简单最近邻 resize（实际应用可用 OpenCV，这里用 NumPy 演示概念）
        注意：真实场景建议使用 cv2.resize
        """
        h, w = self.target_size
        if image.ndim == 3:
            # 简单下采样/上采样（每隔几个像素取一个，或复制）
            # 这里为了演示，使用简单的切片方式（不完全精确但展示概念）
            old_h, old_w = image.shape[:2]
            row_scale = old_h / h
            col_scale = old_w / w

            row_idx = (np.arange(h) * row_scale).astype(int)
            col_idx = (np.arange(w) * col_scale).astype(int)

            return image[np.ix_(row_idx, col_idx)]
        return image

    def normalize(self, image):
        """
        归一化：减去均值，除以标准差
        支持单张 (H,W,3) 或 Batch (B,H,W,3)
        """
        image = image.astype(np.float32)

        # 广播减法：(H,W,3) 或 (B,H,W,3) 减去 (1,1,3)
        normalized = (image - self.mean) / self.std

        return normalized

    def hwc_to_chw(self, image):
        """
        转换维度顺序：HWC → CHW（PyTorch 需要）

        Args:
            image: (H,W,C) 或 (B,H,W,C)
        """
        if image.ndim == 3:
            # 单张图：(H,W,C) → (C,H,W)
            return np.transpose(image, (2, 0, 1))
        else:
            # Batch：(B,H,W,C) → (B,C,H,W)
            return np.transpose(image, (0, 3, 1, 2))

    def preprocess_single(self, image):
        """
        处理单张图像

        Returns:
            处理后的图像，形状 (C, H, W)
        """
        # 1. Resize
        image = self.resize(image)
        # 2. Normalize
        image = self.normalize(image)
        # 3. HWC to CHW
        image = self.hwc_to_chw(image)

        return image

    def preprocess_batch(self, images):
        """
        处理批量图像

        Args:
            images: 列表 of (H,W,3) 数组，或形状 (B,H,W,C) 的数组

        Returns:
            Batch 数组，形状 (B, C, H, W)
        """
        # 如果是列表，先 stack 成 Batch
        if isinstance(images, list):
            images = np.stack(images, axis=0)  # (B, H, W, C)

        # Batch 处理（利用广播）
        images = self.resize_batch(images)
        images = self.normalize(images)  # 广播自动处理 Batch 维度
        images = self.hwc_to_chw(images)

        return images

    def resize_batch(self, batch_images):
        """Batch resize（这里简化处理，实际应逐张或向量化）"""
        # 为了演示，简单返回原图（假设尺寸已正确）
        # 真实场景需要实现高效的 batch resize
        return batch_images

    def augment_flip(self, batch_images, horizontal=True):
        """
        数据增强：随机翻转（Batch 操作）

        Args:
            batch_images: (B, C, H, W) 或 (B, H, W, C)
            horizontal: True 为水平翻转，False 为垂直
        """
        if horizontal:
            # 水平翻转：翻转 W 维度（索引 -1 或 3，取决于格式）
            return batch_images[..., ::-1] if batch_images.ndim == 4 else batch_images[:, ::-1, :]
        else:
            # 垂直翻转：翻转 H 维度
            return batch_images[:, ::-1, ...]


def test_preprocessor():
    """测试预处理器"""
    print("=== 测试 Batch 图像预处理器 ===\n")

    preprocessor = ImagePreprocessor(target_size=(64, 64))

    # 1. 单张图测试
    print("1. 单张图处理")
    single_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    processed = preprocessor.preprocess_single(single_img)
    print(f"输入：{single_img.shape} → 输出：{processed.shape}")
    print(f"输出范围：[{processed.min():.2f}, {processed.max():.2f}]")

    # 2. Batch 处理测试
    print("\n2. Batch 处理")
    batch = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
    batch_processed = preprocessor.preprocess_batch(batch)
    print(f"输入：4张 {batch[0].shape} → 输出：{batch_processed.shape}")
    print(f"Batch 均值：{batch_processed.mean():.4f}（应接近 0）")

    # 3. 数据增强测试
    print("\n3. 数据增强（水平翻转）")
    flipped = preprocessor.augment_flip(batch_processed, horizontal=True)
    print(f"翻转后形状：{flipped.shape}")
    print(f"原图[0,0,0]={batch_processed[0, 0, 0, 0]:.4f}, 翻转后[0,0,0,-1]={flipped[0, 0, 0, -1]:.4f}")

`
if __name__ == "__main__":
    test_preprocessor()