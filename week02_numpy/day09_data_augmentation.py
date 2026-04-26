import numpy as np


class DataAugmentor:
    """
    图像数据增强器（纯 NumPy 实现）
    用于深度学习训练时的数据扩充
    """

    def __init__(self, seed=None):
        """初始化，可选随机种子保证可复现"""
        if seed is not None:
            np.random.seed(seed)

    def random_crop(self, image, crop_size):
        """
        随机裁剪
        image: (H, W, C) 或 (H, W)
        crop_size: (crop_h, crop_w)
        """
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size

        # 随机选择左上角
        max_y = h - crop_h
        max_x = w - crop_w

        if max_y < 0 or max_x < 0:
            raise ValueError(f"裁剪尺寸{crop_size}大于图像尺寸({h},{w})")

        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)

        if image.ndim == 3:
            return image[y:y + crop_h, x:x + crop_w, :]
        else:
            return image[y:y + crop_h, x:x + crop_w]

    def center_crop(self, image, crop_size):
        """中心裁剪（验证时常用）"""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size

        y = (h - crop_h) // 2
        x = (w - crop_w) // 2

        if image.ndim == 3:
            return image[y:y + crop_h, x:x + crop_w, :]
        else:
            return image[y:y + crop_h, x:x + crop_w]

    def random_horizontal_flip(self, image, prob=0.5):
        """随机水平翻转"""
        if np.random.random() < prob:
            if image.ndim == 3:
                return image[:, ::-1, :]  # W 维度翻转
            else:
                return image[:, ::-1]
        return image

    def random_vertical_flip(self, image, prob=0.5):
        """随机垂直翻转"""
        if np.random.random() < prob:
            if image.ndim == 3:
                return image[::-1, :, :]  # H 维度翻转
            else:
                return image[::-1, :]
        return image

    def rotate_90(self, image, k=1):
        """
        旋转 90 度的 k 倍（k=1 是 90度，k=2 是 180度，k=3 是 270度）
        """
        # np.rot90 默认是 (H,W) 或 (H,W,C) 的前两个维度
        return np.rot90(image, k=k, axes=(0, 1))

    def normalize(self, image, mean, std):
        """归一化（支持 HWC 格式）"""
        image = image.astype(np.float32)
        mean = np.array(mean).reshape(1, 1, -1)
        std = np.array(std).reshape(1, 1, -1)
        return (image - mean) / std

    def to_chw(self, image):
        """HWC → CHW，并确保内存连续"""
        if image.ndim == 3:
            chw = image.transpose(2, 0, 1)
        else:
            chw = image[np.newaxis, :, :]  # 灰度图增加通道维度
        return np.ascontiguousarray(chw)

    def augment_single(self, image, crop_size=None, normalize_params=None):
        """
        单张图像的完整增强流程

        Args:
            image: 输入图像 (H, W, C)
            crop_size: 随机裁剪尺寸，None 则不裁剪
            normalize_params: (mean, std) 元组，None 则不归一化

        Returns:
            增强后的图像，CHW 格式
        """
        # 1. 随机裁剪（或中心裁剪）
        if crop_size is not None:
            image = self.random_crop(image, crop_size)

        # 2. 随机翻转
        image = self.random_horizontal_flip(image)
        image = self.random_vertical_flip(image, prob=0.3)  # 垂直翻转概率低一些

        # 3. 随机旋转（0/90/180/270 度）
        k = np.random.randint(0, 4)  # 0, 1, 2, 3
        if k > 0:
            image = self.rotate_90(image, k)

        # 4. 归一化
        if normalize_params is not None:
            mean, std = normalize_params
            image = self.normalize(image, mean, std)

        # 5. 转为 CHW 并确保连续
        image = self.to_chw(image)

        return image

    def augment_batch(self, images, crop_size=None, normalize_params=None):
        """
        Batch 增强
        images: 列表 of (H, W, C) 数组
        """
        processed = []
        for img in images:
            aug_img = self.augment_single(img, crop_size, normalize_params)
            processed.append(aug_img)

        # Stack 成 Batch
        batch = np.stack(processed, axis=0)  # (B, C, H, W)
        return batch


def test_augmentor():
    """测试数据增强器"""
    print("=== 测试数据增强器 ===\n")

    augmentor = DataAugmentor(seed=42)  # 固定种子，结果可复现

    # 创建测试图像（100x100 的 RGB 图）
    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    print(f"原图形状：{test_img.shape}")

    # 1. 随机裁剪
    cropped = augmentor.random_crop(test_img, (64, 64))
    print(f"\n随机裁剪后：{cropped.shape}")

    # 2. 中心裁剪
    center = augmentor.center_crop(test_img, (64, 64))
    print(f"中心裁剪后：{center.shape}")

    # 3. 翻转
    flipped = augmentor.random_horizontal_flip(test_img, prob=1.0)  # 强制翻转
    print(f"\n水平翻转后形状：{flipped.shape}")
    print(f"原图[0,0]={test_img[0, 0]}, 翻转后[0,-1]={flipped[0, -1]}（应相等）")

    # 4. 旋转
    rotated = augmentor.rotate_90(test_img, k=1)
    print(f"\n旋转90度后：{rotated.shape}")  # (100, 100, 3) 但内容旋转了

    # 5. 完整增强流程
    print("\n=== 完整增强流程 ===")
    aug_img = augmentor.augment_single(
        test_img,
        crop_size=(64, 64),
        normalize_params=([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
    )
    print(f"增强后：{aug_img.shape}, 格式：CHW")
    print(f"数据范围：[{aug_img.min():.2f}, {aug_img.max():.2f}]")
    print(f"内存连续：{aug_img.flags['C_CONTIGUOUS']}")

    # 6. Batch 增强
    print("\n=== Batch 增强 ===")
    batch_imgs = [test_img for _ in range(4)]  # 4 张相同的图
    batch_aug = augmentor.augment_batch(
        batch_imgs,
        crop_size=(64, 64),
        normalize_params=([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
    )
    print(f"Batch 增强后：{batch_aug.shape}")  # (4, 3, 64, 64)


if __name__ == "__main__":
    test_augmentor()