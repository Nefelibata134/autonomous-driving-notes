import cv2
import numpy as np
import os


class ImagePipeline:
    """
    真实图像预处理 Pipeline
    整合 OpenCV 读取 + NumPy 处理 + 深度学习格式输出
    """

    def __init__(self, target_size=(224, 224),
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        self.target_size = target_size  # (width, height) - OpenCV 用
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def load(self, image_path):
        """加载图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到文件：{image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")

        print(f"已加载：{image_path}, 形状：{img.shape}")
        return img

    def resize(self, img, keep_aspect=False):
        """
        调整尺寸
        keep_aspect: 是否保持长宽比（保持则短边对齐，其余 pad）
        """
        tw, th = self.target_size

        if not keep_aspect:
            # 直接 resize（可能变形）
            return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)

        # 保持长宽比（Letterbox 方式，YOLO 常用）
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建画布并居中放置
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)  # 灰色填充
        y_offset = (th - new_h) // 2
        x_offset = (tw - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def to_rgb(self, img):
        """BGR → RGB"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def normalize(self, img):
        """
        归一化
        img: uint8 RGB, 范围 [0, 255]
        返回: float32, 范围约 [-2, +2]
        """
        img = img.astype(np.float32)
        # 广播：(H,W,3) - (3,) → 自动扩展
        img = (img - self.mean) / self.std
        return img

    def to_chw(self, img):
        """HWC → CHW"""
        return np.transpose(img, (2, 0, 1))

    def preprocess(self, image_path, keep_aspect=False, return_original=False):
        """
        完整预处理流程

        Returns:
            chw_image: (C, H, W) float32 数组，内存连续
            original: 原始 BGR 图（可选）
        """
        # 1. 加载
        img = self.load(image_path)
        original = img.copy() if return_original else None

        # 2. Resize
        img = self.resize(img, keep_aspect=keep_aspect)

        # 3. BGR → RGB
        img = self.to_rgb(img)

        # 4. 归一化
        img = self.normalize(img)

        # 5. HWC → CHW
        img = self.to_chw(img)

        # 6. 确保内存连续（关键！）
        img = np.ascontiguousarray(img)

        # 7. 增加 Batch 维度（从 (C,H,W) → (1,C,H,W)）
        img = np.expand_dims(img, axis=0)

        return (img, original) if return_original else img

    def preprocess_batch(self, image_paths, keep_aspect=False):
        """
        Batch 处理多张图
        """
        processed = []
        for path in image_paths:
            single = self.preprocess(path, keep_aspect=keep_aspect)
            processed.append(single[0] if isinstance(single, tuple) else single)

        # Stack: [(1,C,H,W), (1,C,H,W), ...] → (B,C,H,W)
        batch = np.vstack(processed)
        return batch

    def save_tensor_visual(self, chw_image, save_path):
        """
        将处理后的张量转回可视化图像（用于检查预处理是否正确）
        chw_image: (1,C,H,W) 或 (C,H,W)
        """
        if chw_image.ndim == 4:
            chw_image = chw_image[0]  # 去掉 batch 维度

        # CHW → HWC
        img = np.transpose(chw_image, (1, 2, 0))

        # 反归一化
        img = img * self.std + self.mean

        # 裁剪到 0-255 并转 uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        # RGB → BGR 保存
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)
        print(f"可视化图像已保存：{save_path}")


def test_pipeline():
    """测试 Pipeline"""
    print("=== 测试图像预处理 Pipeline ===\n")

    pipeline = ImagePipeline(target_size=(224, 224))

    # 1. 单张图处理
    print("1. 单张图预处理")
    try:
        tensor = pipeline.preprocess('test_image.jpg', keep_aspect=False)
        print(f"输出张量形状：{tensor.shape}")  # (1, 3, 224, 224)
        print(f"数据类型：{tensor.dtype}")  # float32
        print(f"内存连续：{tensor.flags['C_CONTIGUOUS']}")
        print(f"数值范围：[{tensor.min():.2f}, {tensor.max():.2f}]")

        # 保存可视化检查
        pipeline.save_tensor_visual(tensor, 'check_preprocess.jpg')

    except Exception as e:
        print(f"处理失败：{e}")
        print("请确保 test_image.jpg 存在")
        return

    # 2. 保持长宽比处理
    print("\n2. 保持长宽比预处理")
    tensor_letter = pipeline.preprocess('test_image.jpg', keep_aspect=True)
    print(f"Letterbox 输出：{tensor_letter.shape}")
    pipeline.save_tensor_visual(tensor_letter, 'check_letterbox.jpg')

    # 3. Batch 处理
    print("\n3. Batch 预处理")
    # 复制同一张图模拟 Batch
    batch_paths = ['test_image.jpg'] * 4
    batch_tensor = pipeline.preprocess_batch(batch_paths, keep_aspect=False)
    print(f"Batch 输出：{batch_tensor.shape}")  # (4, 3, 224, 224)
    print(f"Batch 内存连续：{batch_tensor.flags['C_CONTIGUOUS']}")


if __name__ == "__main__":
    test_pipeline()