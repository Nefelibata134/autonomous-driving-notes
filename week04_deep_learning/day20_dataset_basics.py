import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path

print("=== 自定义 Dataset ===")

# 练习 1：最简单的 Dataset（纯 Tensor）
print("\n1. 纯 Tensor Dataset")


class TensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # (N, D)
        self.labels = labels  # (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 创建假数据
fake_data = torch.randn(100, 10)  # 100个样本，每个10维
fake_labels = torch.randint(0, 2, (100,))  # 二分类标签

simple_dataset = TensorDataset(fake_data, fake_labels)
print(f"数据集大小: {len(simple_dataset)}")
x, y = simple_dataset[0]
print(f"第0个样本: x形状={x.shape}, y={y}")

# 练习 2：图像 Dataset（OpenCV 读取 + 预处理）
print("\n2. 图像 Dataset")


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        image_dir: 图片文件夹路径
        transform: 预处理函数（可选）
        """
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.jpg"))
        self.image_paths.extend(self.image_dir.glob("*.png"))
        self.transform = transform

        print(f"找到 {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        path = str(self.image_paths[idx])
        img = cv2.imread(path)
        if img is None:
            # 如果读取失败，返回一张黑图（实际应用应更严谨）
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 统一尺寸
        img = cv2.resize(img, (224, 224))

        # 预处理（如果提供）
        if self.transform:
            img = self.transform(img)

        return img



# 测试（如果 test_images 文件夹存在）
import os

if os.path.exists("test_images"):
    img_dataset = ImageDataset("test_images")
    sample_img = img_dataset[0]
    print(f"第0张图形状: {sample_img.shape}, 类型: {sample_img.dtype}")
else:
    print("请创建 test_images 文件夹并放入图片测试")
