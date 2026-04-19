import cv2
import numpy as np
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor  # 多线程加速


class BatchProcessor:
    """
    批量图像处理器
    功能：文件夹 → 预处理 → 保存为 NumPy 数据集
    """

    def __init__(self, target_size=(224, 224), augment=True):
        self.target_size = target_size
        self.augment = augment
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'classes': {}
        }

    def process_single(self, image_path, label=None):
        """
        处理单张图像（整合 Day 9 增强 + Day 10 Pipeline）
        """
        try:
            # 1. 读取
            img = cv2.imread(str(image_path))
            if img is None:
                self.stats['failed'] += 1
                return None

            # 2. 数据增强（Day 9 内容）
            if self.augment:
                # 随机水平翻转
                if np.random.random() > 0.5:
                    img = cv2.flip(img, 1)

                # 随机旋转 90 度的倍数
                k = np.random.randint(0, 4)
                if k > 0:
                    img = np.rot90(img, k)

            # 3. 基础处理（Day 10 Pipeline）
            img = cv2.resize(img, self.target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) - self.mean) / self.std
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            img = np.ascontiguousarray(img)

            self.stats['success'] += 1

            return {
                'image': img,
                'label': label,
                'path': str(image_path)
            }

        except Exception as e:
            print(f"处理失败 {image_path}: {e}")
            self.stats['failed'] += 1
            return None

    def process_folder(self, input_dir, output_dir, max_workers=4):
        """
        批量处理文件夹（支持多线程）
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 收集所有图片路径（假设按文件夹分类，文件夹名即标签）
        image_list = []
        for class_dir in input_path.iterdir():
            if class_dir.is_dir():
                label = class_dir.name
                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        image_list.append((img_path, label))

        self.stats['total'] = len(image_list)
        print(f"找到 {len(image_list)} 张待处理图像")

        # 处理（单线程版本，适合学习调试）
        results = []
        for i, (img_path, label) in enumerate(image_list):
            if i % 20 == 0:
                print(f"进度：{i}/{len(image_list)} ({100 * i / len(image_list):.1f}%)")

            result = self.process_single(img_path, label)
            if result:
                results.append(result)

        # 多线程版本（生产环境使用，取消注释即可）
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     futures = [executor.submit(self.process_single, p, l) for p, l in image_list]
        #     results = [f.result() for f in futures if f.result()]

        return results

    def save_numpy_dataset(self, results, output_path):
        """
        保存为 NumPy 数据集文件（.npz 压缩格式）
        """
        if not results:
            print("没有数据可保存")
            return

        # 分离图像和标签
        images = np.array([r['image'] for r in results])  # (B, C, H, W)
        labels = [r['label'] for r in results]
        paths = [r['path'] for r in results]

        # 标签转整数（类别编码）
        unique_labels = sorted(list(set(labels)))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        int_labels = np.array([label_to_int[l] for l in labels])

        # 保存为 npz（压缩，包含多个数组）
        np.savez_compressed(
            output_path,
            images=images,
            labels=int_labels,
            label_names=unique_labels,
            paths=paths
        )

        print(f"\n数据集已保存：{output_path}")
        print(f"  图像数组：{images.shape}, 类型：{images.dtype}")
        print(f"  标签数组：{int_labels.shape}")
        print(f"  类别映射：{label_to_int}")
        print(f"  文件大小：{Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

        # 同时保存 JSON 格式的标签映射（方便查看）
        json_path = str(output_path).replace('.npz', '_labels.json')
        with open(json_path, 'w') as f:
            json.dump({
                'label_to_int': label_to_int,
                'int_to_label': {v: k for k, v in label_to_int.items()},
                'total_images': len(images),
                'image_shape': list(images.shape[1:])  # (C,H,W)
            }, f, indent=2)
        print(f"  标签映射：{json_path}")


# 使用示例
if __name__ == "__main__":
    processor = BatchProcessor(target_size=(224, 224), augment=True)

    # 假设目录结构：
    # dataset/
    #   cars/
    #     001.jpg
    #     002.jpg
    #   persons/
    #     003.jpg

    input_dir = "dataset"
    if Path(input_dir).exists():
        results = processor.process_folder(input_dir, "output")
        if results:
            processor.save_numpy_dataset(results, "output/processed_data.npz")
            print(f"\n处理统计：{processor.stats}")
    else:
        print(f"请创建 {input_dir} 文件夹并放入一些图片测试")
        print("建议结构：dataset/类别名/图片.jpg")