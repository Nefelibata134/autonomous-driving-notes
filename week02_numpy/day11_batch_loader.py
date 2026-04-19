import os
from pathlib import Path
import cv2
import numpy as np

# 练习 1：三种遍历方式对比
print("=== 文件遍历方法 ===")

# 方法 1：os.listdir（简单，但不递归）
image_dir = "test_images"  # 假设有这个文件夹
if os.path.exists(image_dir):
    files = os.listdir(image_dir)
    images = [f for f in files if f.endswith(('.jpg', '.png'))]
    print(f"listdir 找到 {len(images)} 张图")

# 方法 2：os.walk（递归遍历子文件夹，最常用）
print("\n=== os.walk 递归遍历 ===")
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            full_path = os.path.join(root, file)
            print(f"  找到：{full_path}")

# 方法 3：pathlib（现代 Python 推荐，面向对象）
print("\n=== pathlib 遍历 ===")
path_obj = Path(image_dir)
image_paths = list(path_obj.glob("**/*.jpg"))  # 递归找所有 jpg
image_paths.extend(path_obj.glob("**/*.png"))
print(f"pathlib 找到 {len(image_paths)} 张图")

# 练习 2：生成器模式（处理成千上万张图时节省内存）
print("\n=== 生成器模式（内存友好） ===")


def image_generator(folder_path, target_size=None):
    """
    生成器：按需加载图像，不一次性载入所有
    """
    path_obj = Path(folder_path)
    for img_path in path_obj.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 读取
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 可选：预处理
            if target_size:
                img = cv2.resize(img, target_size)

            # 生成（yield）：暂停并返回，下次从这里继续
            yield {
                'path': str(img_path),
                'name': img_path.name,
                'data': img,
                'shape': img.shape
            }


# 使用生成器（每次只处理一张，内存占用恒定）
if os.path.exists(image_dir):
    gen = image_generator(image_dir, target_size=(224, 224))
    for i, item in enumerate(gen):
        print(f"  处理第 {i + 1} 张：{item['name']}, 形状：{item['shape']}")
        if i >= 4:  # 只看前5张
            break

# 练习 3：文件夹结构解析（模拟数据集）
print("\n=== 解析数据集结构 ===")


def parse_dataset_structure(root_dir):
    """
    解析常见数据集结构：
    root/
      class_a/
        img1.jpg
        img2.jpg
      class_b/
        img3.jpg
    """
    dataset = {}
    root = Path(root_dir)

    for class_dir in root.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            images = list(class_dir.glob("*.jpg"))
            images.extend(class_dir.glob("*.png"))
            dataset[class_name] = [str(p) for p in images]
            print(f"类别 {class_name}: {len(images)} 张图")

    return dataset


def load_batch(image_paths, target_size=(224, 224)):
    """
    批量加载图像，返回 NumPy Batch 数组
    适用于中小数据集（内存足够容纳整个 batch）
    """
    batch = []  # 存储图像数组的列表
    valid_paths = []  # 记录成功加载的路径（与 batch 一一对应）

    for i, path in enumerate(image_paths):
        # enumerate 同时提供索引 i 和元素 path

        if i % 10 == 0:  # 每处理 10 张更新一次进度
            print(f"  进度：{i}/{len(image_paths)}", end='\r')
            # end='\r'：回车不换行，覆盖当前行（动态进度条效果）
            # 在 Jupyter 等环境可能需要改为 print(f"...", end='\r', flush=True)

        img = cv2.imread(path)
        if img is None:  # 跳过损坏文件
            continue

        img = cv2.resize(img, target_size)  # 统一尺寸，确保能 stack 为数组
        batch.append(img)
        valid_paths.append(path)

    print(f"\n成功加载 {len(batch)}/{len(image_paths)} 张")
    # \n：最后换行，防止后续打印与进度条混在一行

    return np.array(batch), valid_paths
    # np.array(batch)：将列表转为 NumPy 数组，形状 (B, H, W, 3)
    # B = len(batch)，自动推断
    # valid_paths：用于调试（知道哪些图成功加载，哪些失败）