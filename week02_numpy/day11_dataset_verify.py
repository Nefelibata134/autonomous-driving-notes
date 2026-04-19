import numpy as np
import cv2
import json


def verify_dataset(npz_path):
    """
    验证生成的数据集，并反归一化显示样本
    """
    print(f"=== 验证数据集：{npz_path} ===")

    # 加载
    data = np.load(npz_path)
    images = data['images']
    labels = data['labels']
    label_names = data['label_names']

    print(f"图像批次：{images.shape}")
    print(f"标签分布：{np.bincount(labels)}")

    # 随机抽取 5 张可视化
    indices = np.random.choice(len(images), 5, replace=False)

    for i, idx in enumerate(indices):
        img_chw = images[idx]
        label = labels[idx]

        # 反预处理：CHW → HWC → RGB → BGR（用于保存）
        img = np.transpose(img_chw, (1, 2, 0))  # CHW → HWC
        img = img * np.array([58.395, 57.12, 57.375]) + np.array([123.675, 116.28, 103.53])
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 保存
        cv2.imwrite(f'sample_{i}_class{label}.jpg', img)
        print(f"  保存样本 {i}：类别 {label_names[label]}")

    print("\n验证完成，请检查 sample_*.jpg 是否正常")


if __name__ == "__main__":
    verify_dataset("output/processed_data.npz")