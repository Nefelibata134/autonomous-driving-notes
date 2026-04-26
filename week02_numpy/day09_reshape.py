import numpy as np

# 练习 1：reshape 基础（改变形状，数据不变）
print("=== reshape 基础 ===")
arr = np.arange(12)  # [0, 1, 2, ..., 11]
print(f"原数组：{arr}, 形状：{arr.shape}")

# 变 2D
arr_2d = arr.reshape(3, 4)  # 3行4列
print(f"\nreshape(3,4)：\n{arr_2d}")
print(f"形状：{arr_2d.shape}")

# 自动推断维度（-1 表示"自动计算"）
arr_auto = arr.reshape(2, -1)  # 2行，列数自动=6
print(f"\nreshape(2, -1)：\n{arr_auto}")
print(f"形状：{arr_auto.shape}")

# 变 3D（图像 Batch：B, H, W）
arr_3d = arr.reshape(2, 2, 3)  # 2个2x3的"图"
print(f"\nreshape(2,2,3)：\n{arr_3d}")

# 练习 2：reshape 是视图（共享内存！）
print("\n=== reshape 是视图（修改会影响原数组） ===")
arr = np.arange(6)
view = arr.reshape(2, 3)

print(f"修改前原数组：{arr}")
view[0, 0] = 999  # 修改视图
print(f"修改后原数组：{arr}")  # 原数组也被改了！

# 如果需要独立副本
arr_copy = arr.reshape(2, 3).copy()
arr_copy[0, 0] = 888
print(f"修改副本后原数组：{arr}")  # 原数组不变

# 练习 3：flatten vs ravel（降维）
print("\n=== flatten vs ravel ===")
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# flatten：总是返回副本（安全但慢）
flat_copy = arr_2d.flatten()
flat_copy[0] = 999
print(f"flatten 修改后原数组：\n{arr_2d}")  # 不变

# ravel：尽量返回视图（快但可能共享内存）
flat_view = arr_2d.ravel()
flat_view[0] = 888
print(f"ravel 修改后原数组：\n{arr_2d}")  # 可能被改（取决于内存布局）

# 练习 4：squeeze 与 expand_dims（增减维度）
print("\n=== squeeze 与 expand_dims ===")

# squeeze：移除长度为1的维度
arr = np.arange(6).reshape(1, 2, 1, 3)  # 形状 (1, 2, 1, 3)
print(f"原形状：{arr.shape}")
squeezed = np.squeeze(arr)
print(f"squeeze 后：{squeezed.shape}")  # (2, 3)

# expand_dims：在指定位置增加维度
arr = np.arange(3)  # (3,)
expanded = np.expand_dims(arr, axis=0)  # 位置0增加维度
print(f"\nexpand_dims(axis=0)：{expanded.shape}")  # (1, 3)

expanded2 = np.expand_dims(arr, axis=1)  # 位置1增加维度
print(f"expand_dims(axis=1)：{expanded2.shape}")  # (3, 1)

# 练习 5：实际应用 - 图像尺寸调整
print("\n=== 实际应用：图像展平与恢复 ===")
# 模拟 2x2 的灰度图
img = np.array([[10, 20], [30, 40]])
print(f"原图：\n{img}, 形状：{img.shape}")

# 展平为向量（用于全连接层输入）
flat = img.flatten()
print(f"展平：{flat}, 形状：{flat.shape}")

# 恢复原形状（用于卷积层）
restored = flat.reshape(2, 2)
print(f"恢复：\n{restored}")

# 练习 6：reshape 在 Batch 处理中的应用
print("\n=== Batch 图像 reshape ===")
# 模拟 4 张 3x3 的灰度图
batch = np.arange(36).reshape(4, 3, 3)
print(f"Batch 形状：{batch.shape}")  # (4, 3, 3)

# 展平每张图（用于全连接分类器）
batch_flat = batch.reshape(4, -1)  # (4, 9)
print(f"展平后：{batch_flat.shape}")

# 恢复（用于卷积）
batch_restored = batch_flat.reshape(4, 3, 3)
print(f"恢复后：{batch_restored.shape}")

