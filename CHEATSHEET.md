# Python + NumPy 速查手册

## Week 1: Python 基础
### 列表推导式
[expression for item in iterable if condition]

### 字典操作
d.get(key, default)  # 安全获取
d.items()  # 遍历键值

### 文件读写
with open(file, 'r', encoding='utf-8') as f:
    content = f.read()

## Week 2: NumPy 核心
### 数组创建
np.zeros((H, W, 3), dtype=np.uint8)
np.random.randint(0, 256, (H, W, 3))

### 形状操作
arr.reshape(B, -1)  # -1 自动计算
arr.transpose(0, 3, 1, 2)  # BHWC -&gt; BCHW
np.expand_dims(arr, axis=0)  # 增加 batch 维度

### 广播机制
(H,W,3) - (3,)  # 自动在 H,W 扩展
保持维度一致：mean.reshape(1,1,3)

### 内存优化
np.ascontiguousarray(arr)  # 确保连续
arr.copy()  # 断开引用

## OpenCV 关键
### 读取与转换
img = cv2.imread(path)  # BGR uint8
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### 预处理 Pipeline
1. cv2.resize(img, (W, H))
2. cvtColor BGR-&gt;RGB
3. astype(float32) / 255.0  # 归一化
4. transpose(2,0,1)  # HWC-&gt;CHW
5. np.ascontiguousarray()
6. expand_dims(0)  # 加 batch