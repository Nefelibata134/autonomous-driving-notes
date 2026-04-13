# 练习 1：函数定义与调用
print("=== 基础函数 ===")

def greet(name):
    """打招呼函数（这是文档字符串 docstring）"""
    return f"你好，{name}！"

# 调用
msg = greet("张三")
print(msg)

# 练习 2：参数类型（位置参数、默认参数、关键字参数）
print("\n=== 参数详解 ===")

def calculate(a, b, operation="add"):
    """
    计算器函数
    a, b: 位置参数（必须传）
    operation: 默认参数（不传就用默认值 "add"）
    """

    if operation == "add":
        return a + b
    elif operation == "sub":
        return a - b
    elif operation == "mul":
        return a * b
    elif operation == "div":
        if b == 0:
            return "错误：除数不能为0"
        return a / b
    else:
        return "未知操作"


print(calculate(10, 5))
print(calculate(10, 5, "mul"))
print(calculate(10, 5, operation="sub"))

# 练习 3：返回值（可以返回多个值，实际上是元组）
print("\n=== 多返回值 ===")


def get_stats(numbers):
    """返回列表的最大值、最小值、平均值"""
    if not numbers:
        return None, None, None

    max_val = max(numbers)
    min_val = min(numbers)
    avg_val = sum(numbers) / len(numbers)

    return max_val, min_val, avg_val  # 实际上是返回 (max_val, min_val, avg_val)

# 解包接收
scores = [85, 92, 78, 90, 88]
maximum, minimum, average = get_stats(scores)
print(f"最高：{maximum}，最低：{minimum}，平均：{average:.2f}")

print("\n=== 作用域 ===")

counter = 0  # 全局变量

def increment():
    global counter  # 声明使用全局变量（否则 Python 会认为是局部变量）
    counter += 1
    print(f"计数器：{counter}")

increment()
increment()
print(f"全局 counter：{counter}")

# 注意：尽量避免使用 global，推荐用参数和返回值传递数据
def increment_safe(count):
    """更安全的写法：通过参数和返回值"""
    return count + 1

counter = increment_safe(counter)
print(f"安全写法后的 counter：{counter}")