# 练习 1：导入模块的多种方式
print("=== 模块导入 ===")

# 方式 1：导入整个模块（推荐，避免命名冲突）
import random
print(random.randint(1, 10))

# 方式 2：从模块导入特定函数（代码更简洁）
from datetime import datetime, timedelta
now = datetime.now()
print(f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}")

# 方式 3：导入并重命名（简写）
import json as js
data = js.dumps({"name": "张三"})
print(data)

# 练习 2：使用标准库常用模块
print("\n=== 标准库实践 ===")

import os
print(f"当前工作目录：{os.getcwd()}")
print(f"文件是否存在：{os.path.exists('day04_functions.py')}")

import sys
print(f"Python 版本：{sys.version_info}")

# 练习 3：异常处理（让程序不崩溃）
print("\n=== 异常处理 ===")

def safe_divide(a, b):
    """安全的除法"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return "错误：除数不能为0"
    except TypeError:
        return "错误：请输入数字"
    except Exception as e:
        return f"未知错误：{e}"

# 测试
print(safe_divide(10, 2))   # 正常
print(safe_divide(10, 0))   # 除零错误
print(safe_divide(10, "a")) # 类型错误

# 练习 4：文件操作的异常处理（最常用）
print("\n=== 文件异常处理 ===")

def read_file_safe(filename):
    """安全地读取文件"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {filename} 不存在"
    except PermissionError:
        return f"错误：没有权限读取 {filename}"
    except Exception as e:
        return f"读取失败：{e}"

# 测试
content = read_file_safe("students.json")  # 假设这个文件存在
print(f"文件内容长度：{len(content)}")

content = read_file_safe("不存在的文件.txt")
print(content)

# 练习 5：try/except/else/finally 完整结构
print("\n=== 完整异常结构 ===")

def process_number(input_str):
    try:
        num = int(input_str)  # 可能出错
    except ValueError:
        print("转换失败：不是有效数字")
        return None
    else:
        # 没有异常时执行
        print(f"转换成功：{num}")
        return num * 2
    finally:
        # 无论是否异常都执行（常用于清理资源）
        print("处理结束")

result = process_number("100")  # 成功
result = process_number("abc")  # 失败