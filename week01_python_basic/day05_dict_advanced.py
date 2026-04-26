# 练习 1：基础回顾与新增方法
print("=== 字典基础 ===")
student = {
    "name": "张三",
    "age": 20,
    "scores": [85, 92, 78]  # 值可以是列表！
}

# 访问
print(student["name"])       # 张三
print(student.get("phone", "无电话"))  # 安全获取，不存在返回默认值

# 新增/修改
student["city"] = "北京"
student["age"] = 21

# 删除
del student["city"]
phone = student.pop("phone", None)  # 弹出并返回

# 练习 2：遍历字典（高频考点）
print("\n=== 字典遍历 ===")
for key in student:  # 默认遍历键
    print(f"{key}: {student[key]}")

for key, value in student.items():  # 同时遍历键值（最常用）
    print(f"{key} = {value}")

# 练习 3：嵌套字典（重要！表示复杂结构）
print("\n=== 嵌套字典：班级学生表 ===")
class_data = {
    "class_name": "自动驾驶1班",
    "students": {
        "2024001": {"name": "张三", "phone": "13800138000", "city": "北京"},
        "2024002": {"name": "李四", "phone": "13900139000", "city": "上海"},
        "2024003": {"name": "王五", "phone": "13700137000", "city": "广州"}
    },
    "teacher": "李老师"
}

# 过滤字典（筛选城市为北京的学生）
beijing_students = {
    k: v for k, v in class_data["students"].items()
    if v["city"] == "北京"
}
print(f"北京学生：{beijing_students}")

# 练习 5：合并字典（Python 3.9+ 语法）
print("\n=== 字典合并 ===")
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}  # b 重复
merged = dict1 | dict2  # dict2 覆盖 dict1 的重复键
print(merged)  # {'a': 1, 'b': 3, 'c': 4}

