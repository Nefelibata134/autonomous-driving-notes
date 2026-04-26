import json
import csv
import os

# 练习 1：JSON 读写（自动驾驶中最常用！）
print("=== JSON 操作 ===")
data = {
    "dataset": "BDD100K",
    "classes": ["car", "person", "traffic light"],
    "num_images": 100000,
    "meta": {
        "author": "Nefelibata",
        "date": "2026-04-13"
    }
}

# 写入 JSON（indent 美化格式）
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("已保存 config.json")

# 读取 JSON
with open("config.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)
print(f"读取到数据集：{loaded['dataset']}, 类别数：{len(loaded['classes'])}")

# 练习 2：CSV 读写（表格数据）
print("\n=== CSV 操作 ===")
students = [
    ["学号", "姓名", "电话", "城市"],  # 表头
    ["2024001", "张三", "13800138000", "北京"],
    ["2024002", "李四", "13900139000", "上海"],
    ["2024003", "王五", "13700137000", "广州"]
]

# 写入 CSV
with open("students.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(students)  # 写入多行
print("已保存 students.csv（可用 Excel 打开）")

# 读取 CSV
print("\n读取 CSV 内容：")
with open("students.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            print(f"表头：{row}")
        else:
            print(f"第{i}行数据：{row}")

# 练习 3：CSV 字典方式读写（更结构化）
print("\n=== CSV DictReader/DictWriter ===")
# 写入（用字典代替列表，更清晰）
fieldnames = ["id", "name", "score"]
with open("scores.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"id": "S001", "name": "张三", "score": "95"})
    writer.writerow({"id": "S002", "name": "李四", "score": "88"})

# 读取（每行自动转成字典）
with open("scores.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"学生：{row['name']}, 成绩：{row['score']}")
