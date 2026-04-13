# 练习 1：创建和访问
print("=== 列表基础 ===")
scores = [85, 92, 78, 90, 88]  # 一个班的成绩

print(f"第1个成绩：{scores[0]}")   # 索引从0开始
print(f"最后1个成绩：{scores[-1]}") # 负数倒数
print(f"前3个成绩：{scores[0:3]}") # 切片 [0,3) 含头不含尾

# 练习 2：增删改（列表可变）
print("\n=== 增删改 ===")
scores.append(95)       # 末尾添加
print(f"添加后：{scores}")

scores.insert(2, 100)   # 在索引2位置插入
print(f"插入后：{scores}")

scores.remove(100)      # 删除第一个值为100的元素
print(f"删除后：{scores}")

popped = scores.pop()   # 弹出最后一个（并返回）
print(f"弹出：{popped}，剩余：{scores}")

scores[0] = 88          # 修改
print(f"修改后：{scores}")

# 练习 3：列表方法（统计和查找）
print("\n=== 统计方法 ===")
print(f"长度：{len(scores)}")
print(f"最大值：{max(scores)}")
print(f"最小值：{min(scores)}")
print(f"总和：{sum(scores)}")
print(f"平均分：{sum(scores)/len(scores):.2f}")

print(f"90分出现次数：{scores.count(90)}")
print(f"90分首次出现位置：{scores.index(90)}")

# 练习 4：列表排序
print("\n=== 排序 ===")
scores.sort()           # 升序（修改原列表）
print(f"升序：{scores}")

scores.sort(reverse=True)  # 降序
print(f"降序：{scores}")

# 练习 5：列表推导式（重点！一行代码生成列表）
print("\n=== 列表推导式 ===")
# 传统写法
squares = []
for x in range(1, 6):
    squares.append(x**2)
print(f"传统写法：{squares}")

# 推导式写法（推荐）
squares_new = [x**2 for x in range(1, 6)]
print(f"推导式：{squares_new}")

# 带条件的推导式（筛选）
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"偶数平方：{even_squares}")

# 练习 6：嵌套列表（二维数组，类似表格）
print("\n=== 二维列表（学生成绩表） ===")
# 每个学生有 [姓名, 语文, 数学, 英语]
students = [
    ["张三", 85, 92, 78],
    ["李四", 90, 88, 85],
    ["王五", 78, 95, 92]
]

for student in students:
    name = student[0]
    chinese = student[1]
    math = student[2]
    english = student[3]
    total = chinese + math + english
    print(f"{name}: 总分{total}, 平均分{total/len(student):.1f}")