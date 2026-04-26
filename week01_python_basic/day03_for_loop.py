# 练习 1：基础 range 循环（对比 while）
print("=== for vs while 写法对比 ===")

# while 写法（需要手动维护计数器）
count = 0
while count < 5:
    print (count,end=" ")
    count += 1
print()

# for 写法（自动迭代，更简洁）
for i in range(5):     #0,1,2,3,4
    print (i,end=" ")
print()

# 练习 2：range 的三种形式
print("\n=== range 详解 ===")
print("rang(5):",list(range(5)))
print("range(1,6):", list(range(1, 6)))   # [1,2,3,4,5]  含头不含尾
print("range(0,10,2):", list(range(0, 10, 2)))  # [0,2,4,6,8]  步长为2

# 练习 3：遍历字符串和列表（最常用）
print("\n=== 遍历列表 ===")
students = ["张三", "李四", "王五"]
for name in students:
    print(f"欢迎，{name}！")

# 练习 4：enumerate 获取索引（重要！）
print("\n=== enumerate 获取序号 ===")
for index, name in enumerate(students):
    print(f"第{index+1}位：{name}")

# 练习 5：zip 并行遍历（多列表同时迭代）
print("\n=== zip 并行遍历 ===")
names = ["张三", "李四", "王五"]
scores = [90, 80, 70]

for name, score in zip(names, scores):
    print(f"{name}:{score}分")

print("\n=== for 中的 break ===")
for i in range(10):
    if i == 5:
        break  # 遇到 5 就停止
    print(i, end=" ")  # 输出 0 1 2 3 4