# 练习 1：数数到 10（对比 for 和 while）
print("=== 方法 1：for 循环 ===")
for i in range(1, 11):
    print(i, end=" ")
print()  # 换行

print("=== 方法 2：while 循环 ===")
count = 1
while count <= 10:
    print(count, end=" ")
    count += 1  # 必须手动加 1，否则死循环！
print()

# 练习 2：用户输入控制循环（q 退出）
print("\n=== 输入控制循环 ===")
while True:
    msg = input("请输入内容（输入 q 退出）：")
    if msg == "q":
        print("检测到 q，准备退出...")
        break  # 跳出 while
    print(f"你输入了：{msg}")

print("循环已退出，程序结束")

# 练习 3：continue 跳过（只打印奇数）
print("\n=== continue 练习：只打印奇数 ===")
num = 0
while num < 10:
    num += 1
    if num % 2 == 0:
        continue
    print(num, end=" ")
