name="张三"
age=25
height=1.75
is_student=False

print(type(name))      # <class 'str'>
print(type(age))       # <class 'int'>
print(type(height))    # <class 'float'>
print(type(is_student)) # <class 'bool'>

# 2. 字符串拼接（错误示范 vs 正确示范）
# 错误：print("年龄：" + age)  # 字符串不能加整数
# 正确：
print("年龄：" + str(age))
print(f"姓名：{name}，年龄：{age}岁")  # f-string 推荐写法

# 3. 类型转换练习
user_input = "18"  # 模拟用户输入
number = int(user_input)
print(number+2)

#day01_calculator.py

# 输入函数（注意：input返回的是字符串！）
a = input("请输入第一个数字：")
b = input("请输入第二个数字：")

a=float(a)
b=float(b)

print(f"加法:{a+b}")
print(f"减法：{a - b}")
print(f"乘法：{a * b}")
print(f"除法：{a / b}")
print(f"整除：{a // b}")  # 如 7//2 = 3
print(f"取余：{a % b}")   # 如 7%2 = 1
print(f"幂运算：{a ** b}") # 如 2**3 = 8

print(f"a > b 吗？{a > b}")
print(f"a 等于 b 吗？{a == b}")

is_positive = (a>0) and (b>0)
print(f"两个都是正数吗？ {is_positive}")