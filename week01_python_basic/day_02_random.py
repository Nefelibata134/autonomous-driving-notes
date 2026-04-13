import random

# 练习 1：掷骰子
print("=== 掷骰子 ===")
dice = random.randint(1, 6)
print(f"掷出了：{dice}")

# 练习 2：猜硬币（随机 True/False）
print("\n=== 猜硬币 ===")
coin = random.choice(["正面", "反面"])
guess = input("猜硬币（正面/反面）：")
if guess == coin:
    print(f"结果是{coin}，猜对了！")
else:
    print(f"结果是{coin}，猜错了！")

# 练习 3：随机抽奖（从列表选）
print("\n=== 今日幸运数字 ===")
lucky_numbers = [7, 14, 23, 42, 88]
today_lucky = random.choice(lucky_numbers)
print(f"今天的幸运数字是：{today_lucky}")

# 练习 4：打乱列表（洗牌）
cards = ["红桃A", "黑桃K", "方块Q", "梅花J"]
print(f"\n原顺序：{cards}")
random.shuffle(cards)  # 直接修改原列表，无返回值
print(f"打乱后：{cards}")

# 练习 5：生成随机验证码（字符串操作复习）
import string  # 字母数字库
all_chars = string.digits + string.ascii_letters  # 0-9 + a-z + A-Z
code = ""
for _ in range(6):  # 循环 6 次，_ 表示"不用这个变量"
    code += random.choice(all_chars)
print(f"\n随机验证码：{code}")