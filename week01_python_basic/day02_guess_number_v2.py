# day02_guess_number_v2.py
import random
import os


def load_best_score():
    """从文件读取历史最佳成绩"""
    if os.path.exists("best_score.txt"):
        with open("best_score.txt", "r") as f:
            return int(f.read().strip())
    return 999  # 默认很高，表示还没记录


def save_best_score(score):
    """保存最佳成绩到文件"""
    with open("best_score.txt", "w") as f:
        f.write(str(score))


# 游戏开始
print("=== 猜数字游戏 2.0 ===")
print("1. 简单 (1-50)")
print("2. 中等 (1-100)")
print("3. 困难 (1-1000)")

choice = input("请选择难度 (1/2/3)：")
if choice == "1":
    max_num = 50
elif choice == "3":
    max_num = 1000
else:
    max_num = 100

secret = random.randint(1, max_num)
attempts = 0
best_score = load_best_score()

print(f"\n游戏开始！范围是 1-{max_num}，历史最佳成绩是 {best_score} 次")

while True:
    guess = input("你的猜测：")

    try:
        guess = int(guess)
    except ValueError:
        print("请输入有效数字！")
        continue

    if guess < 1 or guess > max_num:
        print(f"超出范围！请输入 1-{max_num} 之间的数字")
        continue

    attempts += 1

    if guess < secret:
        print("📉 太小了")
    elif guess > secret:
        print("📈 太大了")
    else:
        print(f"\n🎯 猜中了！答案就是 {secret}！")
        print(f"你用了 {attempts} 次")

        if attempts < best_score:
            print(f"🏆 新纪录！打破了之前的 {best_score} 次")
            save_best_score(attempts)
        else:
            print(f"最佳纪录是 {best_score} 次，继续加油！")

        # 询问是否再来一局
        again = input("\n再玩一局吗？(y/n)：")
        if again.lower() == "y":
            secret = random.randint(1, max_num)
            attempts = 0
            print("\n=== 新游戏开始 ===")
            continue
        else:
            print("再见！")
            break
