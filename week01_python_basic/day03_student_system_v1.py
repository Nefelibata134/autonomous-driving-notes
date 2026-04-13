# day03_student_system_v1.py

def show_menu():
    print("\n=== 学生成绩管理系统 ===")
    print("1. 添加学生")
    print("2. 删除学生")
    print("3. 查询学生")
    print("4. 显示所有")
    print("5. 统计信息")
    print("6. 退出")


def add_student(students):
    name = input("姓名：")
    try:
        chinese = int(input("语文成绩："))
        math = int(input("数学成绩："))
        english = int(input("英语成绩："))

        # 数据验证
        if not (0 <= chinese <= 100 and 0 <= math <= 100 and 0 <= english <= 100):
            print("成绩必须在 0-100 之间！")
            return

        # 检查是否已存在
        for s in students:
            if s[0] == name:
                print("该学生已存在！")
                return

        students.append([name, chinese, math, english])
        print(f"添加成功！已添加 {len(students)} 名学生")
    except ValueError:
        print("请输入数字！")


def delete_student(students):
    name = input("要删除的姓名：")
    for i, s in enumerate(students):
        if s[0] == name:
            students.pop(i)
            print(f"已删除 {name}")
            return
    print("未找到该学生")


def query_student(students):
    name = input("要查询的姓名：")
    for s in students:
        if s[0] == name:
            total = s[1] + s[2] + s[3]
            avg = total / 3
            print(f"\n{s[0]} 的成绩：")
            print(f"  语文：{s[1]}")
            print(f"  数学：{s[2]}")
            print(f"  英语：{s[3]}")
            print(f"  总分：{total}，平均分：{avg:.1f}")
            return
    print("未找到该学生")


def show_all(students):
    if not students:
        print("暂无学生")
        return

    print("\n所有学生成绩：")
    print(f"{'姓名':<10}{'语文':<6}{'数学':<6}{'英语':<6}{'总分':<6}{'平均':<6}")
    print("-" * 40)

    for s in students:
        total = s[1] + s[2] + s[3]
        avg = total / 3
        print(f"{s[0]:<10}{s[1]:<6}{s[2]:<6}{s[3]:<6}{total:<6}{avg:<6.1f}")


def show_statistics(students):
    if not students:
        print("暂无数据")
        return

    # 提取各科成绩列表（列表推导式）
    chinese_scores = [s[1] for s in students]
    math_scores = [s[2] for s in students]
    english_scores = [s[3] for s in students]

    print("\n班级统计：")
    print(f"学生人数：{len(students)}")
    print(
        f"语文：最高{max(chinese_scores)}, 最低{min(chinese_scores)}, 平均{sum(chinese_scores) / len(chinese_scores):.1f}")
    print(f"数学：最高{max(math_scores)}, 最低{min(math_scores)}, 平均{sum(math_scores) / len(math_scores):.1f}")
    print(
        f"英语：最高{max(english_scores)}, 最低{min(english_scores)}, 平均{sum(english_scores) / len(english_scores):.1f}")


def main():
    students = []  # 内存存储

    while True:
        show_menu()
        choice = input("请选择 (1-6)：")

        if choice == "1":
            add_student(students)
        elif choice == "2":
            delete_student(students)
        elif choice == "3":
            query_student(students)
        elif choice == "4":
            show_all(students)
        elif choice == "5":
            show_statistics(students)
        elif choice == "6":
            print("再见！")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()