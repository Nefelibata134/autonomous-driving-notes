# 练习 1：定义第一个类
print("=== 类与对象基础 ===")

class Student:
    """学生类（类名首字母大写是惯例）"""

    def __init__(self, name, age, grade):
        """
        构造方法：创建对象时自动调用
        self 代表"这个对象自己"，必须有！
        """
        self.name = name  # 实例属性（每个对象独立）
        self.age = age
        self.grade = grade
        self.scores = []  # 默认空列表

    def study(self,hours):
        print(f"{self.name} 学习了 {hours} 小时")
        self.grade += hours * 0.5

    def add_score(self, subject, score):
        """添加成绩"""
        self.scores.append({"subject": subject, "score": score})

    def get_average(self):
        """计算平均分"""
        if not self.scores:
            return 0
        total = sum(a["score"] for a in self.scores)
        return total / len(self.scores)

    def __str__(self):
        """魔法方法：print(对象) 时自动调用"""
        return f"学生[{self.name}, {self.age}岁, 成绩{self.grade:.1f}]"

# 创建对象（实例化）
stu1 = Student("张三", 20, 80.0)  # 自动调用 __init__
stu2 = Student("李四", 21, 85.0)

print(stu1)  # 自动调用 __str__
print(stu2)

# 调用方法
stu1.study(2)
stu1.add_score("数学", 90)
stu1.add_score("英语", 85)
print(f"{stu1.name} 的平均分：{stu1.get_average():.1f}")

# 练习 2：理解 self 的本质
print("\n=== self 的本质 ===")
print(f"stu1 的 id：{id(stu1)}")
print(f"stu1.name：{stu1.name}")  # 等价于 Student.name(stu1) 但不这么写

# 练习 3：类属性 vs 实例属性
print("\n=== 类属性（共享）vs 实例属性（独立） ===")


class School:
    school_name = "自动驾驶学院"  # 类属性（所有对象共享）

    def __init__(self, student_name):
        self.student_name = student_name  # 实例属性（每个对象独立）

s1 = School("张三")
s2 = School("李四")

print(f"s1 学校：{s1.school_name}, 学生：{s1.student_name}")
print(f"s2 学校：{s2.school_name}, 学生：{s2.student_name}")

School.school_name = "AI 学院"
# 通过类名.属性名 = 新值 修改类属性
# 这会改变类本身的属性，因此所有实例访问 school_name 都会看到新值

print(f"修改后 s1：{s1.school_name}")    # 输出：AI 学院（s1 看到的是修改后的共享值）
print(f"修改后 s2：{s2.school_name}")    # 输出：AI 学院（s2 同样看到修改后的值）

# 修改实例属性（只影响自己）
s1.school_name = "私人学院"  # 注意：这会创建 s1 自己的 school_name，不再共享！
print(f"s1 私有：{s1.school_name}")
print(f"s2 仍共享：{s2.school_name}")