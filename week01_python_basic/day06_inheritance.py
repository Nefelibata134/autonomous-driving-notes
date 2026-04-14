# 练习 1：继承基础
print("=== 继承：子类复用父类 ===")


class Vehicle:
    """交通工具基类"""

    def __init__(self, brand, speed):
        self.brand = brand
        self.speed = speed
        self.odometer = 0  # 里程表

    def drive(self, distance):
        """行驶"""
        self.odometer += distance
        print(f"{self.brand} 行驶了 {distance} km，总里程 {self.odometer} km")

    def honk(self):
        """鸣笛（默认实现）"""
        print(f"{self.brand} 发出了鸣笛声")


class Car(Vehicle):
    """小汽车，继承 Vehicle"""

    def __init__(self, brand, speed, seats):
        super().__init__(brand, speed)  # 调用父类构造方法（必须！）
        self.seats = seats  # 新增属性

    def honk(self):
        """重写父类方法（多态）"""
        print(f"{self.brand} 汽车：滴滴！")

    def drift(self):
        """子类特有方法"""
        print(f"{self.brand} 正在漂移！🏎️")


class Truck(Vehicle):
    """卡车，继承 Vehicle"""

    def __init__(self, brand, speed, load_capacity):
        super().__init__(brand, speed)
        self.load_capacity = load_capacity  # 载重

    def honk(self):
        """重写"""
        print(f"{self.brand} 卡车：叭叭叭！！🚚")

    def load_cargo(self, weight):
        """子类特有"""
        if weight <= self.load_capacity:
            print(f"装载 {weight} 吨货物成功")
        else:
            print(f"超载！最大载重 {self.load_capacity} 吨")


# 使用
print("--- 汽车 ---")
my_car = Car("宝马", 120, 5)
my_car.drive(100)  # 继承自父类
my_car.honk()  # 调用重写后的方法
my_car.drift()  # 子类特有

print("\n--- 卡车 ---")
my_truck = Truck("东风", 80, 10)
my_truck.drive(200)  # 继承自父类
my_truck.honk()  # 调用重写后的方法
my_truck.load_cargo(8)

# 练习 2：isinstance 类型检查（自动驾驶中常用）
print("\n=== 类型检查 ===")
vehicles = [Car("奥迪", 100, 4), Truck("解放", 60, 15), Car("奔驰", 110, 4)]

for v in vehicles:
    if isinstance(v, Car):
        print(f"{v.brand} 是汽车，座位数：{v.seats}")
    elif isinstance(v, Truck):
        print(f"{v.brand} 是卡车，载重：{v.load_capacity}吨")

# 练习 3：多态的威力（统一接口，不同实现）
print("\n=== 多态：统一调用，不同表现 ===")


def test_drive(vehicle):
    """测试任何交通工具（不关心具体类型）"""
    vehicle.drive(50)
    vehicle.honk()


test_drive(my_car)
test_drive(my_truck)