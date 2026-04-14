# day06_address_book_oop.py

import json
import csv
import os


class Contact:
    """单个联系人"""

    def __init__(self, cid, name, phone, email=None, city="未知"):
        self.cid = cid  # 联系人ID
        self.name = name
        self.phone = phone
        self.email = email
        self.city = city

    def update(self, **kwargs):
        """更新信息（关键字参数）"""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        print(f"已更新 {self.name} 的信息")

    def to_dict(self):
        """转为字典（用于 JSON 序列化）"""
        return {
            "cid": self.cid,
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "city": self.city
        }

    @classmethod
    def from_dict(cls, data):
        """类方法：从字典创建对象（替代构造方法）"""
        return cls(
            cid=data["cid"],
            name=data["name"],
            phone=data["phone"],
            email=data.get("email"),
            city=data.get("city", "未知")
        )

    def __str__(self):
        return f"[{self.cid}] {self.name} | {self.phone} | {self.city}"


class AddressBook:
    """通讯录管理类"""

    DATA_FILE = "address_book.json"

    def __init__(self):
        self.contacts = {}  # cid -> Contact 对象
        self.load()

    def add(self, cid, name, phone, email=None, city="未知"):
        """添加联系人"""
        if cid in self.contacts:
            print(f"错误：ID {cid} 已存在")
            return False

        contact = Contact(cid, name, phone, email, city)
        self.contacts[cid] = contact
        print(f"添加成功：{contact}")
        return True

    def delete(self, cid):
        """删除"""
        if cid in self.contacts:
            name = self.contacts[cid].name
            del self.contacts[cid]
            print(f"已删除 {name}")
            return True
        print("未找到该联系人")
        return False

    def search(self, keyword):
        """模糊搜索（在 name/phone/city 中搜索）"""
        results = []
        keyword = keyword.lower()

        for contact in self.contacts.values():
            text = f"{contact.cid} {contact.name} {contact.phone} {contact.city}".lower()
            if keyword in text:
                results.append(contact)

        return results

    def update_contact(self, cid, **kwargs):
        """更新联系人信息"""
        if cid not in self.contacts:
            print("联系人不存在")
            return False
        self.contacts[cid].update(**kwargs)
        return True

    def export_csv(self, filename):
        """导出 CSV"""
        if not self.contacts:
            print("通讯录为空")
            return

        with open(filename, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["cid", "name", "phone", "email", "city"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for contact in self.contacts.values():
                writer.writerow(contact.to_dict())

        print(f"已导出 {len(self.contacts)} 条记录到 {filename}")

    def import_csv(self, filename):
        """导入 CSV"""
        if not os.path.exists(filename):
            print("文件不存在")
            return

        count = 0
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("cid") or f"imported_{count}"
                self.add(
                    cid=cid,
                    name=row["name"],
                    phone=row["phone"],
                    email=row.get("email"),
                    city=row.get("city", "未知")
                )
                count += 1

        print(f"成功导入 {count} 条记录")

    def save(self):
        """保存到 JSON"""
        data = {cid: c.to_dict() for cid, c in self.contacts.items()}
        with open(self.DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(self.contacts)} 条记录")

    def load(self):
        """从 JSON 加载"""
        if not os.path.exists(self.DATA_FILE):
            return

        with open(self.DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            for cid, cdata in data.items():
                self.contacts[cid] = Contact.from_dict(cdata)

        print(f"已加载 {len(self.contacts)} 条记录")

    def list_all(self):
        """显示所有"""
        if not self.contacts:
            print("通讯录为空")
            return

        print(f"\n{'ID':<12}{'姓名':<10}{'电话':<15}{'城市':<10}")
        print("-" * 50)
        for contact in self.contacts.values():
            print(f"{contact.cid:<12}{contact.name:<10}{contact.phone:<15}{contact.city:<10}")

    def __len__(self):
        """魔法方法：len(book) 返回联系人数"""
        return len(self.contacts)


# 主程序（交互式）
def main():
    book = AddressBook()  # 自动加载

    while True:
        print("\n=== OOP 通讯录管理系统 ===")
        print("1. 添加联系人")
        print("2. 删除联系人")
        print("3. 搜索联系人")
        print("4. 更新联系人")
        print("5. 显示所有")
        print("6. 导出 CSV")
        print("7. 导入 CSV")
        print("8. 保存并退出")
        print(f"当前共 {len(book)} 位联系人")

        choice = input("选择：")

        if choice == "1":
            cid = input("ID：")
            name = input("姓名：")
            phone = input("电话：")
            email = input("邮箱（可选）：") or None
            city = input("城市（可选）：") or "未知"
            book.add(cid, name, phone, email, city)

        elif choice == "2":
            cid = input("要删除的ID：")
            book.delete(cid)

        elif choice == "3":
            keyword = input("搜索关键词：")
            results = book.search(keyword)
            if results:
                print(f"找到 {len(results)} 条结果：")
                for c in results:
                    print(f"  {c}")
            else:
                print("未找到")

        elif choice == "4":
            cid = input("要更新的ID：")
            if cid in book.contacts:
                print("输入新值（直接回车保持不变）：")
                name = input(f"姓名[{book.contacts[cid].name}]：") or None
                phone = input(f"电话[{book.contacts[cid].phone}]：") or None
                book.update_contact(cid, name=name, phone=phone)

        elif choice == "5":
            book.list_all()

        elif choice == "6":
            filename = input("文件名（默认 export.csv）：") or "export.csv"
            book.export_csv(filename)

        elif choice == "7":
            filename = input("CSV 文件名：")
            book.import_csv(filename)

        elif choice == "8":
            book.save()
            print("再见！")
            break

        else:
            print("无效选择")


if __name__ == "__main__":
    main()