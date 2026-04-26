# day05_address_book.py
import json
import csv
import os

DATA_FILE = "address_book.json"


def load_data():
    """加载通讯录"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_data(data):
    """保存通讯录"""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_contact(book):
    """添加联系人"""
    contact_id = input("联系人ID（如 phone_001）：")
    if contact_id in book:
        print("该ID已存在！")
        return

    name = input("姓名：")
    phone = input("电话：")
    email = input("邮箱（可选）：") or None
    city = input("城市（可选）：") or "未知"

    book[contact_id] = {
        "name": name,
        "phone": phone,
        "email": email,
        "city": city
    }
    print(f"已添加 {name}")


def search_contact(book):
    """模糊搜索（关键功能）"""
    keyword = input("搜索关键词（姓名/电话/城市）：").lower()
    results = []

    for cid, info in book.items():
        # 在多个字段中搜索
        text = f"{cid} {info['name']} {info['phone']} {info['city']}".lower()
        if keyword in text:
            results.append((cid, info))

    if not results:
        print("未找到匹配联系人")
        return

    print(f"\n找到 {len(results)} 条结果：")
    print(f"{'ID':<15}{'姓名':<10}{'电话':<15}{'城市':<10}")
    print("-" * 50)
    for cid, info in results:
        print(f"{cid:<15}{info['name']:<10}{info['phone']:<15}{info['city']:<10}")


def delete_contact(book):
    """删除"""
    cid = input("要删除的联系人ID：")
    if cid in book:
        confirm = input(f"确认删除 {book[cid]['name']}？(y/n)：")
        if confirm.lower() == "y":
            del book[cid]
            print("已删除")
    else:
        print("未找到该ID")


def export_to_csv(book):
    """导出为 CSV（方便 Excel 查看）"""
    if not book:
        print("通讯录为空")
        return

    filename = input("导出文件名（默认 address_export.csv）：") or "address_export.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "name", "phone", "email", "city"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cid, info in book.items():
            row = {"id": cid, **info}  # 合并字典
            writer.writerow(row)

    print(f"已导出 {len(book)} 条记录到 {filename}")


def import_from_csv(book):
    """从 CSV 导入"""
    filename = input("CSV 文件名：")
    if not os.path.exists(filename):
        print("文件不存在")
        return

    count = 0
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("id") or f"imported_{count}"
            book[cid] = {
                "name": row["name"],
                "phone": row["phone"],
                "email": row.get("email"),
                "city": row.get("city", "未知")
            }
            count += 1

    print(f"成功导入 {count} 条记录")


def show_menu():
    print("\n=== 通讯录管理系统 2.0 ===")
    print("1. 添加联系人")
    print("2. 搜索联系人（模糊匹配）")
    print("3. 删除联系人")
    print("4. 显示所有")
    print("5. 导出 CSV")
    print("6. 导入 CSV")
    print("7. 保存并退出")


def main():
    book = load_data()
    print(f"已加载 {len(book)} 条联系人")

    while True:
        show_menu()
        choice = input("选择：")

        if choice == "1":
            add_contact(book)
        elif choice == "2":
            search_contact(book)
        elif choice == "3":
            delete_contact(book)
        elif choice == "4":
            for cid, info in book.items():
                print(f"{cid}: {info['name']} ({info['phone']}, {info['city']})")
        elif choice == "5":
            export_to_csv(book)
        elif choice == "6":
            import_from_csv(book)
        elif choice == "7":
            save_data(book)
            print(f"已保存 {len(book)} 条记录，再见！")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()