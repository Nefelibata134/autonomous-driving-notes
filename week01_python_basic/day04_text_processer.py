# day04_text_processor.py
import os
from datetime import datetime
from utils import count_words, filter_sensitive, format_with_line_number, save_result


def show_menu():
    print("\n=== 文本处理工具箱 ===")
    print("1. 统计词频")
    print("2. 敏感词过滤")
    print("3. 添加行号")
    print("4. 综合处理（全部功能）")
    print("5. 退出")


def load_text():
    """加载文本文件"""
    filename = input("请输入文件名：")
    if not os.path.exists(filename):
        print("文件不存在！")
        return None
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"读取失败：{e}")
        return None


def main():
    sensitive_words = ["暴力", "色情", "赌博", "脏话"]  # 示例敏感词

    while True:
        show_menu()
        choice = input("请选择：")

        if choice == "1":
            text = load_text()
            if text:
                words = count_words(text)
                print("\n词频统计 Top 10：")
                for word, count in words[:10]:
                    print(f"  {word}: {count}次")

        elif choice == "2":
            text = load_text()
            if text:
                filtered = filter_sensitive(text, sensitive_words)
                print("\n过滤后内容（前200字符）：")
                print(filtered[:200] + "...")
                save_result(filtered, "filtered_output.txt")
                print("已保存到 filtered_output.txt")

        elif choice == "3":
            text = load_text()
            if text:
                numbered = format_with_line_number(text)
                print("\n添加行号后（前5行）：")
                print("\n".join(numbered.split("\n")[:5]))
                save_result(numbered, "numbered_output.txt")
                print("已保存到 numbered_output.txt")

        elif choice == "4":
            text = load_text()
            if text:
                # 流水线处理：过滤 → 加行号 → 统计
                filtered = filter_sensitive(text, sensitive_words)
                numbered = format_with_line_number(filtered)
                words = count_words(filtered)

                # 生成报告
                report = f"""处理时间：{datetime.now()}
原始长度：{len(text)} 字符
过滤后长度：{len(filtered)} 字符
总行数：{len(numbered.split(chr(10)))}
高频词：{words[0][0]}({words[0][1]}次)

处理后的文本：
{numbered[:500]}...
"""
                save_result(report, "final_report.txt")
                print("综合处理完成，已保存到 final_report.txt")

        elif choice == "5":
            print("再见！")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()