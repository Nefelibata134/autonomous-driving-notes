# utils.py - 文本处理工具模块

def count_words(text):
    """统计词频，返回字典"""
    # 清理标点，简单分词（按空格分割）
    words = text.lower().replace(",", " ").replace(".", " ").replace("!", " ").split()

    word_count = {}
    for word in words:
        if word:  # 跳过空字符串
            word_count[word] = word_count.get(word, 0) + 1

    # 按频率排序（从高到低）
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_words


def filter_sensitive(text, sensitive_words):
    """敏感词过滤"""
    for word in sensitive_words:
        text = text.replace(word, "*" * len(word))
    return text


def format_with_line_number(text):
    """添加行号"""
    lines = text.split("\n")
    formatted = []
    for i, line in enumerate(lines, 1):
        formatted.append(f"{i:03d}: {line}")
    return "\n".join(formatted)


def save_result(text, filename):
    """保存结果"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"保存失败：{e}")
        return False