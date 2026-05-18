# Prompts Everywhere


from ..meta import MetaField
from ..formatters import calc_model_hash, calc_lora_hash, convert_skip_clip, calc_unet_hash


POSITIVE_KEYWORDS = [
    # 中文
    "正面", "积极", "正向",
    # English
    "positive", "positives", "front", "frontal",
    # French
    "positif", "positive", "positifs", "positives",
    # German
    "positiv",
    # Spanish / Portuguese / Italian
    "positivo", "positiva", "positivos", "positivas",
    # Russian
    "позитивный", "позитивная", "позитивные",
    "положительный", "положительная", "положительные",
    # Japanese
    "ポジティブ", "前面",
    # Korean
    "긍정", "전면",
    # Arabic
    "إيجابي", "إيجابية",
    # Hindi
    "सकारात्मक",
    # Indonesian / Malay
    "positif", "positifnya",
    # Thai
    "เชิงบวก", "บวก",
    # Vietnamese
    "tích cực", "tích-cực",
]

NEGATIVE_KEYWORDS = [
    # 中文
    "负面", "消极", "否定", "负向", "负面的",
    # English
    "negative", "negatives", "bad", "adverse", "unfavorable",
    # French
    "négatif", "négative", "négatifs", "négatives", "défavorable",
    # German
    "negativ", "schlecht", "nachteilig",
    # Spanish / Portuguese / Italian
    "negativo", "negativa", "negativos", "negativas", "desfavorable", "desfavorável",
    "negative",  # Italian plural feminine
    # Russian
    "негативный", "негативная", "негативные",
    "отрицательный", "отрицательная", "отрицательные",
    "плохой",
    # Japanese
    "ネガティブ", "否定的", "マイナス", "悪い",
    # Korean
    "부정", "부정적", "마이너스", "나쁜",
    # Arabic
    "سلبي", "سلبية", "غير موات", "سيئ",
    # Hindi
    "नकारात्मक", "बुरा",
    # Indonesian / Malay
    "negatif", "buruk", "kurang baik",
    # Thai
    "เชิงลบ", "ลบ", "ไม่ดี",
    # Vietnamese
    "tiêu cực", "tiêu-cực", "xấu",
]

def is_positive_title(title: str) -> bool:
    # 1 先做 Unicode 折叠，适配土耳其等奇怪大小写
    hay = title.casefold()
    # 2 纯遍历 O(n*k)；若词库巨大可考虑 Trie/AC 自动机
    for kw in POSITIVE_KEYWORDS:
        if kw.casefold() in hay:
            return True
    return False
def is_negative_title(title: str) -> bool:
    hay = title.casefold()
    return any(kw.casefold() in hay for kw in NEGATIVE_KEYWORDS)


def is_positive_prompt_everywhere(node_id, obj, prompt, extra_data, outputs, input_data_all):
    title = obj['_meta']['title']
    if title:
        if is_positive_title(title):
            return True
    return False

def is_negative_prompt_everywhere(node_id, obj, prompt, extra_data, outputs, input_data_all):
    title = obj['_meta']['title']
    if title:
        if is_negative_title(title):
            return True
    return False

SAMPLERS = {
}

CAPTURE_FIELD_LIST = {
    "ShowText|pysssss": {
        MetaField.POSITIVE_PROMPT: {
            "field_name": "text",
            "validate": is_positive_prompt_everywhere
        },
        MetaField.NEGATIVE_PROMPT: {
            "field_name": "text",
            "validate": is_negative_prompt_everywhere
        },
    }
}