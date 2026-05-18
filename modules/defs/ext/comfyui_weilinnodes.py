#https://github.com/Light-x02/ComfyUI-FluxSettingsNode
import json
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_lora_hash, convert_skip_clip


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


def is_positive_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    title = obj['_meta']['title']
    # Positive by default.
    if title:
        if is_positive_title(title):
            return True
        if is_negative_title(title):
            return False
    return True

def is_negative_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    title = obj['_meta']['title']
    if title:
        if is_negative_title(title):
            return True
    return False

# ref format:
# '[{"name":"cecilia_shiro seijo to kuro bokushi_IllustriousXL_last","weight":0.5,"text_encoder_weight":0.5,
# "lora":"cecilia_shiro seijo to kuro bokushi_IllustriousXL_last.safetensors","loraWorks":""}]'
def get_lora_model_name_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["lora_str"]

    if toggled_on:
        lora_names = []
        for lora_str in input_data[0]["lora_str"]:
            if lora_str == "":
                continue
            lora_data = json.loads(lora_str)
            lora_names.extend([item["name"] for item in lora_data])
        return lora_names
    else:
        return []

def get_lora_strength_model_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["lora_str"]

    if toggled_on:
        lora_names = []
        for lora_str in input_data[0]["lora_str"]:
            if lora_str == "":
                continue
            lora_data = json.loads(lora_str)
            lora_names.extend([item["weight"] for item in lora_data])
        return lora_names
    else:
        return []

def get_lora_strength_clip_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["lora_str"]

    if toggled_on:
        lora_names = []
        for lora_str in input_data[0]["lora_str"]:
            if lora_str == "":
                continue
            lora_data = json.loads(lora_str)
            lora_names.extend([item["text_encoder_weight"] for item in lora_data])
        return lora_names
    else:
        return []

def get_lora_model_hash_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["lora_str"]

    if toggled_on:
        lora_names = []
        for lora_str in input_data[0]["lora_str"]:
            if lora_str == "":
                continue
            lora_data = json.loads(lora_str)
            lora_names.extend([calc_lora_hash(item["lora"], input_data) for item in lora_data])
        return lora_names
    else:
        return []

CAPTURE_FIELD_LIST = {
    "WeiLinComfyUIPromptToLoras":
        {
            MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
            MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
        },
    "WeiLinPromptUI":
        {
            MetaField.LORA_MODEL_NAME: {"selector": get_lora_model_name_stack},
            MetaField.LORA_MODEL_HASH: {"selector": get_lora_model_hash_stack},
            MetaField.LORA_STRENGTH_MODEL: {"selector": get_lora_strength_model_stack},
            MetaField.LORA_STRENGTH_CLIP: {"selector": get_lora_strength_clip_stack},
            # by @Aaalice
            MetaField.POSITIVE_PROMPT: { "field_name": "positive",
                                         "validate": is_positive_prompt},
            MetaField.NEGATIVE_PROMPT: { "field_name": "positive",
                                         "validate": is_negative_prompt},
        },
}
