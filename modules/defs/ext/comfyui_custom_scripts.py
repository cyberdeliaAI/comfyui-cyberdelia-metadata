# https://github.com/pythongosssss/ComfyUI-Custom-Scripts
from ..meta import MetaField
from ..formatters import calc_lora_hash, calc_model_hash


def get_lora_model_name_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    return get_lora_data_stack(input_data, "lora")


def get_lora_model_hash_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    return [
        calc_lora_hash(model_name, input_data)
        for model_name in get_lora_data_stack(input_data, "lora")
    ]


def get_lora_strength_model_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    raw = get_lora_data_stack(input_data, "strength")
    return format_strengths(raw)


def get_lora_strength_clip_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    raw = get_lora_data_stack(input_data, "strength")
    return format_strengths(raw)


def get_lora_data_stack(input_data, attribute):
    return [
        v[0]
        for k, v in input_data[0].items()
        if k.startswith(attribute + "_") and v[0] not in ("None", None, "")
    ]


def format_strengths(strengths):
    formatted = []
    for s in strengths:
        try:
            val = float(s)
            val_str = f"{val:.2f}"
            formatted.append(val_str)
        except Exception as e:
            formatted.append("0.00")
    return formatted


CAPTURE_FIELD_LIST = {
    "LoraLoader|pysssss": {
        MetaField.LORA_MODEL_NAME: {"selector": get_lora_model_name_stack},
        MetaField.LORA_MODEL_HASH: {"selector": get_lora_model_hash_stack},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_lora_strength_model_stack},
        MetaField.LORA_STRENGTH_CLIP: {"selector": get_lora_strength_clip_stack},
    },
    "CheckpointLoader|pysssss": {
        MetaField.MODEL_NAME: {"field_name": "ckpt_name"},
        MetaField.MODEL_HASH: {"field_name": "ckpt_name", "format": calc_model_hash},
    },
}