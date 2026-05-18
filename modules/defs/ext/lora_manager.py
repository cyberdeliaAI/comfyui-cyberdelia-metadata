# loraManager
# https://github.com/willmiao/ComfyUI-Lora-Manager

import json
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_lora_hash, convert_skip_clip, calc_unet_hash


def get_lora_model_name_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["loras"]

    if toggled_on:
        lora_names = []
        for loras in input_data[0]["loras"][0]["__value__"]:
            lora_str = loras["name"]
            if lora_str == "":
                continue
            lora_names.append(lora_str)
        return lora_names
    else:
        return []

def get_lora_strength_model_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["loras"]

    if toggled_on:
        lora_names = []
        for loras in input_data[0]["loras"][0]["__value__"]:
            lora_str = loras["strength"]
            if lora_str == "":
                continue
            lora_names.append(lora_str)
        return lora_names
    else:
        return []

def get_lora_strength_clip_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["loras"]

    if toggled_on:
        lora_names = []
        for loras in input_data[0]["loras"][0]["__value__"]:
            lora_str = loras["clipStrength"]
            if lora_str == "":
                continue
            lora_names.append(lora_str)
        return lora_names
    else:
        return []

def get_lora_model_hash_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["loras"]

    if toggled_on:
        lora_names = []
        for loras in input_data[0]["loras"][0]["__value__"]:
            lora_str = loras["name"]
            if lora_str == "":
                continue
            lora_names.append(calc_lora_hash(lora_str + ".safetensors"))
        return lora_names
    else:
        return []

SAMPLERS = {
}

CAPTURE_FIELD_LIST = {
    "Lora Loader (LoraManager)": {
            MetaField.LORA_MODEL_NAME: {"selector": get_lora_model_name_stack},
            MetaField.LORA_MODEL_HASH: {"selector": get_lora_model_hash_stack},
            MetaField.LORA_STRENGTH_MODEL: {"selector": get_lora_strength_model_stack},
            MetaField.LORA_STRENGTH_CLIP: {"selector": get_lora_strength_clip_stack},
    }
}