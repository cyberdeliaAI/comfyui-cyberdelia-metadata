# https://github.com/yolain/ComfyUI-Easy-Use
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_lora_hash, calc_vae_hash, convert_skip_clip, extract_embedding_hashes, extract_embedding_names
import re


def get_embedding_names(node_id, obj, prompt, extra_data, outputs, input_data):
    embedding_names = extract_embedding_names(input_data[0]["positive"][0])
    embedding_names += extract_embedding_names(input_data[0]["negative"][0])

    return embedding_names


def get_embedding_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    embedding_hashes = extract_embedding_hashes(input_data[0]["positive"][0])
    embedding_hashes += extract_embedding_hashes(input_data[0]["negative"][0])

    return embedding_hashes


def get_lora_model_names(node_id, obj, prompt, extra_data, outputs, input_data):
    lora_name = input_data[0]["lora_name"]

    lora_names = []

    # check against 'lora_name': ['None']
    if lora_name[0] != "None":
        lora_names.append(lora_name[0])

    # 'optional_lora_stack': [[('Style/Smooth_Booster_v3.safetensors', 0.3, 0.6)]]
    if "optional_lora_stack" in input_data[0]:
        optional_lora_stack = input_data[0]["optional_lora_stack"][0]
        for lora in optional_lora_stack:
            lora_names.append(lora[0])

    return lora_names


def get_lora_model_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    lora_names = get_lora_model_names(node_id, obj, prompt, extra_data, outputs, input_data)

    lora_hashes = []

    for name in lora_names:
        lora_hashes.append(calc_lora_hash(name))

    return lora_hashes


def get_lora_strengths(node_id, obj, prompt, extra_data, outputs, input_data):
    lora_strengths = []

    # check against 'lora_name': ['None']
    if input_data[0]["lora_name"][0] != "None":
        lora_strengths.append(input_data[0]["lora_model_strength"][0])

    # 'optional_lora_stack': [[('Style/Smooth_Booster_v3.safetensors', 0.3, 0.6)]]
    if "optional_lora_stack" in input_data[0]:
        optional_lora_stack = input_data[0]["optional_lora_stack"][0]
        for lora in optional_lora_stack:
            lora_strengths.append(lora[1])

    return lora_strengths


def get_lora_clip_strengths(node_id, obj, prompt, extra_data, outputs, input_data):
    lora_clip_strengths = []

    # check against 'lora_name': ['None']
    if input_data[0]["lora_name"][0] != "None":
        lora_clip_strengths.append(input_data[0]["lora_clip_strength"][0])

    # 'optional_lora_stack': [[('Style/Smooth_Booster_v3.safetensors', 0.3, 0.6)]]
    if "optional_lora_stack" in input_data[0]:
        optional_lora_stack = input_data[0]["optional_lora_stack"][0]
        for lora in optional_lora_stack:
            lora_clip_strengths.append(lora[2])

    return lora_clip_strengths


def get_lora_model_name_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["toggle"][0]
    
    if toggled_on:
        return get_lora_data_stack(input_data, "lora_\d_name")
    else:
        return []


def get_lora_model_hash_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    return [
        calc_lora_hash(model_name, input_data)
        for model_name in get_lora_data_stack(input_data, "lora_\d_name")
    ]


def get_lora_strength_model_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0]["mode"][0] == "advanced":
        return get_lora_data_stack(input_data, "lora_\d_model_strength")
    return get_lora_data_stack(input_data, "lora_\d_strength")


def get_lora_strength_clip_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0]["mode"][0] == "advanced":
        return get_lora_data_stack(input_data, "lora_\d_clip_strength")
    return get_lora_data_stack(input_data, "lora_\d_strength")


def get_lora_data_stack(input_data, attribute):
    lora_count = input_data[0]["num_loras"][0]
    return [
        v[0]
        for k, v in input_data[0].items()
        if re.search(attribute, k) != None and v[0] != "None"
    ][:lora_count]


def get_lora_model_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0]["lora_name"][0] != "None":
        return calc_lora_hash(input_data[0]["lora_name"][0], input_data)
    else:
        return ""


SAMPLERS = {
    "easy fullkSampler": {
        "positive": "positive",
        "negative": "negative",
    },
    "easy preSampling": {
        
    },
    "easy preSamplingAdvanced": {
        
    },
    "easy preSamplingCascade": {
        
    },
    "easy preSamplingCustom": {
        
    },
    "easy preSamplingDynamicCFG": {
        
    },
    "easy preSamplingLayerDiffusion": {
        
    },
    "easy preSamplingNoiseIn": {
        
    },
    "easy preSamplingSdTurbo": {
        
    }
}





CAPTURE_FIELD_LIST = {
    "easy fullLoader": {
        MetaField.MODEL_NAME: {"field_name": "ckpt_name"},
        MetaField.MODEL_HASH: {"field_name": "ckpt_name", "format": calc_model_hash},
        MetaField.VAE_NAME: {"field_name": "vae_name"},
        MetaField.VAE_HASH: {"field_name": "vae_name", "format": calc_vae_hash},
        MetaField.CLIP_SKIP: {"field_name": "clip_skip", "format": convert_skip_clip},
        MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
        MetaField.EMBEDDING_NAME: {"selector": get_embedding_names},
        MetaField.EMBEDDING_HASH: {"selector": get_embedding_hashes},
        MetaField.IMAGE_WIDTH: {"field_name": "empty_latent_width"},
        MetaField.IMAGE_HEIGHT: {"field_name": "empty_latent_height"},
        MetaField.LORA_MODEL_NAME: {"selector": get_lora_model_names},
        MetaField.LORA_MODEL_HASH: {"selector": get_lora_model_hashes},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_lora_strengths},
        MetaField.LORA_STRENGTH_CLIP: {"selector": get_lora_clip_strengths},
    },
    "easy comfyLoader": {
        MetaField.MODEL_NAME: {"field_name": "ckpt_name"},
        MetaField.MODEL_HASH: {"field_name": "ckpt_name", "format": calc_model_hash},
        MetaField.CLIP_SKIP: {"field_name": "clip_skip", "format": convert_skip_clip},
        MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
        MetaField.IMAGE_WIDTH: {"field_name": "empty_latent_width"},
        MetaField.IMAGE_HEIGHT: {"field_name": "empty_latent_height"},
        MetaField.LORA_MODEL_NAME: {"field_name": "lora_name"},
        MetaField.LORA_MODEL_HASH: {"selector": get_lora_model_hash},
        MetaField.LORA_STRENGTH_MODEL: {"field_name": "lora_model_strength"},
        MetaField.LORA_STRENGTH_CLIP: {"field_name": "lora_clip_strength"},
    },
    "easy fullkSampler": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy preSampling": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy preSamplingAdvanced": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy preSamplingCascade": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy preSamplingCustom": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy preSamplingDynamicCFG": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy preSamplingLayerDiffusion": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy preSamplingNoiseIn": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy preSamplingSdTurbo": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "easy loraStack": {
        MetaField.LORA_MODEL_NAME: {"selector": get_lora_model_name_stack},
        MetaField.LORA_MODEL_HASH: {"selector": get_lora_model_hash_stack},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_lora_strength_model_stack},
        MetaField.LORA_STRENGTH_CLIP: {"selector": get_lora_strength_clip_stack},
    },
}