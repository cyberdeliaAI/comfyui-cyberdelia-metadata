# https://github.com/kijai/ComfyUI-WanVideoWrapper

from ..meta import MetaField
from ..formatters import calc_lora_hash, calc_vae_hash, calc_model_hash, convert_skip_clip

# -------------------------------------------------------------------
# Helpers for safe hashing (skip None, "", "none")
# -------------------------------------------------------------------

def get_wan_model_hash(path, _input_data=None):
    if not path or (isinstance(path, str) and path.strip().lower() == "none"):
        return None
    try:
        return calc_model_hash(path)
    except Exception:
        return None

def get_wan_vae_hash(path, _input_data=None):
    if not path or (isinstance(path, str) and path.strip().lower() == "none"):
        return None
    try:
        return calc_vae_hash(path)
    except Exception:
        return None

def get_wan_lora_hash(path, input_data=None):
    if not path or (isinstance(path, str) and path.strip().lower() == "none"):
        return None
    try:
        if input_data:
            return calc_lora_hash(path, input_data)
        return calc_lora_hash(path)
    except Exception:
        return None

# -------------------------------------------------------------------
# Helpers for WanVideoLoraSelectMulti
# -------------------------------------------------------------------

def _coerce_to_scalar_strength(val, default=1.0):
    """Coerce strength-like values to a single float scalar."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return default
        return _coerce_to_scalar_strength(val[0], default)
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return default
    try:
        return float(val)
    except Exception:
        return default

def _coerce_to_string_name(val):
    """Coerce different possible name/path shapes to a string (or None)."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return None
        return _coerce_to_string_name(val[0])
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("path") or val.get("name") or val.get("model") or None
    try:
        return str(val)
    except Exception:
        return None

def _extract_prev_lora_list(item):
    """Normalize prev_lora / lora_stack items into [(name_or_path, strength, clip), ...]"""
    results = []
    if item is None:
        return results

    if isinstance(item, (list, tuple)):
        for el in item:
            if el is None:
                continue
            if isinstance(el, dict):
                name_raw = el.get("path") or el.get("name") or el.get("model") or el.get("model_name")
                name = _coerce_to_string_name(name_raw)
                if not name or (isinstance(name, str) and name.strip().lower() == "none"):
                    continue
                strength = _coerce_to_scalar_strength(el.get("strength", 1.0))
                clip = el.get("clip_strength") or el.get("clip") or el.get("clip_scale") or None
                clip = _coerce_to_scalar_strength(clip, default=None) if clip is not None else None
                results.append((name, strength, clip))
            elif isinstance(el, (list, tuple)):
                name = _coerce_to_string_name(el[0]) if len(el) > 0 else None
                if not name or (isinstance(name, str) and name.strip().lower() == "none"):
                    continue
                strength = _coerce_to_scalar_strength(el[1]) if len(el) > 1 else 1.0
                clip = _coerce_to_scalar_strength(el[2], default=None) if len(el) > 2 else None
                results.append((name, strength, clip))
            elif isinstance(el, str):
                if el.strip().lower() == "none":
                    continue
                results.append((el, 1.0, None))
    elif isinstance(item, dict):
        for k in ("prev_lora", "lora", "loras", "lora_stack", "lora_list"):
            if k in item and item[k]:
                results.extend(_extract_prev_lora_list(item[k]))
    return results

def get_wan_lora_stack_from_inputs(input_data):
    """Returns list of (model_name_or_path, strength, clip_strength_or_None)"""
    results = []

    for item in input_data:
        if not item:
            continue
        if isinstance(item, dict):
            for key in ("prev_lora", "lora", "loras", "lora_stack", "lora_list"):
                if key in item and item[key]:
                    results.extend(_extract_prev_lora_list(item[key]))
            if "lora_stack" in item and item["lora_stack"]:
                for stack in item["lora_stack"]:
                    results.extend(_extract_prev_lora_list(stack))

    if results:
        filtered = []
        for n, s, c in results:
            name = _coerce_to_string_name(n)
            if not name or (isinstance(name, str) and name.strip().lower() == "none"):
                continue
            strength = _coerce_to_scalar_strength(s)
            clip = _coerce_to_scalar_strength(c, default=None) if c is not None else None
            filtered.append((name, strength, clip))
        return filtered

    merged = {}
    for item in input_data:
        if isinstance(item, dict):
            merged.update(item)

    for i in range(5):
        lkey = f"lora_{i}"
        skey = f"strength_{i}"
        lval = merged.get(lkey)
        if not lval or (isinstance(lval, str) and lval.strip().lower() == "none"):
            continue
        strength = _coerce_to_scalar_strength(merged.get(skey, 1.0))
        if strength == 0.0:
            continue
        name = _coerce_to_string_name(lval)
        if not name or (isinstance(name, str) and name.strip().lower() == "none"):
            continue
        results.append((name, strength, None))
    return results

def get_wan_lora_model_names(node_id, obj, prompt, extra_data, outputs, input_data):
    stack = get_wan_lora_stack_from_inputs(input_data)
    return [entry[0] for entry in stack]

def get_wan_lora_model_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    names = get_wan_lora_model_names(node_id, obj, prompt, extra_data, outputs, input_data)
    return [get_wan_lora_hash(n, input_data) if n else None for n in names]

def get_wan_lora_strength_model(node_id, obj, prompt, extra_data, outputs, input_data):
    stack = get_wan_lora_stack_from_inputs(input_data)
    return [entry[1] for entry in stack]

def get_wan_lora_strength_clip(node_id, obj, prompt, extra_data, outputs, input_data):
    stack = get_wan_lora_stack_from_inputs(input_data)
    return [entry[2] for entry in stack]

# -------------------------------------------------------------------
# SAMPLERS mapping
# -------------------------------------------------------------------

SAMPLERS = {
    "WanVideoSampler": {
        "positive": "text_embeds",
        "negative": "text_embeds",
    },
}

# -------------------------------------------------------------------
# CAPTURE_FIELD_LIST
# -------------------------------------------------------------------

CAPTURE_FIELD_LIST = {
    "WanVideoModelLoader": {
        MetaField.MODEL_NAME: {"field_name": "model"},
        MetaField.MODEL_HASH: {"field_name": "model", "format": get_wan_model_hash},
        MetaField.CLIP_SKIP: {"field_name": "clip_skip", "format": convert_skip_clip},
        MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
    },
    "WanVideoVAELoader": {
        MetaField.VAE_NAME: {"field_name": "model_name"},
        MetaField.VAE_HASH: {"field_name": "model_name", "format": get_wan_vae_hash},
    },
    "WanVideoLoraSelectMulti": {
        MetaField.LORA_MODEL_NAME: {"selector": get_wan_lora_model_names},
        MetaField.LORA_MODEL_HASH: {"selector": get_wan_lora_model_hashes},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_wan_lora_strength_model},
        MetaField.LORA_STRENGTH_CLIP: {"selector": get_wan_lora_strength_clip},
    },
    "WanVideoSampler": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise_strength"},
    },
}
