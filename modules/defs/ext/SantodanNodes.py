import re
import os
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_vae_hash, calc_lora_hash

# --- Existing Helpers (Restored) ---

try:
    from ..formatters import calc_clip_hash
except ImportError:
    def calc_clip_hash(name):
        return f"hash_for_{name}"

def get_model_name(node_id, obj, prompt, extra_data, outputs, input_data):
    mode = input_data[0].get("load_mode", ["full_checkpoint"])[0]
    key = "ckpt_name" if mode == "full_checkpoint" else "base_model"
    return input_data[0].get(key, [None])[0]

def get_model_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    model_name = get_model_name(node_id, obj, prompt, extra_data, outputs, input_data)
    if model_name:
        return calc_model_hash(model_name)
    return None

def get_vae_name(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("vae_model", [None])[0]
    return None

def get_vae_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    vae_name = get_vae_name(node_id, obj, prompt, extra_data, outputs, input_data)
    if vae_name:
        return calc_vae_hash(vae_name)
    return None

def get_clip_names(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        clip_names = []
        for key in ["clip_model_1", "clip_model_2", "clip_model_3"]:
            name = input_data[0].get(key, [None])[0]
            if name and name != "None":
                clip_names.append(name)
        return clip_names if clip_names else None
    return None

def get_clip_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    names = get_clip_names(node_id, obj, prompt, extra_data, outputs, input_data)
    if names:
        return [calc_clip_hash(name) for name in names]
    return None

def get_clip_type(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("clip_type", [None])[0]
    return None

def get_unet_dtype(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("weight_dtype", [None])[0]
    return None

def get_metadata_field(field_name, node_id, obj, prompt, extra_data, outputs, input_data):
    metadata_dict = input_data[0].get("metadata", [None])[0]
    if metadata_dict and isinstance(metadata_dict, dict):
        return metadata_dict.get(field_name)
    return None

# --- Hub Node Logic ---

def _get_hub_combined_string(node_id, obj, prompt, extra_data, outputs, input_data):
    """
    Get the combined LoRA string from LoraMetadataHub.

    Primary source: the node's runtime output slot 1 (combined_loras),
    available in _resolved_node_texts after the bulk cache scan.
    Fallback: parse loras_X inputs from input_data (works when inputs are
    hardcoded strings, not links).
    """
    # Primary: use the resolved output string from the cache
    try:
        from ...capture import _resolved_node_texts
        nid = str(node_id)
        combined = (
            _resolved_node_texts.get(f"{nid}:1")   # slot 1 = combined_loras
            or _resolved_node_texts.get(nid)
        )
        if combined and isinstance(combined, str):
            # Strip leading "None, " entries produced when some loras_X inputs are empty
            parts = [p.strip() for p in combined.split(",") if p.strip() and p.strip().lower() != "none"]
            combined = ", ".join(parts)
            if combined:
                return combined
    except Exception:
        pass

    # Fallback: parse loras_X from input_data (hardcoded inputs only)
    inputs = input_data[0] if input_data else {}
    parts = []
    for i in range(1, 4):
        val = inputs.get(f"loras_{i}", "")
        if isinstance(val, list):
            val = val[0] if val else ""
        if val and isinstance(val, str) and val.strip().lower() != "none":
            parts.append(val.strip())
    return ", ".join(filter(None, parts)) or None


def parse_lora_hub_data(node_id, obj, prompt, extra_data, outputs, input_data):
    """
    Parse LoRA name/strength pairs from the combined LoRA string.
    Format produced by RandomLoRAFolderModel: "path/lora.safetensors (0.85)"
    """
    combined = _get_hub_combined_string(node_id, obj, prompt, extra_data, outputs, input_data)
    if not combined:
        return []
    matches = re.findall(r"([^,]+?)\s\(([-+]?\d*\.?\d+)\)", combined)
    return [{"name": m[0].strip(), "strength": float(m[1])} for m in matches]


def get_hub_lora_names(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(node_id, obj, prompt, extra_data, outputs, input_data)
    return [d["name"] for d in data] if data else None

def get_hub_lora_strengths(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(node_id, obj, prompt, extra_data, outputs, input_data)
    return [d["strength"] for d in data] if data else None

def get_hub_lora_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(node_id, obj, prompt, extra_data, outputs, input_data)
    if not data:
        return None
    hashes = []
    for d in data:
        try:
            h = calc_lora_hash(d["name"], input_data)
            hashes.append(h if h else None)
        except Exception:
            hashes.append(None)
    return hashes


# --- Mapping ---

CAPTURE_FIELD_LIST = {
    "ModelAssembler": {
        MetaField.MODEL_NAME: {"selector": get_model_name},
        MetaField.MODEL_HASH: {"selector": get_model_hash},
        MetaField.VAE_NAME: {"selector": get_vae_name},
        MetaField.VAE_HASH: {"selector": get_vae_hash},
        "Clip Model Name(s)": {"selector": get_clip_names},
        "Clip Model Hash(es)": {"selector": get_clip_hashes},
        "Clip Type": {"selector": get_clip_type},
        "UNet Weight Type": {"selector": get_unet_dtype},
    },
    "ModelAssemblerMetadata": {
        MetaField.MODEL_NAME:     {"selector": lambda *args: get_metadata_field("model_name", *args)},
        MetaField.MODEL_HASH:     {"selector": lambda *args: get_metadata_field("model_hash", *args)},
        MetaField.VAE_NAME:       {"selector": lambda *args: get_metadata_field("vae_name", *args)},
        MetaField.VAE_HASH:       {"selector": lambda *args: get_metadata_field("vae_hash", *args)},
        "Clip Model Name(s)": {"selector": lambda *args: get_metadata_field("clip_names", *args)},
        "Clip Model Hash(es)": {"selector": lambda *args: get_metadata_field("clip_hashes", *args)},
        "Clip Type":          {"selector": lambda *args: get_metadata_field("clip_type", *args)},
        "UNet Weight Type":   {"selector": lambda *args: get_metadata_field("unet_dtype", *args)},
    },
    "LoraMetadataHub": {
        MetaField.LORA_MODEL_NAME:     {"selector": get_hub_lora_names},
        MetaField.LORA_MODEL_HASH:     {"selector": get_hub_lora_hashes},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_hub_lora_strengths},
        MetaField.LORA_STRENGTH_CLIP:  {"selector": get_hub_lora_strengths},
    },
}