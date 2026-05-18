# checkpoint-Discovery-Hub - https://github.com/Light-x02/ComfyUI-checkpoint-Discovery-Hub

import json
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_unet_hash, calc_vae_hash

def _cdh_extract_ckpt(selection_data, input_data=None):
    try:
        if isinstance(selection_data, (bytes, bytearray)):
            selection_data = selection_data.decode("utf-8", "ignore")
        if isinstance(selection_data, str):
            s = selection_data.strip()
            data = json.loads(s) if s else {}
        elif isinstance(selection_data, dict):
            data = selection_data
        else:
            return ""
        return (data.get("ckpt") or "").strip()
    except Exception:
        return ""

def _cdh_calc_model_hash(selection_data, input_data=None):
    name = _cdh_extract_ckpt(selection_data, input_data)
    if not name:
        return ""
    try:
        h = calc_unet_hash(name)
        if h:
            return h
    except Exception:
        pass
    try:
        h = calc_model_hash(name)
        if h:
            return h
    except Exception:
        pass
    return ""

def _cdh_extract_vae(selection_data, input_data=None):
    try:
        if isinstance(selection_data, (bytes, bytearray)):
            selection_data = selection_data.decode("utf-8", "ignore")
        if isinstance(selection_data, str):
            s = selection_data.strip()
            data = json.loads(s) if s else {}
        elif isinstance(selection_data, dict):
            data = selection_data
        else:
            return ""
        vae = data.get("vae") or {}
        return (vae.get("vae_name") or "").strip()
    except Exception:
        return ""

def _cdh_calc_vae_hash(selection_data, input_data=None):
    name = _cdh_extract_vae(selection_data, input_data)
    if not name:
        return ""
    try:
        return calc_vae_hash(name) or ""
    except Exception:
        return ""

CAPTURE_FIELD_LIST = {
    "CheckpointDiscoveryHub": {
        MetaField.MODEL_NAME: {"field_name": "selection_data", "format": _cdh_extract_ckpt},
        MetaField.MODEL_HASH: {"field_name": "selection_data", "format": _cdh_calc_model_hash},
        MetaField.VAE_NAME:   {"field_name": "selection_data", "format": _cdh_extract_vae},
        MetaField.VAE_HASH:   {"field_name": "selection_data", "format": _cdh_calc_vae_hash},
    },
}
