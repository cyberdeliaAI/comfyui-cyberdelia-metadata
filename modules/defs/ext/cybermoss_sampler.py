"""Metadata integration for cyberdeliaAI/ComfyUI-CyberMOSS-Sampler."""

import re

from ..meta import MetaField
from .cyberkrea_sampler import _resolve_scalar


def _input(obj, prompt, input_data, name, batch_index=0, default=None):
    """Prefer resolved execution inputs; follow scalar links when unavailable."""
    data = input_data[0] if isinstance(input_data, (list, tuple)) and input_data else {}
    value = data.get(name) if isinstance(data, dict) else None
    if isinstance(value, (list, tuple)):
        # get_input_data returns batch lists, including two-integer batches
        # which must never be mistaken for API prompt links.
        value = value[min(max(batch_index, 0), len(value) - 1)] if value else None
    if isinstance(value, (str, int, float, bool)):
        return value
    value = _resolve_scalar(obj.get("inputs", {}).get(name), prompt)
    return default if value is None else value


def _field(name, default=None):
    def select(node_id, obj, prompt, extra_data, outputs, input_data, batch_index=0):
        return _input(obj, prompt, input_data, name, batch_index, default)
    return {"selector": select, "per_image": True}


def get_sampler_name(node_id, obj, prompt, extra_data, outputs, input_data, batch_index=0):
    sampler = _input(obj, prompt, input_data, "sampler", batch_index)
    return {
        "euler": "CyberMOSS Euler",
        "euler_2m": "CyberMOSS Euler 2M",
    }.get(sampler, f"CyberMOSS {sampler}" if sampler else "CyberMOSS")


def get_scheduler(node_id, obj, prompt, extra_data, outputs, input_data, batch_index=0):
    mode = _input(obj, prompt, input_data, "schedule_mode", batch_index, "cybermoss")
    return {"reference": "moss_reference"}.get(mode, mode)


def get_sampler_details(node_id, obj, prompt, extra_data, outputs, input_data, batch_index=0):
    fields = {
        "MOSS preset": "preset",
        "MOSS schedule": "schedule_mode",
        "MOSS skip layers": "skip_layers",
        "MOSS restart fraction": "restart_frac",
        "MOSS restart sigma": "sigma_r",
        "MOSS plunge": "plunge",
        "MOSS detail": "detail",
        "MOSS eta": "eta0",
        "MOSS sigma gate": "sigma_gate",
        "MOSS contraction": "contraction",
    }
    details = {label: _input(obj, prompt, input_data, field, batch_index)
               for label, field in fields.items()}
    # CyberMOSS fixes shift=5 independently of upstream model sampling patches.
    details["MOSS shift"] = 5.0
    return details


def _dimension(index):
    def select(node_id, obj, prompt, extra_data, outputs, input_data, batch_index=0):
        resolution = _input(obj, prompt, input_data, "resolution", batch_index)
        size = _input(obj, prompt, input_data, "size", batch_index)
        if not isinstance(resolution, str):
            return None
        # Use the sampler's own bucket metadata for its stale-tier recovery.
        from nodes import NODE_CLASS_MAPPINGS
        node_class = NODE_CLASS_MAPPINGS.get("CyberMOSSEmptyLatent")
        if node_class is not None:
            required = node_class.INPUT_TYPES().get("required", {})
            options = required.get("size", (None, {}))[1].get("cybermoss_resolutions", {})
            labels = options.get(size, [])
            if labels and resolution not in labels:
                suffix = "(" + resolution.rpartition("(")[2]
                resolution = next((label for label in labels if label.endswith(suffix)), resolution)
        match = re.match(r"^\s*(\d+)x(\d+)", resolution)
        return int(match.group(index)) if match else None
    return {"selector": select, "per_image": True}


# This registration also anchors model/LoRA/VAE discovery to the sampler's
# ancestry, rather than the checkpoint-only fallback for unknown samplers.
SAMPLERS = {
    "CyberMOSSSampler": {"positive": "positive", "negative": "negative"},
}

CAPTURE_FIELD_LIST = {
    "CyberMOSSSampler": {
        MetaField.SEED: _field("seed"),
        MetaField.STEPS: _field("steps"),
        MetaField.CFG: _field("cfg"),
        MetaField.SAMPLER_NAME: {"selector": get_sampler_name, "per_image": True},
        MetaField.SCHEDULER: {"selector": get_scheduler, "per_image": True},
        MetaField.DENOISE: _field("denoise", 1.0),
        MetaField.CUSTOM_PARAMETERS: {"selector": get_sampler_details, "per_image": True},
    },
    "CyberMOSSEmptyLatent": {
        MetaField.IMAGE_WIDTH: _dimension(1),
        MetaField.IMAGE_HEIGHT: _dimension(2),
    },
}
