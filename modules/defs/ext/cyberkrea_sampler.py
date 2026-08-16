"""Metadata integration for cyberdeliaAI/ComfyUI-CyberKrea-Sampler."""

import re

from ..meta import MetaField


_CONTEXT_OUTPUT_FIELDS = {
    8: "seed",
    9: "steps",
    11: "cfg",
}


def _is_link(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _resolve_scalar(value, prompt, visited=None):
    """Resolve a primitive through value nodes and rgthree context relays."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if not _is_link(value):
        return None

    node_id, output_slot = str(value[0]), value[1]
    visited = set() if visited is None else visited
    marker = (node_id, output_slot)
    if marker in visited:
        return None
    visited = visited | {marker}

    node = prompt.get(node_id)
    if node is None:
        return None
    inputs = node.get("inputs", {})
    class_type = node.get("class_type", "")

    if "Context" in class_type and "rgthree" in class_type:
        field_name = _CONTEXT_OUTPUT_FIELDS.get(output_slot)
        if field_name:
            resolved = _resolve_scalar(inputs.get(field_name), prompt, visited)
            if resolved is not None:
                return resolved
            base_ctx = inputs.get("base_ctx")
            if _is_link(base_ctx):
                return _resolve_scalar(
                    [base_ctx[0], output_slot], prompt, visited
                )

    for field_name in ("value", "seed", "noise_seed", "steps", "number"):
        if field_name not in inputs:
            continue
        resolved = _resolve_scalar(inputs[field_name], prompt, visited)
        if resolved is not None:
            return resolved
    return None


def get_seed(node_id, obj, prompt, extra_data, outputs, input_data):
    return _resolve_scalar(obj.get("inputs", {}).get("seed"), prompt)


def get_cfg(node_id, obj, prompt, extra_data, outputs, input_data):
    # CyberKrea has a CFG=1 baseline. A directly connected negative enables a
    # sigma-window guider, whose variable range is recorded in the detailed
    # fields below instead of pretending it is one fixed CFG value.
    return 1.0


def get_sampler_name(node_id, obj, prompt, extra_data, outputs, input_data):
    sampler = obj.get("inputs", {}).get("sampler")
    return {
        "euler": "CyberKrea Euler",
        "euler_2m": "CyberKrea Euler 2M",
    }.get(sampler, f"CyberKrea {sampler}" if sampler else "CyberKrea")


def get_sampler_details(node_id, obj, prompt, extra_data, outputs, input_data):
    inputs = obj.get("inputs", {})
    negative_connected = _is_link(inputs.get("negative"))
    guidance = "Window 1-2.25 (sigma 0.7-0.9)" if negative_connected else "Off / NegPiP"
    return {
        "Krea preset": inputs.get("preset"),
        "Krea guidance": guidance,
        "Krea restart fraction": inputs.get("restart_frac"),
        "Krea restart sigma": inputs.get("sigma_r"),
        "Krea plunge": inputs.get("plunge"),
        "Krea detail": inputs.get("detail"),
        "Krea eta": inputs.get("eta0"),
        "Krea sigma gate": inputs.get("sigma_gate"),
        "Krea contraction": inputs.get("contraction"),
    }


def _resolution_dimension(obj, index):
    resolution = obj.get("inputs", {}).get("resolution")
    if not isinstance(resolution, str):
        return None
    match = re.match(r"^\s*(\d+)x(\d+)", resolution)
    return int(match.group(index)) if match else None


def get_width(node_id, obj, prompt, extra_data, outputs, input_data):
    return _resolution_dimension(obj, 1)


def get_height(node_id, obj, prompt, extra_data, outputs, input_data):
    return _resolution_dimension(obj, 2)


SAMPLERS = {
    "CyberKreaSampler": {
        "positive": "positive",
        "negative": "negative",
    },
}


CAPTURE_FIELD_LIST = {
    "CyberKreaSampler": {
        MetaField.SEED: {"selector": get_seed},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"selector": get_cfg},
        MetaField.SAMPLER_NAME: {"selector": get_sampler_name},
        MetaField.SCHEDULER: {"value": "cyberkrea_restart"},
        MetaField.DENOISE: {"value": 1.0},
        MetaField.CUSTOM_PARAMETERS: {"selector": get_sampler_details},
    },
    "CyberKreaEmptyLatent": {
        MetaField.IMAGE_WIDTH: {"selector": get_width},
        MetaField.IMAGE_HEIGHT: {"selector": get_height},
    },
}
