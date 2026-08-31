"""Metadata integration for pollockjj/ComfyUI-MultiGPU UNET loaders."""

from ..formatters import calc_unet_hash
from ..meta import MetaField


def _unet_fields():
    return {
        MetaField.MODEL_NAME: {"field_name": "unet_name"},
        MetaField.MODEL_HASH: {
            "field_name": "unet_name",
            "format": calc_unet_hash,
        },
    }


CAPTURE_FIELD_LIST = {
    "UNETLoaderMultiGPU": _unet_fields(),
    "UNETLoaderDisTorch2MultiGPU": _unet_fields(),
}
