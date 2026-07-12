"""Metadata integration for cyberdeliaAI/ComfyUI-NegPiP-ZImage."""

from ..meta import MetaField


CAPTURE_FIELD_LIST = {
    "ZImageNegPipPrompt": {
        MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
    },
}
