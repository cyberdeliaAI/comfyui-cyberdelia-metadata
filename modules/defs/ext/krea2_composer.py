"""Capture the runtime prompts and regional LoRAs of Krea2 Composer."""
from ..meta import MetaField
from ..formatters import calc_lora_hash
from ...composer_metadata import COMPOSER_CLASS, positive_text, loras, details


def _selector(field):
    def select(node_id, obj, prompt, extra_data, outputs, input_data, batch_index=0):
        from ...capture import _composer_metadata_for_node
        metadata = _composer_metadata_for_node(node_id, obj, prompt, outputs, batch_index)
        rows = loras(metadata)
        if field == "positive":
            return positive_text(metadata)
        if field == "negative":
            return metadata.get("negative_prompt", "")
        if field == "lora_names":
            return [row["name"] for row in rows]
        if field == "lora_hashes":
            return [calc_lora_hash(row["name"]) or "" for row in rows]
        if field == "lora_strengths":
            return [row["strength"] for row in rows]
        if field == "details":
            return details(metadata)
        return metadata.get(field)
    return {"selector": select, "per_image": True}


CAPTURE_FIELD_LIST = {
    COMPOSER_CLASS: {
        MetaField.POSITIVE_PROMPT: _selector("positive"),
        MetaField.NEGATIVE_PROMPT: _selector("negative"),
        MetaField.LORA_MODEL_NAME: _selector("lora_names"),
        MetaField.LORA_MODEL_HASH: _selector("lora_hashes"),
        MetaField.LORA_STRENGTH_MODEL: _selector("lora_strengths"),
        MetaField.CUSTOM_PARAMETERS: _selector("details"),
        MetaField.IMAGE_WIDTH: _selector("width"),
        MetaField.IMAGE_HEIGHT: _selector("height"),
    },
}
