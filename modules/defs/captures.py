from .meta import MetaField
from .validators import is_positive_prompt, is_negative_prompt
from .formatters import (
    calc_model_hash,
    calc_upscale_hash,
    calc_vae_hash,
    calc_lora_hash,
    calc_unet_hash,
    convert_skip_clip,
    get_scaled_width,
    get_scaled_height,
    extract_embedding_names,
    extract_embedding_hashes,
)

def is_latent_excuted(node_id, obj, prompt, extra_data, outputs, input_data_all):
    """Return scaled width or fall back to scaled_by when sample path is missing."""
    try:
        # Attempt to reach the ndarray: input_data[0]["samples"][0]["samples"]
        samples = input_data_all[0]["samples"][0]["samples"]
        # If samples exists, compute the width with SCALING_FACTOR
        if samples is not None:
            return True
    except (KeyError, IndexError, TypeError, AttributeError):
        # Any failure in the lookup chain or missing attribute -> graceful fallback
        pass
    # Default path: structure changed, simply return the incoming scale
    return False

CAPTURE_FIELD_LIST = {
    "CheckpointLoaderSimple": {
        MetaField.MODEL_NAME: {"field_name": "ckpt_name"},
        MetaField.MODEL_HASH: {"field_name": "ckpt_name", "format": calc_model_hash},
    },
    "Checkpoint Loader with Name (Image Saver)": {
        MetaField.MODEL_NAME: {"field_name": "ckpt_name"},
        MetaField.MODEL_HASH: {"field_name": "ckpt_name", "format": calc_model_hash},
    },
    "CLIPSetLastLayer": {
        MetaField.CLIP_SKIP: {
            "field_name": "stop_at_clip_layer",
            "format": convert_skip_clip,
        },
    },
    "VAELoader": {
        MetaField.VAE_NAME: {"field_name": "vae_name"},
        MetaField.VAE_HASH: {"field_name": "vae_name", "format": calc_vae_hash},
    },
    "EmptyLatentImage": {
        MetaField.IMAGE_WIDTH: {"field_name": "width"},
        MetaField.IMAGE_HEIGHT: {"field_name": "height"},
    },
    "CLIPTextEncode": {
        MetaField.POSITIVE_PROMPT: {
            "field_name": "text",
            "validate": is_positive_prompt,
        },
        MetaField.NEGATIVE_PROMPT: {
            "field_name": "text",
            "validate": is_negative_prompt,
        },
        MetaField.EMBEDDING_NAME: {
            "field_name": "text",
            "format": extract_embedding_names,
        },
        MetaField.EMBEDDING_HASH: {
            "field_name": "text",
            "format": extract_embedding_hashes,
        },
    },
    "KSampler": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise"},
    },
    "KSamplerAdvanced": {
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "LatentUpscale": {
        MetaField.IMAGE_WIDTH: {"field_name": "width"},
        MetaField.IMAGE_HEIGHT: {"field_name": "height"},
    },
    "LatentUpscaleBy": {
        MetaField.IMAGE_WIDTH: {"field_name": "scale_by", "format": get_scaled_width, "validate": is_latent_excuted},
        MetaField.IMAGE_HEIGHT: {
            "field_name": "scale_by",
            "format": get_scaled_height,
            "validate": is_latent_excuted
        },
    },
    "LoraLoader": {
        MetaField.LORA_MODEL_NAME: {"field_name": "lora_name"},
        MetaField.LORA_MODEL_HASH: {
            "field_name": "lora_name",
            "format": calc_lora_hash,
        },
        MetaField.LORA_STRENGTH_MODEL: {"field_name": "strength_model"},
        MetaField.LORA_STRENGTH_CLIP: {"field_name": "strength_clip"},
    },
    "LoraLoaderModelOnly": {
        MetaField.LORA_MODEL_NAME: {"field_name": "lora_name"},
        MetaField.LORA_MODEL_HASH: {
            "field_name": "lora_name",
            "format": calc_lora_hash,
        },
        MetaField.LORA_STRENGTH_MODEL: {"field_name": "strength_model"},
        MetaField.LORA_STRENGTH_CLIP: {"value": 0},
    },
    "UpscaleModelLoader": {
        MetaField.UPSCALE_MODEL_NAME: {"field_name": "model_name"},
        MetaField.UPSCALE_MODEL_HASH: {
            "field_name": "model_name",
            "format": calc_upscale_hash,
        },
    },
    "ImageScaleBy": {
        MetaField.UPSCALE_BY: {"field_name": "scale_by"},
    },
    # Flux - https://comfyanonymous.github.io/ComfyUI_examples/flux/
    "UNETLoader": {
        MetaField.MODEL_NAME: {"field_name": "unet_name"},
        MetaField.MODEL_HASH: {"field_name": "unet_name", "format": calc_unet_hash},
    },
    "RandomNoise": {
        MetaField.SEED: {"field_name": "noise_seed"},
    },
    "Seed (rgthree)": {
        MetaField.SEED: {"field_name": "seed"},
    },
    "BasicScheduler": {
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise"},
    },
    "BetaSamplingScheduler": {
        MetaField.STEPS: {"field_name": "steps"},
    },
    "KSamplerSelect": {
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
    },
    "CFGGuider": {
        MetaField.CFG: {"field_name": "cfg"},
    },
    "SamplerCustom": {
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SEED: {"field_name": "noise_seed"},
    },
}
