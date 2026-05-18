# https://github.com/ClownsharkBatwing/RES4LYF
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_lora_hash, convert_skip_clip


SAMPLERS = {
    "SharkSampler": {
        "positive": "positive",
        "negative": "negative",
    },
    "SharkSampler_Beta": {
        "positive": "positive",
        "negative": "negative",
    },
    "SharkChainsampler_Beta": {
        "positive": "positive",
        "negative": "negative",
    },
    "ClownsharKSampler_Beta": {
        "positive": "positive",
        "negative": "negative",
    },
    "ClownsharkChainsampler_Beta": {
        "positive": "positive",
        "negative": "negative",
    },
    "BongSampler": {
        "positive": "positive",
        "negative": "negative",
    },
}


CAPTURE_FIELD_LIST = {
    "SharkGuider": {
        MetaField.CFG: {"field_name": "cfg"},
    },
    "SharkSampler": {
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise"},
    },
    "SharkSampler_Beta": {
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise"},
    },
    "SharkChainsampler_Beta": {
        MetaField.CFG: {"field_name": "cfg"},
    },
    "ClownsharKSampler_Beta": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise"},
    },
    "ClownsharkChainsampler_Beta": {
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
    },
}