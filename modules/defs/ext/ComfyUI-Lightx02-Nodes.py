from ..meta import MetaField

SAMPLERS = {
    "FluxSettingsPipe": {
        "positive": "conditioning.positive",
        "negative": "conditioning.negative",
    },
    "FluxPipeUnpack": {
        "positive": "conditioning.positive",
        "negative": "conditioning.negative",
    },
}

CAPTURE_FIELD_LIST = {
    "FluxSettingsPipe": {
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.DENOISE: {"field_name": "denoise"},
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.IMAGE_WIDTH: {"field_name": "width"},
        MetaField.IMAGE_HEIGHT: {"field_name": "height"},
        MetaField.POSITIVE_PROMPT: {"field_name": "conditioning.positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "conditioning.negative"},
    },
    "FluxPipeUnpack": {
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.DENOISE: {"field_name": "denoise"},
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.IMAGE_WIDTH: {"field_name": "width"},
        MetaField.IMAGE_HEIGHT: {"field_name": "height"},
        MetaField.POSITIVE_PROMPT: {"field_name": "conditioning.positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "conditioning.negative"},
    },
}
