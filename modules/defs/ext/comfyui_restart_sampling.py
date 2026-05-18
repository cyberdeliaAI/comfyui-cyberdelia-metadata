# https://github.com/ssitu/ComfyUI_restart_sampling
from ..meta import MetaField


SAMPLERS = {
    "KRestartSampler": {
        "positive": "positive",
        "negative": "negative",
    },
    "KRestartSamplerSimple": {
        "positive": "positive",
        "negative": "negative",
    },
    "KRestartSamplerAdv": {
        "positive": "positive",
        "negative": "negative",
    },
    "KRestartSamplerCustom": {
        "positive": "positive",
        "negative": "negative",
    },
}


CAPTURE_FIELD_LIST = {
    "KRestartSampler": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise"},
    },
    "KRestartSamplerSimple": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise"},
    },
    "KRestartSamplerAdv": {
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "KRestartSamplerCustom": {
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "RestartScheduler": {
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
}
