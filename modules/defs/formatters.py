import re
import folder_paths
from ..utils.hash import calc_hash
from ..utils.embedding import get_embedding_file_path

cache_model_hash = {}

# Generalized hash calculation for different folder types
def calc_hash_for_type(folder_type, model_name):
    try:
        filename = folder_paths.get_full_path(folder_type, model_name)
        return calc_hash(filename)
    except Exception as e:
        return ""  # Return empty string if unable to calculate hash

# Replacing calc_model_hash, calc_vae_hash, calc_lora_hash, and calc_unet_hash
def calc_model_hash(model_name, input_data=None):
    return calc_hash_for_type("checkpoints", model_name)

def calc_vae_hash(model_name, input_data=None):
    return calc_hash_for_type("vae", model_name)

def calc_lora_hash(model_name, input_data=None):
    return calc_hash_for_type("loras", model_name)

def calc_unet_hash(model_name, input_data=None):
    return calc_hash_for_type("unet", model_name)

def calc_upscale_hash(model_name, input_data=None):
    return calc_hash_for_type("upscale_models", model_name)


def convert_skip_clip(stop_at_clip_layer, input_data=None):
    return stop_at_clip_layer * -1


SCALING_FACTOR = 8

def get_scaled_width(scaled_by, input_data):
    samples = input_data[0]["samples"][0]["samples"]
    return round(samples.shape[3] * scaled_by * SCALING_FACTOR)

def get_scaled_height(scaled_by, input_data):
    samples = input_data[0]["samples"][0]["samples"]
    return round(samples.shape[2] * scaled_by * SCALING_FACTOR)


embedding_pattern = re.compile(r"embedding:\(?([^\s),]+)\)?")

def _extract_embedding_names_from_text(text):
    return [match.group(1) for match in embedding_pattern.finditer(text)] if "embedding:" in text else []

def extract_embedding_names(text, input_data=None):
    return _extract_embedding_names_from_text(text)

def extract_embedding_hashes(text, input_data=None):
    names = extract_embedding_names(text)
    hashes = [calc_hash(get_embedding_file_path(name)) or "" for name in names]
    return hashes
