import os
import glob
import importlib

def load_extensions(ext_folder: str, target_package: str, capture_dict: dict, sampler_dict: dict):
    for module_path in glob.glob(os.path.join(ext_folder, "*.py")):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        import_path = f"{target_package}.ext.{module_name}"

        try:
            module = importlib.import_module(import_path)
            if hasattr(module, "CAPTURE_FIELD_LIST"):
                capture_dict.update(module.CAPTURE_FIELD_LIST)
            if hasattr(module, "SAMPLERS"):
                sampler_dict.update(module.SAMPLERS)
        except Exception as e:
            print(f"[MetadataExtension] Failed to load {import_path}: {e}")
