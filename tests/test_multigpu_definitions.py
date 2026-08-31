import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_multigpu_extension():
    package_name = f"_metadata_multigpu_test_{uuid.uuid4().hex}"

    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "modules" / "defs")]
    sys.modules[package_name] = package

    meta_name = f"{package_name}.meta"
    meta_spec = importlib.util.spec_from_file_location(
        meta_name, ROOT / "modules" / "defs" / "meta.py"
    )
    meta = importlib.util.module_from_spec(meta_spec)
    sys.modules[meta_name] = meta
    meta_spec.loader.exec_module(meta)

    formatters = types.ModuleType(f"{package_name}.formatters")
    formatters.calc_unet_hash = lambda value, input_data=None: (
        f"hash:{value}"
    )
    sys.modules[formatters.__name__] = formatters

    ext_package = types.ModuleType(f"{package_name}.ext")
    ext_package.__path__ = [str(ROOT / "modules" / "defs" / "ext")]
    sys.modules[ext_package.__name__] = ext_package

    module_name = f"{package_name}.ext.comfyui_multigpu"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "modules" / "defs" / "ext" / "comfyui_multigpu.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, meta.MetaField


class MultiGpuDefinitionsTests(unittest.TestCase):
    def test_unet_loaders_capture_name_and_unet_hash(self):
        extension, meta = _load_multigpu_extension()

        for class_type in (
            "UNETLoaderMultiGPU",
            "UNETLoaderDisTorch2MultiGPU",
        ):
            with self.subTest(class_type=class_type):
                fields = extension.CAPTURE_FIELD_LIST[class_type]
                self.assertEqual(
                    fields[meta.MODEL_NAME]["field_name"], "unet_name"
                )
                self.assertEqual(
                    fields[meta.MODEL_HASH]["field_name"], "unet_name"
                )
                self.assertEqual(
                    fields[meta.MODEL_HASH]["format"](
                        "ComfyUI_00001_.safetensors"
                    ),
                    "hash:ComfyUI_00001_.safetensors",
                )


if __name__ == "__main__":
    unittest.main()
