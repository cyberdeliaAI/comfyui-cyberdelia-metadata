import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_rgthree_extension():
    package_name = f"_metadata_rgthree_test_{uuid.uuid4().hex}"

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
    formatters.calc_lora_hash = lambda *args, **kwargs: "test-hash"
    sys.modules[formatters.__name__] = formatters

    ext_package = types.ModuleType(f"{package_name}.ext")
    ext_package.__path__ = [str(ROOT / "modules" / "defs" / "ext")]
    sys.modules[ext_package.__name__] = ext_package

    module_name = f"{package_name}.ext.rgthree"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "modules" / "defs" / "ext" / "rgthree.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, meta.MetaField


class RgthreeDefinitionsTests(unittest.TestCase):
    def test_ksampler_config_exposes_sampler_metadata_fields(self):
        extension, meta = _load_rgthree_extension()
        config = extension.CAPTURE_FIELD_LIST["KSampler Config (rgthree)"]

        self.assertEqual(config[meta.STEPS]["field_name"], "steps_total")
        self.assertEqual(config[meta.CFG]["field_name"], "cfg")
        self.assertEqual(config[meta.SAMPLER_NAME]["field_name"], "sampler_name")
        self.assertEqual(config[meta.SCHEDULER]["field_name"], "scheduler")


if __name__ == "__main__":
    unittest.main()
