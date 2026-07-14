import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_formatters():
    package_name = "_metadata_formatter_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "modules")]
    sys.modules[package_name] = package

    utils = types.ModuleType(f"{package_name}.utils")
    utils.__path__ = []
    sys.modules[utils.__name__] = utils

    hash_module = types.ModuleType(f"{package_name}.utils.hash")
    hash_module.calc_hash = lambda path: "test-hash"
    sys.modules[hash_module.__name__] = hash_module

    embedding = types.ModuleType(f"{package_name}.utils.embedding")
    embedding.get_embedding_file_path = lambda name: name
    sys.modules[embedding.__name__] = embedding

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda folder_type, name: name
    sys.modules["folder_paths"] = folder_paths

    name = f"{package_name}.defs.formatters"
    defs = types.ModuleType(f"{package_name}.defs")
    defs.__path__ = [str(ROOT / "modules" / "defs")]
    sys.modules[defs.__name__] = defs
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "modules" / "defs" / "formatters.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class UpscalerDisplayNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formatters = _load_formatters()

    def test_realistic_rescaler_uses_public_name(self):
        self.assertEqual(
            self.formatters.format_upscale_model_name(
                "4x_RealisticRescaler_100000_G.pth"
            ),
            "RealisticRescaler",
        )

    def test_other_upscaler_names_are_unchanged(self):
        for name in (
            "4x-UltraSharp.pth",
            "1xSkinContrast-SuperUltraCompact.pth",
            "upscalers/custom/model.safetensors",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    self.formatters.format_upscale_model_name(name), name
                )

    def test_embedding_formatters_ignore_non_text_values(self):
        for value in (None, [], [None], (), (None,), 42):
            with self.subTest(value=value):
                self.assertEqual(
                    self.formatters.extract_embedding_names(value), []
                )
                self.assertEqual(
                    self.formatters.extract_embedding_hashes(value), []
                )


if __name__ == "__main__":
    unittest.main()
