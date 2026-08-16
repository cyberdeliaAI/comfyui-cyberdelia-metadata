import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_cyberkrea_extension():
    package_name = f"_metadata_cyberkrea_test_{uuid.uuid4().hex}"

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

    ext_package = types.ModuleType(f"{package_name}.ext")
    ext_package.__path__ = [str(ROOT / "modules" / "defs" / "ext")]
    sys.modules[ext_package.__name__] = ext_package

    module_name = f"{package_name}.ext.cyberkrea_sampler"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "modules" / "defs" / "ext" / "cyberkrea_sampler.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, meta.MetaField


class CyberKreaDefinitionsTests(unittest.TestCase):
    def setUp(self):
        self.extension, self.meta = _load_cyberkrea_extension()

    def test_sampler_is_registered_with_standard_and_detailed_fields(self):
        sampler = self.extension.CAPTURE_FIELD_LIST["CyberKreaSampler"]

        self.assertIn("CyberKreaSampler", self.extension.SAMPLERS)
        self.assertEqual(sampler[self.meta.STEPS]["field_name"], "steps")
        self.assertIn("selector", sampler[self.meta.SEED])
        self.assertIn("selector", sampler[self.meta.SAMPLER_NAME])
        self.assertEqual(
            sampler[self.meta.SCHEDULER]["value"], "cyberkrea_restart"
        )
        self.assertIn("selector", sampler[self.meta.CUSTOM_PARAMETERS])

    def test_seed_resolves_through_the_real_rgthree_relay_shape(self):
        prompt = {
            "seed": {
                "class_type": "Seed (rgthree)",
                "inputs": {"seed": 507808542323527},
            },
            "context_a": {
                "class_type": "Context Big (rgthree)",
                "inputs": {"seed": ["seed", 0]},
            },
            "context_b": {
                "class_type": "Context Big (rgthree)",
                "inputs": {
                    "base_ctx": ["context_a", 0],
                    "steps": ["context_a", 8],
                },
            },
            "context_c": {
                "class_type": "Context Big (rgthree)",
                "inputs": {
                    "base_ctx": ["context_b", 0],
                    "seed": ["context_b", 9],
                },
            },
        }
        sampler = {
            "class_type": "CyberKreaSampler",
            "inputs": {"seed": ["context_c", 8]},
        }

        self.assertEqual(
            self.extension.get_seed(
                "sampler", sampler, prompt, None, None, None
            ),
            507808542323527,
        )

    def test_sampler_values_are_named_truthfully(self):
        sampler = {
            "inputs": {
                "preset": "quality",
                "sampler": "euler_2m",
                "restart_frac": 0.25,
                "sigma_r": 0.65,
                "plunge": True,
                "detail": 0.7,
                "eta0": 1.0,
                "sigma_gate": 0.1,
                "contraction": 0.7,
            }
        }

        self.assertEqual(
            self.extension.get_sampler_name(
                "sampler", sampler, {}, None, None, None
            ),
            "CyberKrea Euler 2M",
        )
        details = self.extension.get_sampler_details(
            "sampler", sampler, {}, None, None, None
        )
        self.assertEqual(details["Krea preset"], "quality")
        self.assertEqual(details["Krea guidance"], "Off / NegPiP")
        self.assertEqual(details["Krea detail"], 0.7)

    def test_empty_latent_resolution_supplies_dimensions(self):
        latent = {"inputs": {"resolution": "1184x1776 (2:3)"}}

        self.assertEqual(
            self.extension.get_width("latent", latent, {}, None, None, None),
            1184,
        )
        self.assertEqual(
            self.extension.get_height("latent", latent, {}, None, None, None),
            1776,
        )


if __name__ == "__main__":
    unittest.main()
