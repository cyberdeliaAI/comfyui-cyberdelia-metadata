import importlib.util
import sys
import types
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_validators():
    package_name = "_metadata_validator_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "modules" / "defs")]
    sys.modules[package_name] = package
    _load_module(
        f"{package_name}.samplers", ROOT / "modules" / "defs" / "samplers.py"
    )
    return _load_module(
        f"{package_name}.validators", ROOT / "modules" / "defs" / "validators.py"
    )


def _load_capture():
    package_name = "_metadata_capture_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "modules")]
    sys.modules[package_name] = package

    hook = types.ModuleType(f"{package_name}.hook")
    hook.current_resolved_texts = {}
    sys.modules[hook.__name__] = hook

    defs = types.ModuleType(f"{package_name}.defs")
    defs.__path__ = [str(ROOT / "modules" / "defs")]
    sys.modules[defs.__name__] = defs

    captures = types.ModuleType(f"{package_name}.defs.captures")
    captures.CAPTURE_FIELD_LIST = {}
    sys.modules[captures.__name__] = captures
    _load_module(f"{package_name}.defs.meta", ROOT / "modules" / "defs" / "meta.py")

    formatters = types.ModuleType(f"{package_name}.defs.formatters")
    for name in (
        "calc_lora_hash",
        "calc_model_hash",
        "extract_embedding_names",
        "extract_embedding_hashes",
    ):
        setattr(formatters, name, lambda *args, **kwargs: None)
    sys.modules[formatters.__name__] = formatters

    utils = types.ModuleType(f"{package_name}.utils")
    utils.__path__ = []
    sys.modules[utils.__name__] = utils
    log = types.ModuleType(f"{package_name}.utils.log")
    log.print_warning = lambda *args, **kwargs: None
    sys.modules[log.__name__] = log

    trace = types.ModuleType(f"{package_name}.trace")
    trace.Trace = type("Trace", (), {})
    sys.modules[trace.__name__] = trace

    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {}
    sys.modules["nodes"] = nodes

    return _load_module(f"{package_name}.capture", ROOT / "modules" / "capture.py")


def _negpip_prompt_graph():
    negative = "(worst quality, low quality, bad anatomy:-1)"
    positive = "masterpiece, best quality, tattooed woman"
    prompt = {
        # Deliberately first: it looks sampler-like but must not beat the real
        # SamplerCustomAdvanced below.
        "context": {
            "class_type": "Context Big (rgthree)",
            "inputs": {
                "positive": ["wrong_positive", 0],
                "negative": ["wrong_negative", 0],
                "seed": 1,
                "steps": 8,
                "cfg": 1.0,
            },
        },
        "wrong_positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "unrelated positive"},
        },
        "wrong_negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "unrelated negative"},
        },
        "sampler": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "guider": ["cfg", 0],
                "noise": ["noise", 0],
                "sampler": ["sampler_select", 0],
                "sigmas": ["scheduler", 0],
            },
        },
        "cfg": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["model", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "cfg": 1.0,
            },
        },
        "positive": {
            "class_type": "CyberdeliaPromptFormatEncode",
            "inputs": {"clip": ["clip", 0], "text": positive},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip", 0], "text": negative},
        },
    }
    return prompt, positive, negative


class NegPipPromptRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validators = _load_validators()
        cls.capture = _load_capture()

    def test_cfg_guider_does_not_classify_negative_clip_as_positive(self):
        prompt, _, _ = _negpip_prompt_graph()
        self.assertFalse(
            self.validators.is_positive_prompt("negative", None, prompt, None, None, None)
        )
        self.assertTrue(
            self.validators.is_negative_prompt("negative", None, prompt, None, None, None)
        )

    def test_basic_guider_does_not_invent_a_negative_branch(self):
        prompt = {
            "sampler": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {"guider": ["guider", 0]},
            },
            "guider": {
                "class_type": "BasicGuider",
                "inputs": {"conditioning": ["positive", 0]},
            },
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a positive prompt"},
            },
        }
        self.assertTrue(
            self.validators.is_positive_prompt("positive", None, prompt, None, None, None)
        )
        self.assertFalse(
            self.validators.is_negative_prompt("positive", None, prompt, None, None, None)
        )
        self.assertEqual(
            self.capture._find_prompt_texts(prompt, outputs=None),
            ("a positive prompt", None),
        )

    def test_classic_ksampler_branches_remain_separate(self):
        prompt = {
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                    "seed": 1,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                },
            },
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "classic positive"},
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "classic negative"},
            },
        }
        self.assertEqual(
            self.capture._find_prompt_texts(prompt, outputs=None),
            ("classic positive", "classic negative"),
        )
        self.assertTrue(
            self.validators.is_positive_prompt("positive", None, prompt, None, None, None)
        )
        self.assertFalse(
            self.validators.is_positive_prompt("negative", None, prompt, None, None, None)
        )
        self.assertTrue(
            self.validators.is_negative_prompt("negative", None, prompt, None, None, None)
        )
        self.assertFalse(
            self.validators.is_negative_prompt("positive", None, prompt, None, None, None)
        )

    def test_real_sampler_beats_sampler_like_context_node(self):
        prompt, positive, negative = _negpip_prompt_graph()
        actual = self.capture._find_prompt_texts(prompt, outputs=None)
        self.assertEqual(actual, (positive, negative))

    def test_graph_roles_replace_valid_looking_misclassified_capture(self):
        prompt, positive, negative = _negpip_prompt_graph()
        meta = sys.modules[f"{self.capture.__package__}.defs.meta"].MetaField
        captured = defaultdict(
            list,
            {
                meta.POSITIVE_PROMPT: [("negative", negative)],
                meta.NEGATIVE_PROMPT: [("negative", negative)],
                meta.STEPS: [("scheduler", 8)],
                meta.SAMPLER_NAME: [("sampler_select", "er_sde")],
                meta.SCHEDULER: [("scheduler", "beta57")],
                meta.CFG: [("cfg", 1.0)],
                meta.SEED: [("noise", 45)],
            },
        )

        pnginfo = self.capture.Capture.gen_pnginfo_dict(
            captured, defaultdict(list), prompt
        )

        self.assertEqual(pnginfo["Positive prompt"], positive)
        self.assertEqual(pnginfo["Negative prompt"], negative)


if __name__ == "__main__":
    unittest.main()
