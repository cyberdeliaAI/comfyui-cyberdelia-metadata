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

    def test_zimage_negpip_conditioning_uses_original_prompt_inputs(self):
        prompt = {
            "sampler": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "guider": ["guider_bundle", 3],
                    "noise": ["noise", 0],
                    "sampler": ["sampler_select", 0],
                    "sigmas": ["sigmas", 0],
                },
            },
            "guider_bundle": {
                "class_type": "CustomGuiderBundle",
                "inputs": {"conditioning": ["sampler_context", 4]},
            },
            "sampler_context": {
                "class_type": "Context Big (rgthree)",
                "inputs": {
                    "positive": ["prompt_context", 4],
                    "negative": ["prompt_context", 5],
                },
            },
            "prompt_context": {
                "class_type": "Context Big (rgthree)",
                "inputs": {
                    "positive": ["negpip", 1],
                    "negative": ["negpip", 2],
                },
            },
            "negpip": {
                "class_type": "ZImageNegPipPrompt",
                "inputs": {
                    "model": ["model", 0],
                    "clip": ["clip", 0],
                    "positive": ["positive_text", 0],
                    "negative": ["negative_text", 0],
                },
            },
            "positive_text": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": "photorealistic portrait"},
            },
            "negative_text": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": "lowres, bad anatomy, watermark"},
            },
        }
        # Reproduce the execution cache: slot 3 and the bare node key contain
        # compiled_prompt, while conditioning slots 1/2 contain no strings.
        self.capture._resolved_node_texts.update({
            "negpip": "photorealistic portrait, (lowres:-1)",
            "negpip:3": "photorealistic portrait, (lowres:-1)",
        })
        try:
            actual = self.capture._find_prompt_texts(
                prompt, outputs=None, sampler_node_id="sampler"
            )
        finally:
            self.capture._resolved_node_texts.clear()

        self.assertEqual(
            actual,
            ("photorealistic portrait", "lowres, bad anatomy, watermark"),
        )

        meta = sys.modules[f"{self.capture.__package__}.defs.meta"].MetaField
        captured = defaultdict(
            list,
            {
                meta.STEPS: [("sigmas", 8)],
                meta.SAMPLER_NAME: [("sampler_select", "euler")],
                meta.SCHEDULER: [("sigmas", "beta")],
                meta.CFG: [("guider", 1.0)],
                meta.SEED: [("noise", 42)],
            },
        )
        pnginfo = self.capture.Capture.gen_pnginfo_dict(
            captured,
            defaultdict(list),
            prompt,
            sampler_node_id="sampler",
        )
        self.assertEqual(pnginfo["Positive prompt"], "photorealistic portrait")
        self.assertEqual(
            pnginfo["Negative prompt"], "lowres, bad anatomy, watermark"
        )

    def test_zimage_negpip_survives_clownshar_context_string_caches(self):
        prompt = {
            "sampler": {
                "class_type": "ClownsharKSampler_Beta",
                "inputs": {
                    "model": ["sampler_context", 1],
                    "positive": ["sampler_context", 4],
                    "negative": ["sampler_context", 5],
                    "latent_image": ["sampler_context", 6],
                    "seed": ["sampler_context", 8],
                },
            },
            "sampler_context": {
                "class_type": "Context Big (rgthree)",
                "inputs": {
                    "positive": ["main_context", 4],
                    "negative": ["main_context", 5],
                },
            },
            "main_context": {
                "class_type": "Context Big (rgthree)",
                "inputs": {
                    "positive": ["prompt_context", 4],
                    "negative": ["prompt_context", 5],
                },
            },
            "prompt_context": {
                "class_type": "Context Big (rgthree)",
                "inputs": {
                    "positive": ["negpip", 1],
                    "negative": ["negpip", 2],
                },
            },
            "negpip": {
                "class_type": "ZImageNegPipPrompt",
                "inputs": {
                    "model": ["model", 0],
                    "clip": ["clip", 0],
                    "positive": ["positive_text", 0],
                    "negative": ["negative_text", 0],
                },
            },
            "positive_text": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": "neon glitch portrait"},
            },
            "negative_text": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": "lowres, bad anatomy, watermark"},
            },
            # Deliberately present but not connected to the selected sampler.
            "unused_negpip": {
                "class_type": "ZImageNegPipPrompt",
                "inputs": {
                    "positive": "wrong positive",
                    "negative": "wrong negative",
                },
            },
        }

        # Context Big exposes unrelated STRING outputs (for example sampler,
        # scheduler and text fields). Before the fix, any such slot cache made
        # the conditioning walk stop on the first Context node. NegPiP itself
        # also exposes compiled_prompt on slot 3, which must stay ignored.
        self.capture._resolved_node_texts.update({
            "sampler_context:12": "multistep/dpmpp_2m",
            "main_context:13": "beta57",
            "prompt_context:17": "unrelated context text",
            "negpip": "neon glitch portrait, (lowres:-1)",
            "negpip:3": "neon glitch portrait, (lowres:-1)",
        })
        try:
            actual = self.capture._find_prompt_texts(
                prompt, outputs=None, sampler_node_id="sampler"
            )
        finally:
            self.capture._resolved_node_texts.clear()

        self.assertEqual(
            actual,
            ("neon glitch portrait", "lowres, bad anatomy, watermark"),
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

    def test_explicit_sampler_id_anchors_prompt_selection(self):
        prompt = {
            "wrong_sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["wrong_positive", 0],
                    "negative": ["wrong_negative", 0],
                },
            },
            "right_sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["right_positive", 0],
                    "negative": ["right_negative", 0],
                },
            },
            "wrong_positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "wrong positive"},
            },
            "wrong_negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "wrong negative"},
            },
            "right_positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "right positive"},
            },
            "right_negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "right negative"},
            },
        }

        actual = self.capture._find_prompt_texts(
            prompt, outputs=None, sampler_node_id="right_sampler"
        )

        self.assertEqual(actual, ("right positive", "right negative"))

    def test_batch_text_selection_preserves_list_positions(self):
        self.assertEqual(
            self.capture._coerce_text_value(["first", "second"], batch_index=1),
            "second",
        )
        self.assertIsNone(
            self.capture._coerce_text_value(["first", "", "third"], batch_index=1)
        )
        self.assertEqual(
            self.capture._coerce_text_value(["first", "", "third"], batch_index=2),
            "third",
        )
        self.assertEqual(
            self.capture.Capture._apply_formatting(
                ["first", "second"], ({},), None, batch_index=1
            ),
            "second",
        )
        self.assertTrue(self.capture._has_text_value(["", "second"]))

    def test_formatting_skips_unresolved_batch_values(self):
        calls = []

        def formatter(value, input_data):
            calls.append((value, input_data))
            return "unexpected"

        for value in ((None,), [None], (), []):
            with self.subTest(value=value):
                self.assertIsNone(
                    self.capture.Capture._apply_formatting(
                        value, ({},), formatter, batch_index=0
                    )
                )

        self.assertEqual(calls, [])

    def test_runtime_prompt_cache_selects_requested_batch_item(self):
        prompt = {
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                },
            },
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "static fallback"},
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "negative"},
            },
        }
        self.capture._resolved_node_texts["positive"] = ["", "batch two"]
        try:
            actual = self.capture._find_prompt_texts(
                prompt, outputs=None, batch_index=1, sampler_node_id="sampler"
            )
        finally:
            self.capture._resolved_node_texts.clear()

        self.assertEqual(actual, ("batch two", "negative"))

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

    def test_integrated_hires_upscaler_beats_later_image_enhancer(self):
        meta = sys.modules[f"{self.capture.__package__}.defs.meta"].MetaField
        inputs = defaultdict(
            list,
            {
                meta.UPSCALE_MODEL_NAME: [
                    ("skin", "1xSkinContrast-SuperUltraCompact.pth", 1),
                    ("hires", "RealisticRescaler", 5),
                ],
                meta.UPSCALE_MODEL_HASH: [
                    ("skin", "skin-hash", 1),
                    ("hires", "hires-hash", 5),
                ],
            },
        )
        prompt = {
            "skin_apply": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": ["skin", 0],
                    "image": ["image", 0],
                },
            },
            "hires_apply": {
                "class_type": "UltimateSDUpscale",
                "inputs": {
                    "upscale_model": ["hires", 0],
                    "image": ["image", 0],
                    "model": ["model", 0],
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                    "vae": ["vae", 0],
                    "upscale_by": 1.5,
                    "denoise": 0.1,
                },
            },
        }

        stage = self.capture._select_hires_upscale_stage(inputs, prompt)

        self.assertEqual(stage["loader_id"], "hires")
        self.assertEqual(stage["name"], "RealisticRescaler")
        self.assertEqual(stage["hash"], "hires-hash")
        self.assertEqual(stage["scale"], 1.5)
        self.assertEqual(stage["denoise"], 0.1)

    def test_hires_metadata_comes_from_one_atomic_stage(self):
        prompt, positive, negative = _negpip_prompt_graph()
        prompt.update({
            "scale": {"class_type": "easy float", "inputs": {"value": 1.5}},
            "skin_apply": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["skin", 0]},
            },
            "hires_apply": {
                "class_type": "UltimateSDUpscale",
                "inputs": {
                    "upscale_model": ["hires", 0],
                    "model": ["model", 0],
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                    "vae": ["vae", 0],
                    "upscale_by": ["scale", 0],
                    "denoise": 0.1,
                },
            },
        })
        meta = sys.modules[f"{self.capture.__package__}.defs.meta"].MetaField
        sampler_inputs = defaultdict(
            list,
            {
                meta.POSITIVE_PROMPT: [("positive", positive)],
                meta.NEGATIVE_PROMPT: [("negative", negative)],
                meta.STEPS: [("scheduler", 8)],
                meta.SAMPLER_NAME: [("sampler_select", "er_sde")],
                meta.SCHEDULER: [("scheduler", "beta57")],
                meta.CFG: [("cfg", 1.0)],
                meta.SEED: [("noise", 45)],
            },
        )
        image_inputs = defaultdict(
            list,
            {
                meta.UPSCALE_MODEL_NAME: [
                    ("skin", "1xSkinContrast-SuperUltraCompact.pth", 1),
                    ("hires", "RealisticRescaler", 5),
                ],
                meta.UPSCALE_MODEL_HASH: [
                    ("skin", "skin-hash", 1),
                    ("hires", "hires-hash", 5),
                ],
            },
        )
        active_trace = {
            "skin_apply": (1, "ImageUpscaleWithModel"),
            "skin": (2, "UpscaleModelLoader"),
            "hires_apply": (4, "UltimateSDUpscale"),
            "hires": (5, "UpscaleModelLoader"),
            "scale": (5, "easy float"),
        }

        pnginfo = self.capture.Capture.gen_pnginfo_dict(
            sampler_inputs,
            image_inputs,
            prompt,
            sampler_node_id="sampler",
            active_trace_tree=active_trace,
        )

        self.assertEqual(pnginfo["Hires upscaler"], "RealisticRescaler")
        self.assertEqual(pnginfo["Hires upscale"], "1.5")
        self.assertEqual(pnginfo["Denoising strength"], 0.1)

    def test_image_only_upscalers_keep_nearest_first_order(self):
        meta = sys.modules[f"{self.capture.__package__}.defs.meta"].MetaField
        inputs = defaultdict(
            list,
            {
                meta.UPSCALE_MODEL_NAME: [
                    ("nearest", "nearest.pth", 1),
                    ("earlier", "earlier.pth", 5),
                ],
            },
        )
        prompt = {
            "nearest_apply": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["nearest", 0]},
            },
            "earlier_apply": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["earlier", 0]},
            },
        }

        stage = self.capture._select_hires_upscale_stage(inputs, prompt)

        self.assertEqual(stage["loader_id"], "nearest")

    def test_downstream_image_scale_is_attached_to_upscale_stage(self):
        meta = sys.modules[f"{self.capture.__package__}.defs.meta"].MetaField
        inputs = defaultdict(
            list,
            {
                meta.UPSCALE_MODEL_NAME: [("model", "upscaler.pth", 3)],
                meta.UPSCALE_BY: [("scale", 2.0, 1)],
            },
        )
        prompt = {
            "apply": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["model", 0]},
            },
            "scale": {
                "class_type": "ImageScaleBy",
                "inputs": {"image": ["apply", 0], "scale_by": 2.0},
            },
        }
        active_trace = {
            "scale": (1, "ImageScaleBy"),
            "apply": (2, "ImageUpscaleWithModel"),
            "model": (3, "UpscaleModelLoader"),
        }

        stage = self.capture._select_hires_upscale_stage(
            inputs, prompt, active_trace
        )

        self.assertEqual(stage["consumer_id"], "apply")
        self.assertEqual(stage["scale"], 2.0)

    def test_off_path_consumer_cannot_reclassify_shared_upscaler(self):
        meta = sys.modules[f"{self.capture.__package__}.defs.meta"].MetaField
        inputs = defaultdict(
            list,
            {meta.UPSCALE_MODEL_NAME: [("shared", "shared.pth", 2)]},
        )
        prompt = {
            "active_apply": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["shared", 0]},
            },
            "off_path_apply": {
                "class_type": "UltimateSDUpscale",
                "inputs": {
                    "upscale_model": ["shared", 0],
                    "model": ["model", 0],
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                },
            },
        }
        active_trace = {
            "active_apply": (1, "ImageUpscaleWithModel"),
            "shared": (2, "UpscaleModelLoader"),
        }

        stage = self.capture._select_hires_upscale_stage(
            inputs, prompt, active_trace
        )

        self.assertFalse(stage["is_diffusion"])
        self.assertEqual(stage["consumer_id"], "active_apply")

    def test_lora_name_and_hash_pair_by_source_node(self):
        meta = sys.modules[f"{self.capture.__package__}.defs.meta"].MetaField
        inputs = defaultdict(
            list,
            {
                meta.LORA_MODEL_NAME: [
                    ("missing", "missing.safetensors", 1),
                    ("matched", "matched.safetensors", 2),
                ],
                meta.LORA_MODEL_HASH: [("matched", "matched-hash", 2)],
            },
        )

        result = self.capture.Capture.extract_model_info(
            inputs, meta.LORA_MODEL_NAME, "Lora"
        )

        self.assertEqual(result["Lora_0 name"], "matched")
        self.assertEqual(result["Lora_0 hash"], "matched-hash")
        self.assertNotIn("Lora_1 name", result)


if __name__ == "__main__":
    unittest.main()
