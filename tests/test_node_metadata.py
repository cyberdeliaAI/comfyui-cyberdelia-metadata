import asyncio
import importlib.util
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _FakeArray:
    def __rmul__(self, value):
        return self

    def astype(self, dtype):
        return self


class _FakeTensor:
    shape = (16, 16, 3)

    def cpu(self):
        return self

    def numpy(self):
        return _FakeArray()


def _load_node_module(output_dir, saved_pnginfo):
    package_name = f"_metadata_node_test_{uuid.uuid4().hex}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "modules")]
    sys.modules[package_name] = package

    numpy = types.ModuleType("numpy")
    numpy.uint8 = object()
    numpy.clip = lambda value, low, high: value
    sys.modules["numpy"] = numpy

    piexif = types.ModuleType("piexif")
    piexif.ExifIFD = types.SimpleNamespace(UserComment=1)
    exif_inserts = []
    piexif.dump = lambda value: value
    piexif.insert = lambda data, path: exif_inserts.append((data, path))
    helper = types.ModuleType("piexif.helper")
    helper.UserComment = types.SimpleNamespace(
        dump=lambda value, encoding=None: value
    )
    piexif.helper = helper
    sys.modules["piexif"] = piexif
    sys.modules["piexif.helper"] = helper

    class FakePngInfo:
        def __init__(self):
            self.chunks = []

        def add_text(self, key, value):
            self.chunks.append((key, value))

    class FakeImage:
        def save(self, path, *args, pnginfo=None, **kwargs):
            saved_pnginfo.append(pnginfo)

    pil = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.fromarray = lambda value: FakeImage()
    png_module = types.ModuleType("PIL.PngImagePlugin")
    png_module.PngInfo = FakePngInfo
    pil.Image = image_module
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = image_module
    sys.modules["PIL.PngImagePlugin"] = png_module

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(output_dir)
    folder_paths.get_save_image_path = lambda *args: (
        str(output_dir), "image", 1, "", "image"
    )
    sys.modules["folder_paths"] = folder_paths

    hook = types.ModuleType(f"{package_name}.hook")
    hook.current_save_image_node_id = "save"
    sys.modules[hook.__name__] = hook

    capture = types.ModuleType(f"{package_name}.capture")

    class FakeCapture:
        @staticmethod
        def gen_parameters_str(info):
            return f"{info.get('Positive prompt', '')}\nSteps: {info.get('Steps', 1)}"

    capture.Capture = FakeCapture
    sys.modules[capture.__name__] = capture

    trace = types.ModuleType(f"{package_name}.trace")
    trace.Trace = type("Trace", (), {})
    sys.modules[trace.__name__] = trace

    utils = types.ModuleType(f"{package_name}.utils")
    utils.__path__ = []
    sys.modules[utils.__name__] = utils
    log = types.ModuleType(f"{package_name}.utils.log")
    log.print_warning = lambda *args, **kwargs: None
    sys.modules[log.__name__] = log

    nodes_package = types.ModuleType(f"{package_name}.nodes")
    nodes_package.__path__ = [str(ROOT / "modules" / "nodes")]
    sys.modules[nodes_package.__name__] = nodes_package
    name = f"{package_name}.nodes.node"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "modules" / "nodes" / "node.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module._test_exif_inserts = exif_inserts
    return module


class NodeMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.saved_pnginfo = []
        self.node_module = _load_node_module(
            Path(self.temp_dir.name), self.saved_pnginfo
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_batch_images_receive_fresh_pnginfo_objects(self):
        node = self.node_module.SaveImageWithMetaData()

        async def fake_gen_pnginfo(prompt, prefer_nearest, batch_index=0):
            return {"Positive prompt": f"prompt {batch_index}", "Steps": 8}

        node.gen_pnginfo = fake_gen_pnginfo
        asyncio.run(node.save_images(
            [_FakeTensor(), _FakeTensor()],
            output_format="png",
            metadata_scope="parameters_only",
        ))

        self.assertEqual(len(self.saved_pnginfo), 2)
        self.assertIsNot(self.saved_pnginfo[0], self.saved_pnginfo[1])
        self.assertEqual(
            [key for key, _ in self.saved_pnginfo[0].chunks], ["parameters"]
        )
        self.assertEqual(
            [key for key, _ in self.saved_pnginfo[1].chunks], ["parameters"]
        )
        self.assertIn("prompt 0", self.saved_pnginfo[0].chunks[0][1])
        self.assertIn("prompt 1", self.saved_pnginfo[1].chunks[0][1])

    def test_none_scope_ignores_extra_metadata_without_crashing(self):
        node = self.node_module.SaveImageWithMetaData()
        asyncio.run(node.save_images(
            [_FakeTensor()],
            output_format="png",
            metadata_scope="none",
            extra_metadata={"ignored": "value"},
        ))

        self.assertEqual(self.saved_pnginfo, [None])

    def test_rgthree_context_input_is_optional_and_typed(self):
        optional_inputs = self.node_module.SaveImageWithMetaData.INPUT_TYPES()[
            "optional"
        ]
        context_input = optional_inputs["context"]

        self.assertEqual(context_input[0], "RGTHREE_CONTEXT")
        self.assertLess(
            list(optional_inputs).index("extra_metadata"),
            list(optional_inputs).index("context"),
        )

    def test_rgthree_context_values_override_graph_metadata(self):
        node = self.node_module.SaveImageWithMetaData()
        baseline = {
            "Positive prompt": "graph positive",
            "Negative prompt": "graph negative",
            "Steps": "8",
            "CFG scale": "1",
            "Seed": "11",
            "Sampler": "Euler",
            "Schedule type": "Normal",
            "Size": "512x768",
            "Model": "old-model",
            "Model hash": "stale-hash",
            "Denoising strength": 0.35,
        }
        context = {
            "model": object(),
            "clip": object(),
            "vae": object(),
            "positive": object(),
            "negative": object(),
            "seed": 123456,
            "steps": 24,
            "cfg": 4.0,
            "ckpt_name": "models/new-model.safetensors",
            "sampler": "euler_ancestral",
            "scheduler": "karras",
            "clip_width": 1024,
            "clip_height": 1536,
            "text_pos_g": "context positive",
            "text_pos_l": "context positive",
            "text_neg_g": "context negative global",
            "text_neg_l": "context negative local",
        }

        merged = node.apply_rgthree_context(baseline, context)

        self.assertEqual(merged["Positive prompt"], "graph positive")
        self.assertEqual(
            merged["Negative prompt"],
            "graph negative",
        )
        self.assertEqual(merged["Steps"], "24")
        self.assertEqual(merged["CFG scale"], "4")
        self.assertEqual(merged["Seed"], "123456")
        self.assertEqual(merged["Sampler"], "euler_ancestral")
        self.assertEqual(merged["Schedule type"], "karras")
        self.assertEqual(merged["Size"], "512x768")
        self.assertEqual(merged["Model"], "new-model")
        self.assertNotIn("Model hash", merged)
        self.assertEqual(merged["Denoising strength"], 0.35)
        self.assertNotIn("model", merged)
        self.assertNotIn("positive", merged)

        prompt_fallback = node.apply_rgthree_context(
            {"Steps": "24"},
            {
                "text_pos_g": "context positive",
                "text_pos_l": "context positive",
                "text_neg_g": "context negative global",
                "text_neg_l": "context negative local",
            },
        )
        self.assertEqual(
            prompt_fallback["Positive prompt"], "context positive"
        )
        self.assertEqual(
            prompt_fallback["Negative prompt"],
            "context negative global\ncontext negative local",
        )

    def test_rgthree_clip_dimensions_only_fill_a_missing_size(self):
        node = self.node_module.SaveImageWithMetaData()

        merged = node.apply_rgthree_context(
            {"Steps": "20"},
            {"clip_width": 1024, "clip_height": 1536},
        )

        self.assertEqual(merged["Size"], "1024x1536")

    def test_empty_rgthree_context_values_keep_graph_fallbacks(self):
        node = self.node_module.SaveImageWithMetaData()
        baseline = {
            "Positive prompt": "graph positive",
            "Steps": "18",
            "Seed": "987",
            "Size": "640x960",
        }
        context = {
            "positive": object(),
            "negative": object(),
            "seed": None,
            "steps": None,
            "text_pos_g": "",
            "text_pos_l": None,
            "clip_width": None,
            "clip_height": None,
        }

        merged = node.apply_rgthree_context(baseline, context)

        self.assertEqual(merged, baseline)

    def test_rgthree_sampler_context_is_applied_before_parameters_are_written(self):
        node = self.node_module.SaveImageWithMetaData()

        async def fake_gen_pnginfo(prompt, prefer_nearest, batch_index=0):
            return {"Positive prompt": "graph prompt", "Steps": 8}

        node.gen_pnginfo = fake_gen_pnginfo
        asyncio.run(node.save_images(
            [_FakeTensor()],
            output_format="png",
            metadata_scope="parameters_only",
            context={"steps": 32, "text_pos_g": "context prompt"},
        ))

        parameters = self.saved_pnginfo[0].chunks[0][1]
        self.assertIn("graph prompt", parameters)
        self.assertIn("Steps: 32", parameters)

    def test_connected_context_keeps_graph_capture_on_the_image_branch(self):
        starts = []

        async def fake_get_inputs(batch_index=0):
            return {}

        def fake_trace(start_node_id, prompt):
            starts.append(start_node_id)
            return {
                str(start_node_id): (0, prompt[str(start_node_id)]["class_type"])
            }

        capture = self.node_module.Capture
        capture.get_inputs = staticmethod(fake_get_inputs)
        def fake_gen_pnginfo_dict(*args, **kwargs):
            positive, negative = kwargs.get("prompt_overrides") or (
                "image-branch prompt",
                "image-branch negative",
            )
            return {
                "Positive prompt": positive,
                "Negative prompt": negative,
                "Steps": "20",
            }

        capture.gen_pnginfo_dict = staticmethod(fake_gen_pnginfo_dict)
        capture.resolve_context_prompts = staticmethod(
            lambda prompt, context_node_id, batch_index=0: (
                "context-branch prompt",
                "context-branch negative",
            )
        )
        trace = self.node_module.Trace
        trace.trace = staticmethod(fake_trace)
        trace.filter_inputs_by_trace_tree = staticmethod(
            lambda inputs, trace_tree, prefer_nearest: {}
        )
        trace.find_sampler_node_id = staticmethod(lambda trace_tree: None)

        self.node_module.hook.current_save_image_node_id = "save"
        prompt = {
            "save": {
                "class_type": "SaveImageWithMetaData",
                "inputs": {
                    "images": ["image", 0],
                    "context": ["detailer_context", 0],
                },
            },
            "image": {"class_type": "VAEDecode", "inputs": {}},
            "detailer_context": {
                "class_type": "Context Big (rgthree)",
                "inputs": {},
            },
        }

        pnginfo = asyncio.run(
            self.node_module.SaveImageWithMetaData.gen_pnginfo(
                prompt, prefer_nearest=True
            )
        )

        self.assertEqual(starts[0], "image")
        self.assertEqual(pnginfo["Positive prompt"], "context-branch prompt")
        self.assertEqual(
            pnginfo["Negative prompt"], "context-branch negative"
        )

        starts.clear()
        prompt["save"]["inputs"].pop("context")
        asyncio.run(
            self.node_module.SaveImageWithMetaData.gen_pnginfo(
                prompt, prefer_nearest=True
            )
        )
        self.assertEqual(starts[0], "save")

    def test_context_list_values_are_selected_per_batch_image(self):
        node = self.node_module.SaveImageWithMetaData()

        async def fake_gen_pnginfo(prompt, prefer_nearest, batch_index=0):
            return {
                "Positive prompt": f"prompt {batch_index}",
                "Steps": 8,
            }

        node.gen_pnginfo = fake_gen_pnginfo
        asyncio.run(node.save_images(
            [_FakeTensor(), _FakeTensor()],
            output_format="png",
            metadata_scope="parameters_only",
            context=[{"steps": 20}, {"steps": 30}],
        ))

        self.assertIn("Steps: 20", self.saved_pnginfo[0].chunks[0][1])
        self.assertIn("Steps: 30", self.saved_pnginfo[1].chunks[0][1])

    def test_parameters_only_never_embeds_prompt_or_workflow_on_fallback(self):
        node = self.node_module.SaveImageWithMetaData()

        async def fake_gen_pnginfo(prompt, prefer_nearest, batch_index=0):
            return {}

        node.gen_pnginfo = fake_gen_pnginfo
        fallback_parameters = "fallback prompt\nSteps: 12, Sampler: euler"
        asyncio.run(node.save_images(
            [_FakeTensor()],
            output_format="png",
            metadata_scope="parameters_only",
            prompt={"secret_graph": {"class_type": "KSampler"}},
            extra_pnginfo={
                "workflow": {"nodes": ["large workflow"]},
                "parameters": fallback_parameters,
            },
            extra_metadata={"custom": "must not be included"},
        ))

        self.assertEqual(
            self.saved_pnginfo[0].chunks,
            [("parameters", fallback_parameters)],
        )

    def test_parameters_only_stays_empty_when_no_parameters_are_available(self):
        node = self.node_module.SaveImageWithMetaData()

        async def fake_gen_pnginfo(prompt, prefer_nearest, batch_index=0):
            return {}

        node.gen_pnginfo = fake_gen_pnginfo
        asyncio.run(node.save_images(
            [_FakeTensor()],
            output_format="png",
            metadata_scope="parameters_only",
            prompt={"secret_graph": {"class_type": "KSampler"}},
            extra_pnginfo={"workflow": {"nodes": ["large workflow"]}},
        ))

        self.assertEqual(self.saved_pnginfo[0].chunks, [])

    def test_jpg_and_webp_exif_respect_metadata_scope(self):
        expected_exif = {
            "full": True,
            "parameters_only": True,
            "default": False,
            "workflow_only": False,
            "none": False,
        }

        for output_format in ("jpg", "webp"):
            for scope, expected in expected_exif.items():
                with self.subTest(output_format=output_format, scope=scope):
                    self.node_module._test_exif_inserts.clear()
                    node = self.node_module.SaveImageWithMetaData()

                    async def fake_gen_pnginfo(
                        prompt, prefer_nearest, batch_index=0
                    ):
                        return {
                            "Positive prompt": "private prompt",
                            "Steps": 16,
                            "Seed": 123,
                        }

                    node.gen_pnginfo = fake_gen_pnginfo
                    asyncio.run(node.save_images(
                        [_FakeTensor()],
                        filename_prefix="%seed%",
                        output_format=output_format,
                        metadata_scope=scope,
                    ))

                    self.assertEqual(
                        bool(self.node_module._test_exif_inserts), expected
                    )

    def test_jpg_and_webp_parameters_only_use_raw_fallback(self):
        fallback = "fallback prompt\nSteps: 12, Sampler: Euler"

        for output_format in ("jpg", "webp"):
            with self.subTest(output_format=output_format):
                self.node_module._test_exif_inserts.clear()
                node = self.node_module.SaveImageWithMetaData()

                async def fake_gen_pnginfo(prompt, prefer_nearest, batch_index=0):
                    return {}

                node.gen_pnginfo = fake_gen_pnginfo
                asyncio.run(node.save_images(
                    [_FakeTensor()],
                    output_format=output_format,
                    metadata_scope="parameters_only",
                    extra_pnginfo={"parameters": fallback},
                ))

                exif_data, _ = self.node_module._test_exif_inserts[0]
                self.assertEqual(exif_data["Exif"][1], fallback)

    def test_full_scope_stores_fallback_parameters_without_json_quoting(self):
        node = self.node_module.SaveImageWithMetaData()

        async def fake_gen_pnginfo(prompt, prefer_nearest, batch_index=0):
            return {}

        node.gen_pnginfo = fake_gen_pnginfo
        fallback = "fallback prompt\nSteps: 12, Sampler: Euler"
        asyncio.run(node.save_images(
            [_FakeTensor()],
            output_format="png",
            metadata_scope="full",
            extra_pnginfo={"parameters": fallback},
        ))

        self.assertEqual(
            self.saved_pnginfo[0].chunks,
            [("parameters", fallback)],
        )

    def test_full_default_and_workflow_scopes_remain_distinct(self):
        expected_keys = {
            "full": ["parameters", "prompt", "workflow", "other", "custom"],
            "default": ["prompt", "workflow", "other"],
            "workflow_only": ["workflow"],
        }

        for scope, expected in expected_keys.items():
            with self.subTest(scope=scope):
                self.saved_pnginfo.clear()
                node = self.node_module.SaveImageWithMetaData()

                async def fake_gen_pnginfo(prompt, prefer_nearest, batch_index=0):
                    return {"Positive prompt": "test prompt", "Steps": 8}

                node.gen_pnginfo = fake_gen_pnginfo
                asyncio.run(node.save_images(
                    [_FakeTensor()],
                    output_format="png",
                    metadata_scope=scope,
                    prompt={"sampler": {"class_type": "KSampler"}},
                    extra_pnginfo={
                        "workflow": {"nodes": []},
                        "other": {"value": 1},
                    },
                    extra_metadata={"custom": "full scope only"},
                ))

                self.assertEqual(
                    [key for key, _ in self.saved_pnginfo[0].chunks],
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
