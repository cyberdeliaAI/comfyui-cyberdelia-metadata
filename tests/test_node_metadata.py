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
    piexif.dump = lambda value: b""
    piexif.insert = lambda data, path: None
    helper = types.ModuleType("piexif.helper")
    helper.UserComment = types.SimpleNamespace(dump=lambda value, encoding=None: b"")
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


if __name__ == "__main__":
    unittest.main()
