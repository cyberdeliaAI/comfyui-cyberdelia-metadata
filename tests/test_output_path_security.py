import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


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


def _load_node_module(output_dir, saved_paths):
    package_name = f"_metadata_path_test_{uuid.uuid4().hex}"
    replacements = {}

    def install(name, module):
        replacements[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = module

    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "modules")]
    install(package_name, package)

    numpy = types.ModuleType("numpy")
    numpy.uint8 = object()
    numpy.clip = lambda value, low, high: value
    install("numpy", numpy)

    piexif = types.ModuleType("piexif")
    piexif.ExifIFD = types.SimpleNamespace(UserComment=1)
    piexif.dump = lambda value: value
    piexif.insert = lambda data, path: None
    helper = types.ModuleType("piexif.helper")
    helper.UserComment = types.SimpleNamespace(
        dump=lambda value, encoding=None: value
    )
    piexif.helper = helper
    install("piexif", piexif)
    install("piexif.helper", helper)

    class FakePngInfo:
        def add_text(self, key, value):
            pass

    class FakeImage:
        def save(self, path, *args, **kwargs):
            saved_paths.append(Path(path).resolve(strict=False))

    pil = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.fromarray = lambda value: FakeImage()
    png_module = types.ModuleType("PIL.PngImagePlugin")
    png_module.PngInfo = FakePngInfo
    pil.Image = image_module
    install("PIL", pil)
    install("PIL.Image", image_module)
    install("PIL.PngImagePlugin", png_module)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(output_dir)
    folder_paths.get_save_image_path = lambda *args: (
        str(output_dir), "image", 1, "", "image"
    )
    install("folder_paths", folder_paths)

    hook = types.ModuleType(f"{package_name}.hook")
    hook.current_save_image_node_id = "save"
    install(hook.__name__, hook)

    capture = types.ModuleType(f"{package_name}.capture")

    class FakeCapture:
        @staticmethod
        def gen_parameters_str(info):
            return ""

    capture.Capture = FakeCapture
    install(capture.__name__, capture)

    trace = types.ModuleType(f"{package_name}.trace")
    trace.Trace = type("Trace", (), {})
    install(trace.__name__, trace)

    utils = types.ModuleType(f"{package_name}.utils")
    utils.__path__ = []
    install(utils.__name__, utils)
    log = types.ModuleType(f"{package_name}.utils.log")
    log.print_warning = lambda *args, **kwargs: None
    install(log.__name__, log)

    nodes_package = types.ModuleType(f"{package_name}.nodes")
    nodes_package.__path__ = [str(ROOT / "modules" / "nodes")]
    install(nodes_package.__name__, nodes_package)

    module_name = f"{package_name}.nodes.node"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "modules" / "nodes" / "node.py"
    )
    module = importlib.util.module_from_spec(spec)
    install(module_name, module)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in replacements.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module


class OutputPathSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temp_dir.name)
        self.output_dir = self.temp_root / "output"
        self.output_dir.mkdir()
        self.saved_paths = []
        self.node_module = _load_node_module(
            self.output_dir, self.saved_paths
        )
        self.node = self.node_module.SaveImageWithMetaData()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save(self, **kwargs):
        defaults = {
            "output_format": "png",
            "metadata_scope": "none",
            "include_batch_num": False,
        }
        defaults.update(kwargs)
        return asyncio.run(self.node.save_images([_FakeTensor()], **defaults))

    def _set_core_path(self, folder, filename="safe_name"):
        self.node_module.folder_paths.get_save_image_path = lambda *args: (
            str(folder), filename, 1, "", filename
        )

    def test_preserves_nested_directory_returned_by_comfyui(self):
        nested = self.output_dir / "portraits" / "2026"
        self._set_core_path(nested)

        self._save(filename_prefix="portraits/2026/safe_name")

        self.assertEqual(
            self.saved_paths,
            [(nested / "safe_name.png").resolve(strict=False)],
        )
        self.assertTrue(nested.is_dir())

    def test_custom_nested_directory_keeps_core_safe_basename(self):
        self._set_core_path(self.output_dir / "ignored", "core_safe_name")

        self._save(
            filename_prefix="raw/prefix/must_not_be_reused",
            subdirectory_name="gallery/approved",
        )

        expected = self.output_dir / "gallery" / "approved" / "core_safe_name.png"
        self.assertEqual(self.saved_paths, [expected.resolve(strict=False)])

    def test_rejects_relative_absolute_and_windows_directory_escapes(self):
        self._set_core_path(self.output_dir)
        bad_subdirectories = (
            "../outside",
            "nested/../../outside",
            "..\\outside",
            str(self.temp_root / "absolute-outside"),
        )

        for subdirectory in bad_subdirectories:
            with self.subTest(subdirectory=subdirectory):
                with self.assertRaises(ValueError):
                    self._save(subdirectory_name=subdirectory)

        self.assertEqual(self.saved_paths, [])

    def test_rejects_output_folder_or_filename_outside_core_contract(self):
        outside = self.temp_root / "outside"
        self._set_core_path(outside)
        with self.assertRaises(ValueError):
            self._save()

        self._set_core_path(self.output_dir, "../outside")
        with self.assertRaises(ValueError):
            self._save()

        self.assertEqual(self.saved_paths, [])

    def test_rejects_subdirectory_that_escapes_through_symlink(self):
        outside = self.temp_root / "outside"
        outside.mkdir()
        link = self.output_dir / "linked"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks are unavailable: {exc}")
        self._set_core_path(self.output_dir)

        with self.assertRaises(ValueError):
            self._save(subdirectory_name="linked/nested")

        self.assertEqual(self.saved_paths, [])

    def test_workflow_sidecar_is_written_beside_image_inside_output(self):
        nested = self.output_dir / "exports" / "nested"
        self._set_core_path(nested, "with_metadata")
        workflow = {"nodes": [{"id": 1}]}

        self._save(
            output_format="png_with_json",
            extra_pnginfo={"workflow": workflow},
        )

        image_path = nested / "with_metadata.png"
        sidecar_path = nested / "with_metadata.json"
        self.assertEqual(
            self.saved_paths, [image_path.resolve(strict=False)]
        )
        self.assertEqual(json.loads(sidecar_path.read_text()), workflow)
        sidecar_path.resolve(strict=False).relative_to(self.output_dir.resolve())


if __name__ == "__main__":
    unittest.main()
