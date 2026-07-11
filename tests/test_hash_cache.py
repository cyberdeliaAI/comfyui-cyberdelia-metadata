import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _load_hash_module(cache_dir):
    """Load hash.py without importing ComfyUI or sharing module globals."""
    package_name = f"_metadata_hash_test_{uuid.uuid4().hex}"

    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "modules")]
    sys.modules[package_name] = package

    config = types.ModuleType(f"{package_name}.config")
    config.NODE_CACHE_DIR = str(cache_dir)
    sys.modules[config.__name__] = config

    utils = types.ModuleType(f"{package_name}.utils")
    utils.__path__ = [str(ROOT / "modules" / "utils")]
    sys.modules[utils.__name__] = utils

    log = types.ModuleType(f"{package_name}.utils.log")
    log.print_warning = lambda message: None
    log.print_error = lambda message: None
    sys.modules[log.__name__] = log

    name = f"{package_name}.utils.hash"
    spec = importlib.util.spec_from_file_location(name, ROOT / "modules" / "utils" / "hash.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, package_name


def _short_hash(data):
    return hashlib.sha256(data).hexdigest()[:10]


class HashCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.cache_dir = self.root / "cache"
        self.loaded_packages = []
        self.hash_module = self._load_module()

    def tearDown(self):
        for package_name in self.loaded_packages:
            for name in tuple(sys.modules):
                if name == package_name or name.startswith(f"{package_name}."):
                    sys.modules.pop(name, None)
        self.temp_dir.cleanup()

    def _load_module(self):
        module, package_name = _load_hash_module(self.cache_dir)
        self.loaded_packages.append(package_name)
        return module

    def _write(self, relative_path, data):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_same_basename_in_different_directories_has_distinct_cache_entries(self):
        first_data = b"first model"
        second_data = b"other model"
        first = self._write("a/shared.safetensors", first_data)
        second = self._write("b/shared.safetensors", second_data)

        # Make both stat signatures identical so only their paths distinguish them.
        common_mtime_ns = max(first.stat().st_mtime_ns, second.stat().st_mtime_ns) + 1_000_000
        for path in (first, second):
            os.utime(path, ns=(common_mtime_ns, common_mtime_ns))

        self.assertEqual(self.hash_module.calc_hash(first), _short_hash(first_data))
        self.assertEqual(self.hash_module.calc_hash(second), _short_hash(second_data))
        self.assertEqual(
            set(self.hash_module.cache_model_hash),
            {os.path.realpath(first), os.path.realpath(second)},
        )

    def test_memory_hit_is_invalidated_when_mtime_ns_changes(self):
        path = self._write("model.safetensors", b"alpha")

        with mock.patch.object(
            self.hash_module, "_hash_file", wraps=self.hash_module._hash_file
        ) as hash_file:
            self.assertEqual(self.hash_module.calc_hash(path), _short_hash(b"alpha"))
            self.assertEqual(self.hash_module.calc_hash(path), _short_hash(b"alpha"))
            self.assertEqual(hash_file.call_count, 1)

            previous = path.stat()
            path.write_bytes(b"bravo")
            changed_mtime_ns = previous.st_mtime_ns + 1_000_000_000
            os.utime(path, ns=(previous.st_atime_ns, changed_mtime_ns))

            self.assertEqual(self.hash_module.calc_hash(path), _short_hash(b"bravo"))
            self.assertEqual(hash_file.call_count, 2)

    def test_file_size_invalidates_cache_even_when_mtime_is_unchanged(self):
        path = self._write("model.safetensors", b"small")
        self.assertEqual(self.hash_module.calc_hash(path), _short_hash(b"small"))
        previous = path.stat()

        path.write_bytes(b"much larger")
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))

        self.assertEqual(self.hash_module.calc_hash(path), _short_hash(b"much larger"))

    def test_new_disk_record_is_reused_after_module_reload(self):
        data = b"persistent model data"
        path = self._write("model.safetensors", data)
        expected = _short_hash(data)
        self.assertEqual(self.hash_module.calc_hash(path), expected)

        canonical_path = os.path.realpath(path)
        record = self.hash_module._disk_cache[canonical_path]
        self.assertEqual(record["file_modification_time_ns"], path.stat().st_mtime_ns)
        self.assertEqual(record["file_size"], path.stat().st_size)

        reloaded = self._load_module()
        with mock.patch.object(
            reloaded, "_hash_file", side_effect=AssertionError("disk cache was not used")
        ):
            self.assertEqual(reloaded.calc_hash(path), expected)

    def test_exact_path_legacy_record_is_migrated(self):
        data = b"legacy model"
        path = self._write("model.safetensors", data)
        expected = _short_hash(data)
        canonical_path = os.path.realpath(path)
        self.cache_dir.mkdir(parents=True)
        (self.cache_dir / "model_hash_cache.json").write_text(
            json.dumps(
                {
                    canonical_path: {
                        "file_hash": expected,
                        "file_modification_date": path.stat().st_mtime,
                    }
                }
            ),
            encoding="utf-8",
        )

        reloaded = self._load_module()
        with mock.patch.object(
            reloaded, "_hash_file", side_effect=AssertionError("legacy cache was not used")
        ):
            self.assertEqual(reloaded.calc_hash(path), expected)

        migrated = reloaded._disk_cache[canonical_path]
        self.assertEqual(migrated["file_modification_time_ns"], path.stat().st_mtime_ns)
        self.assertEqual(migrated["file_size"], path.stat().st_size)

    def test_ambiguous_basename_legacy_record_is_recomputed(self):
        data = b"current model"
        path = self._write("models/shared.safetensors", data)
        self.cache_dir.mkdir(parents=True)
        (self.cache_dir / "model_hash_cache.json").write_text(
            json.dumps(
                {
                    path.name: {
                        "file_hash": "incorrect",
                        "file_modification_date": path.stat().st_mtime,
                    }
                }
            ),
            encoding="utf-8",
        )

        reloaded = self._load_module()
        self.assertEqual(reloaded.calc_hash(path), _short_hash(data))
        self.assertIn(os.path.realpath(path), reloaded._disk_cache)

    def test_mod_time_helper_does_not_cache_old_value(self):
        path = self._write("model.safetensors", b"data")
        original = self.hash_module.get_file_mod_time(path)
        stat_result = path.stat()
        os.utime(
            path,
            ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000),
        )
        self.assertNotEqual(self.hash_module.get_file_mod_time(path), original)

    def test_hashes_are_read_in_large_chunks(self):
        self.assertGreaterEqual(self.hash_module.HASH_READ_CHUNK_SIZE, 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
