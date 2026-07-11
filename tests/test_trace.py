import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_trace():
    package_name = "_metadata_trace_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "modules")]
    sys.modules[package_name] = package

    defs = types.ModuleType(f"{package_name}.defs")
    defs.__path__ = [str(ROOT / "modules" / "defs")]
    sys.modules[defs.__name__] = defs
    _load_module(
        f"{package_name}.defs.samplers", ROOT / "modules" / "defs" / "samplers.py"
    )

    utils = types.ModuleType(f"{package_name}.utils")
    utils.__path__ = []
    sys.modules[utils.__name__] = utils
    log = types.ModuleType(f"{package_name}.utils.log")
    log.print_warning = lambda *args, **kwargs: None
    sys.modules[log.__name__] = log

    return _load_module(f"{package_name}.trace", ROOT / "modules" / "trace.py")


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace_module = _load_trace()
        cls.Trace = cls.trace_module.Trace

    def setUp(self):
        self.Trace._trace_cache.clear()

    def test_follows_source_node_but_not_output_slot(self):
        prompt = {
            "save": {
                "class_type": "SaveImage",
                "inputs": {"images": ["producer", 0]},
            },
            "producer": {"class_type": "Producer", "inputs": {}},
            # This id deliberately matches the link's output slot.
            "0": {"class_type": "Unrelated", "inputs": {}},
        }

        self.assertEqual(
            self.Trace.trace("save", prompt),
            {
                "save": (0, "SaveImage"),
                "producer": (1, "Producer"),
            },
        )

    def test_does_not_traverse_literal_lists(self):
        prompt = {
            "save": {
                "class_type": "SaveImage",
                "inputs": {
                    "labels": ["unrelated", "literal-value"],
                    "dimensions": [512, 768],
                },
            },
            "unrelated": {"class_type": "Unrelated", "inputs": {}},
        }

        self.assertEqual(
            self.Trace.trace("save", prompt),
            {"save": (0, "SaveImage")},
        )

    def test_accepts_integer_source_ids_and_mapping_links(self):
        prompt = {
            "save": {
                "class_type": "SaveImage",
                "inputs": {"images": {"link": [12, 1]}},
            },
            "12": {"class_type": "Producer", "inputs": {}},
        }

        self.assertEqual(
            self.Trace.trace("save", prompt),
            {
                "save": (0, "SaveImage"),
                "12": (1, "Producer"),
            },
        )

    def test_cache_distinguishes_start_nodes_with_same_reachable_set(self):
        prompt = {
            "a": {"class_type": "NodeA", "inputs": {"source": ["b", 0]}},
            "b": {"class_type": "NodeB", "inputs": {"source": ["a", 0]}},
        }

        from_a = self.Trace.trace("a", prompt)
        from_b = self.Trace.trace("b", prompt)

        self.assertEqual(from_a, {"a": (0, "NodeA"), "b": (1, "NodeB")})
        self.assertEqual(from_b, {"b": (0, "NodeB"), "a": (1, "NodeA")})

    def test_cache_distinguishes_different_prompt_edges(self):
        first_prompt = {
            "save": {"class_type": "SaveImage", "inputs": {"image": ["a", 0]}},
            "a": {"class_type": "NodeA", "inputs": {"source": ["b", 0]}},
            "b": {"class_type": "NodeB", "inputs": {}},
        }
        second_prompt = {
            "save": {"class_type": "SaveImage", "inputs": {"image": ["b", 0]}},
            "a": {"class_type": "NodeA", "inputs": {}},
            "b": {"class_type": "NodeB", "inputs": {"source": ["a", 0]}},
        }

        first = self.Trace.trace("save", first_prompt)
        second = self.Trace.trace("save", second_prompt)

        self.assertEqual(first["a"][0], 1)
        self.assertEqual(first["b"][0], 2)
        self.assertEqual(second["b"][0], 1)
        self.assertEqual(second["a"][0], 2)

    def test_unknown_start_returns_empty_trace(self):
        self.assertEqual(
            self.Trace.trace("missing", {"save": {"class_type": "SaveImage"}}),
            {},
        )


if __name__ == "__main__":
    unittest.main()
