import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NodeRegistrationTests(unittest.TestCase):
    def test_backend_keys_are_literal_unique_cyberdelia_ids(self):
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        mapping_node = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name)
                and target.id == "NODE_CLASS_MAPPINGS"
                for target in node.targets
            ):
                mapping_node = node.value
                break

        self.assertIsInstance(mapping_node, ast.Dict)
        keys = {ast.literal_eval(key) for key in mapping_node.keys}
        self.assertEqual(
            keys,
            {
                "CyberdeliaSaveImageWithMetaData",
                "CyberdeliaCreateExtraMetaData",
            },
        )
        self.assertNotIn("SaveImageWithMetaData", keys)
        self.assertNotIn("CreateExtraMetaData", keys)

    def test_frontend_migration_is_pack_scoped_and_runs_before_configure(self):
        source = (ROOT / "web" / "workflow_migration.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("properties.cnr_id === PACK_ID", source)
        self.assertIn("properties.aux_id === PACK_ID", source)
        self.assertIn("beforeConfigureGraph(graphData)", source)


if __name__ == "__main__":
    unittest.main()
