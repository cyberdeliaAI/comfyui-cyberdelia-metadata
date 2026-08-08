import ast
import re
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
        self.assertIn("OWNED_IDS.has(properties.cnr_id)", source)
        self.assertIn("OWNED_IDS.has(properties.aux_id)", source)
        self.assertIn('"comfyui-cyberdelia-metadata"', source)
        self.assertIn(
            '"cyberdeliaAI/comfyui-cyberdelia-metadata"', source
        )
        self.assertIn(
            '"revived_comfyui_image_metadata_extension"', source
        )
        self.assertIn("beforeConfigureGraph(graphData)", source)

    def test_migration_versions_match_pyproject(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
        self.assertIsNotNone(match)
        version = match.group(1)

        frontend = (ROOT / "web" / "workflow_migration.js").read_text(
            encoding="utf-8"
        )
        cli = (ROOT / "scripts" / "migrate_workflows.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(f'const PACK_VERSION = "{version}";', frontend)
        self.assertIn(f'PACK_VERSION = "{version}"', cli)


if __name__ == "__main__":
    unittest.main()
