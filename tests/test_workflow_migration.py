import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "workflow_migration",
    ROOT / "scripts" / "migrate_workflows.py",
)
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


class WorkflowMigrationTests(unittest.TestCase):
    def test_migrates_ui_api_and_nested_nodes(self):
        document = {
            "nodes": [
                {
                    "id": 1,
                    "type": "SaveImageWithMetaData",
                    "properties": {
                        "cnr_id": "comfyui-cyberdelia-metadata",
                        "ver": "1.0.13",
                        "Node name for S&R": "SaveImageWithMetaData",
                    },
                    "widgets_values": ["keep me"],
                }
            ],
            "definitions": {
                "subgraphs": [
                    {
                        "nodes": [
                            {
                                "id": 2,
                                "type": "CreateExtraMetaData",
                                "properties": {
                                    "Node name for S&R": "CreateExtraMetaData"
                                },
                            }
                        ]
                    }
                ]
            },
            "api": {
                "3": {
                    "class_type": "SaveImageWithMetaData",
                    "inputs": {"images": ["2", 0]},
                }
            },
            "unrelated": {
                "type": "OtherNode",
                "label": "SaveImageWithMetaData",
            },
        }

        count = MIGRATION.migrate_document(document, all_legacy=True)

        self.assertEqual(count, 3)
        self.assertEqual(
            document["nodes"][0]["type"],
            "CyberdeliaSaveImageWithMetaData",
        )
        self.assertEqual(
            document["nodes"][0]["properties"]["Node name for S&R"],
            "CyberdeliaSaveImageWithMetaData",
        )
        self.assertEqual(document["nodes"][0]["properties"]["ver"], "2.0.1")
        self.assertEqual(document["nodes"][0]["widgets_values"], ["keep me"])
        self.assertEqual(
            document["definitions"]["subgraphs"][0]["nodes"][0]["type"],
            "CyberdeliaCreateExtraMetaData",
        )
        self.assertEqual(
            document["api"]["3"]["class_type"],
            "CyberdeliaSaveImageWithMetaData",
        )
        self.assertEqual(document["api"]["3"]["inputs"]["images"], ["2", 0])
        self.assertEqual(document["unrelated"]["label"], "SaveImageWithMetaData")
        self.assertEqual(MIGRATION.migrate_document(document, all_legacy=True), 0)

    def test_default_migration_recognizes_owned_cnr_and_aux_ids(self):
        owned_ids = (
            "comfyui-cyberdelia-metadata",
            "cyberdeliaAI/comfyui-cyberdelia-metadata",
            "revived_comfyui_image_metadata_extension",
        )

        for field in ("cnr_id", "aux_id"):
            for owned_id in owned_ids:
                with self.subTest(field=field, owned_id=owned_id):
                    document = {
                        "nodes": [
                            {
                                "type": "CreateExtraMetaData",
                                "properties": {field: owned_id},
                            }
                        ]
                    }

                    self.assertEqual(MIGRATION.migrate_document(document), 1)
                    node = document["nodes"][0]
                    self.assertEqual(
                        node["type"], "CyberdeliaCreateExtraMetaData"
                    )
                    self.assertEqual(
                        node["properties"]["cnr_id"],
                        "comfyui-cyberdelia-metadata",
                    )
                    self.assertEqual(node["properties"]["ver"], "2.0.1")

    def test_valid_aux_id_is_not_masked_by_an_unrelated_cnr_id(self):
        document = {
            "nodes": [
                {
                    "type": "SaveImageWithMetaData",
                    "properties": {
                        "cnr_id": "comfyui-saveimagewithmetadata",
                        "aux_id": (
                            "cyberdeliaAI/comfyui-cyberdelia-metadata"
                        ),
                    },
                }
            ]
        }

        self.assertEqual(MIGRATION.migrate_document(document), 1)
        self.assertEqual(
            document["nodes"][0]["type"],
            "CyberdeliaSaveImageWithMetaData",
        )

    def test_default_migration_only_updates_tagged_cyberdelia_nodes(self):
        document = {
            "nodes": [
                {
                    "type": "SaveImageWithMetaData",
                    "properties": {
                        "cnr_id": "comfyui-cyberdelia-metadata",
                    },
                },
                {
                    "type": "SaveImageWithMetaData",
                    "properties": {
                        "cnr_id": "comfyui-saveimagewithmetadata",
                    },
                },
                {"type": "SaveImageWithMetaData"},
            ],
            "api": {"1": {"class_type": "SaveImageWithMetaData"}},
        }

        self.assertEqual(MIGRATION.migrate_document(document), 1)
        self.assertEqual(
            document["nodes"][0]["type"],
            "CyberdeliaSaveImageWithMetaData",
        )
        self.assertEqual(document["nodes"][1]["type"], "SaveImageWithMetaData")
        self.assertEqual(document["nodes"][2]["type"], "SaveImageWithMetaData")
        self.assertEqual(
            document["api"]["1"]["class_type"], "SaveImageWithMetaData"
        )

    def test_dry_run_preserves_file_and_write_creates_original_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            original = json.dumps({
                "nodes": [{
                    "id": 1,
                    "type": "SaveImageWithMetaData",
                    "properties": {"cnr_id": "comfyui-cyberdelia-metadata"},
                }]
            }).encode("utf-8")
            path.write_bytes(original)

            self.assertEqual(MIGRATION.migrate_file(path, write=False), 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(Path(str(path) + ".bak").exists())

            self.assertEqual(MIGRATION.migrate_file(path, write=True), 1)
            self.assertEqual(Path(str(path) + ".bak").read_bytes(), original)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["nodes"][0]["type"],
                "CyberdeliaSaveImageWithMetaData",
            )

    def test_invalid_json_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{broken", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                MIGRATION.migrate_file(path, write=True)

            self.assertEqual(path.read_text(encoding="utf-8"), "{broken")
            self.assertFalse(Path(str(path) + ".bak").exists())


if __name__ == "__main__":
    unittest.main()
