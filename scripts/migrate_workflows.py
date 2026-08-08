#!/usr/bin/env python3
"""Migrate legacy metadata node ids in ComfyUI workflow/API JSON files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


PACK_ID = "comfyui-cyberdelia-metadata"
PACK_VERSION = "2.0.0"
LEGACY_TO_CANONICAL = {
    "SaveImageWithMetaData": "CyberdeliaSaveImageWithMetaData",
    "CreateExtraMetaData": "CyberdeliaCreateExtraMetaData",
}


def migrate_document(document, *, all_legacy=False):
    """Mutate a workflow/API document and return the number of migrated nodes.

    Tagged Cyberdelia UI nodes are safe to identify automatically. Untagged
    nodes and API JSON use the same legacy ids as nkchocoai's original pack,
    so callers must opt in with ``all_legacy=True``.
    """
    migrated = 0

    def visit(value):
        nonlocal migrated
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        properties = value.get("properties")
        pack_id = None
        if isinstance(properties, dict):
            pack_id = properties.get("cnr_id") or properties.get("aux_id")
        eligible = all_legacy or pack_id == PACK_ID

        node_migrated = False
        for field in ("type", "class_type"):
            replacement = (
                LEGACY_TO_CANONICAL.get(value.get(field)) if eligible else None
            )
            if replacement:
                value[field] = replacement
                node_migrated = True

        if node_migrated and isinstance(properties, dict):
            properties["cnr_id"] = PACK_ID
            properties["ver"] = PACK_VERSION
            search_name = properties.get("Node name for S&R")
            replacement = LEGACY_TO_CANONICAL.get(search_name)
            if replacement:
                properties["Node name for S&R"] = replacement

        if node_migrated:
            migrated += 1

        for nested in value.values():
            visit(nested)

    visit(document)
    return migrated


def _atomic_write_json(path: Path, document) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def migrate_file(path: Path, *, write: bool = False, all_legacy: bool = False):
    """Migrate one JSON file; writes atomically and keeps the first backup."""
    original = path.read_bytes()
    document = json.loads(original.decode("utf-8"))
    migrated = migrate_document(document, all_legacy=all_legacy)

    if migrated and write:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
        _atomic_write_json(path, document)

    return migrated


def iter_json_files(paths):
    seen = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        candidates = path.rglob("*.json") if path.is_dir() else (path,)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or candidate.suffix.lower() != ".json":
                continue
            seen.add(resolved)
            yield candidate


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Replace the colliding SaveImageWithMetaData/CreateExtraMetaData "
            "ids with the unique Cyberdelia ids. The default is a dry run."
        )
    )
    parser.add_argument("paths", nargs="+", help="JSON files or directories")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write changes atomically and create a .json.bak backup",
    )
    parser.add_argument(
        "--all-legacy",
        action="store_true",
        help=(
            "also migrate untagged and API nodes; use only when the selected "
            "legacy nodes are meant to be Cyberdelia nodes"
        ),
    )
    args = parser.parse_args(argv)

    changed_files = 0
    migrated_nodes = 0
    errors = 0
    for path in iter_json_files(args.paths):
        try:
            count = migrate_file(
                path,
                write=args.write,
                all_legacy=args.all_legacy,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors += 1
            print(f"ERROR {path}: {error}")
            continue
        if count:
            changed_files += 1
            migrated_nodes += count
            action = "updated" if args.write else "would update"
            print(f"{action}: {path} ({count} node(s))")

    mode = "write" if args.write else "dry run"
    print(
        f"{mode}: {migrated_nodes} node(s) in {changed_files} file(s); "
        f"{errors} error(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
