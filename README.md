# comfyui-cyberdelia-metadata

Civitai-compatible image metadata for ComfyUI, with robust handling of complex conditioning chains and modern multi-sampler workflows.

By **Cyberdelia AI Lab** · [github.com/cyberdeliaAI](https://github.com/cyberdeliaAI)

## What it does

Adds a Cyberdelia save node that writes structured metadata to PNG, JPG, or WebP files. The format is compatible with Civitai, so when you upload your image it reads back your seed, model, LoRAs, prompts, and sampler settings automatically — no manual entry.

## What's different in this version

Writing prompt and sampler info into a PNG sounds simple, but breaks down in real workflows. This release is focused on the rough edges:

### Correct prompt/sampler attribution in complex graphs

- **ConditioningZeroOut handling** — when your negative is zeroed out, metadata reports it as empty instead of walking past the zero-out and picking up whatever CLIPTextEncode was upstream.
- **rgthree Context Big / Context Switch** — the walker follows the right input slot through these passthrough nodes, so the negative branch doesn't accidentally pull text from the positive branch.
- **ControlNet apply chains** — passthrough resolution that respects positive/negative separation.
- **Multi-sampler workflows** — in base + upscale pass setups, the primary generation sampler (farthest from the save node) is reported, not whichever was found first.

### Runtime text capture

Nodes that compute their final text at runtime — wildcard expanders, dynamic prompts, LLM-based prompt engineers like [Cyberdelia Z-Engineer](https://github.com/cyberdeliaAI/comfyui-cyberdelia-z-engineer) — can register their resolved text via the hook API and have it captured in metadata, instead of falling back to a raw widget value or unresolved wildcard placeholder.

### Broad third-party integration

Out-of-the-box support for ~20 popular custom node packs (rgthree, efficiency-nodes, easyuse-nodes, lora-manager, RES4LYF, WanVideoWrapper, and more — see [`modules/defs/ext/`](modules/defs/ext/)). Adding a new one is a small Python file following an existing pattern.

### Output flexibility

PNG (lossless), JPG, or WebP (lossy or lossless) at adjustable quality. Optional sidecar `.json` workflow file. Subdirectory templating with date masks, model name, and prompt prefixes. Five metadata scopes from `full` to `none`.

## Installation

### Via ComfyUI-Manager

Search for **`comfyui-cyberdelia-metadata`** or **`Cyberdelia`** and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cyberdeliaAI/comfyui-cyberdelia-metadata.git
```

Restart ComfyUI.

Version 2 uses installation-wide unique serialized ids. It can coexist with
`nkchocoai/ComfyUI-SaveImageWithMetaData`; neither pack overwrites the other's
node registration.

## Usage

Replace your `Save Image` node with **Save Image With Metadata (Cyberdelia)**.
Hook up the image input — the rest is automatic.

If you use rgthree, you can optionally connect the `CONTEXT` output of
**Context** or **Context Big** to the node's `context` input. Non-empty context
values for seed, steps, CFG, checkpoint, sampler, and scheduler take priority
over automatic graph detection. The context conditioning selects the prompt;
its text fields and CLIP dimensions fill gaps only when prompt text or generation
size cannot otherwise be detected. Empty context fields keep using the normal
graph fallback, so the input is safe to leave disconnected and existing
workflows remain unchanged.

## Node options

| Option | Description |
|---|---|
| `context` | Optional rgthree `CONTEXT`; populated values are authoritative metadata overrides |
| `filename_prefix` | Filename template — supports mask tokens (see below) |
| `subdirectory_name` | Relative subdirectory inside ComfyUI's output folder — supports mask tokens |
| `output_format` | `png`, `jpg`, `webp`, or any of those + `_with_json` for a sidecar workflow file |
| `quality` | `max` / `lossless WebP` (100%), `high` (80%), `medium` (60%), `low` (30%). PNG ignores this. |
| `metadata_scope` | `full` (default + extras), `default` (Comfy stock), `parameters_only` (A1111 string), `workflow_only`, or `none` |

### Filename and subdirectory templating

`filename_prefix` and `subdirectory_name` accept these mask tokens:

| Token | Replacement |
|---|---|
| `%seed%` | Seed value |
| `%width%` / `%height%` | Image dimensions |
| `%pprompt%` / `%pprompt:[n]%` | Positive prompt (optionally first *n* characters) |
| `%nprompt%` / `%nprompt:[n]%` | Negative prompt (optionally first *n* characters) |
| `%model%` / `%model:[n]%` | Checkpoint name (optionally first *n* characters) |
| `%date%` | Date as `yyyyMMddhhmmss` |
| `%date:[format]%` | Date in a custom format |

Date format identifiers: `yyyy` (year), `MM` (month), `dd` (day), `hh` (hour), `mm` (minute), `ss` (second). Example: `%date:yyyy-MM%` produces `2026-05`.

## Runtime text capture (for node authors)

Building a custom node that computes its final prompt text at runtime? You can register that text so it ends up in saved metadata instead of whatever the user typed in the widget. The mechanism is loose-coupled — no hard import dependency on this extension:

```python
import sys
from comfy_execution.utils import get_executing_context

context = get_executing_context()
if context is not None:
    for mod in sys.modules.values():
        if mod is None:
            continue
        record_fn = getattr(mod, "record_resolved_text", None)
        if callable(record_fn):
            record_fn(context.node_id, final_text, getattr(context, "list_index", None))
```

If users don't have a compatible metadata extension installed, the snippet does nothing. This is how [Cyberdelia Z-Engineer](https://github.com/cyberdeliaAI/comfyui-cyberdelia-z-engineer) gets its LLM-engineered output into metadata.

## Supported third-party nodes

Each file in [`modules/defs/ext/`](modules/defs/ext/) registers a third-party node pack. Currently covered: rgthree, efficiency-nodes, easyuse-nodes, lora-manager, RES4LYF, WanVideoWrapper, Lightx02-Nodes, comfyui-custom-scripts, comfyui-clip-with-break, comfyui-easy-civitai-xt-nodes, comfyui-flux-settings-node, comfyui-gguf, comfyui-miaoshouai-tagger, comfyui-restart-sampling, comfyui-weilinnodes, ComfyUI-NegPiP-ZImage, CheckpointDiscoveryHub, CR_ApplyLoRAStack, everywhere, size_from_presets, SantodanNodes.

> [!TIP]
> If the `full` metadata scope errors out, it's usually an unrecognised third-party node in your workflow. Either swap to a Comfy Core equivalent or add a new file under [`modules/defs/ext/`](modules/defs/ext/) following the existing pattern.

## Migrating workflows to version 2

The legacy ids `SaveImageWithMetaData` and `CreateExtraMetaData` are owned by
the original nkchocoai pack as well. Version 2 therefore uses the unique ids
`CyberdeliaSaveImageWithMetaData` and
`CyberdeliaCreateExtraMetaData`. Registering the old ids as aliases would
reintroduce the same load-order conflict, so they are deliberately not exposed
by the backend.

Normal UI workflows created with this fork are migrated automatically before
ComfyUI checks for missing nodes, but only when their `cnr_id` identifies
`comfyui-cyberdelia-metadata`. Save the workflow once to persist the new ids.
Workflows belonging to nkchocoai are left untouched.

For a directory of workflow JSON files, first run the bundled migrator in dry
run mode:

```bash
python3 scripts/migrate_workflows.py /path/to/workflows
```

Then write the changes; each changed file gets a `.json.bak` backup:

```bash
python3 scripts/migrate_workflows.py --write /path/to/workflows
```

Old API JSON and workflows without `cnr_id` cannot be distinguished from the
original pack automatically. If the selected files are definitely meant to
use the Cyberdelia nodes, opt in explicitly:

```bash
python3 scripts/migrate_workflows.py --all-legacy --write /path/to/workflows
```

This package remains the direct successor to
`revived_comfyui_image_metadata_extension`, but its old workflows need the same
one-time migration because they also use the shared legacy ids.

## Credits

Built on the work of:

- **edelvarden** — [comfyui_image_metadata_extension](https://github.com/edelvarden/comfyui_image_metadata_extension) — original concept and initial implementation.
- **Santodan** — [revived_comfyui_image_metadata_extension](https://github.com/Santodan/revived_comfyui_image_metadata_extension) — picked up maintenance and added LoRA metadata, subdirectory templating, format/quality controls, and metadata scope options.
- **Cyberdelia AI Lab** — this version — conditioning chain resolution rewrite, multi-sampler workflow support, runtime text capture, and ongoing maintenance.

## License

GPL-3.0 — see [LICENSE](LICENSE). Inherited through the fork chain; derivative works must remain GPL-3.0 and preserve copyright notices of all prior authors.
