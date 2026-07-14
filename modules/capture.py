import asyncio
import inspect
import json
import os
import re
from collections import defaultdict, deque
from . import hook
from .defs.captures import CAPTURE_FIELD_LIST
from .defs.meta import MetaField
from .defs.formatters import calc_lora_hash, calc_model_hash, extract_embedding_names, extract_embedding_hashes
from .utils.log import print_warning

from nodes import NODE_CLASS_MAPPINGS
from .trace import Trace


class OutputCacheCompat:
    """Stub — HierarchicalCache (ComfyUI 0.3.68+) uses async frozenset-keyed
    lookups that cannot be called synchronously. All resolution is done via
    pure prompt-graph walking instead. This class is kept for API compatibility
    but _get_outputs_cache() always returns None so it is never instantiated.
    """
    def __init__(self, cache):
        self._cache = None
    def get(self, node_id): return None
    def get_output_cache(self, node_id, unique_id=None): return None
    def get_cache(self, node_id, unique_id=None): return None



# ---------------------------------------------------------------------------
# Runtime-resolved node text store.
# Populated by Capture.get_inputs() after awaiting get_input_data() per node,
# and seeded from hook.current_resolved_texts (CLIPTextEncode wrapper).
# Keyed by node_id (str) -> resolved text (str or list[str]).
# This is the only reliable source for wildcard-expanded / dynamic text.
# ---------------------------------------------------------------------------
_resolved_node_texts: dict = {}


def _clear_resolved_texts():
    _resolved_node_texts.clear()
    _resolved_node_texts.update(getattr(hook, "current_resolved_texts", {}))


def _coerce_text_value(value, batch_index=0):
    """Normalize a text value to a plain string, handling lists/tuples."""
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        idx = min(max(batch_index, 0), len(value) - 1)
        item = value[idx]
        return item if isinstance(item, str) and item.strip() else None
    return None


def _has_text_value(value):
    """Return True when any batch position contains non-empty text."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, str) and item.strip() for item in value)
    return False


_UNRESOLVED_PROMPT_PATTERNS = (
    re.compile(r"__[\w./\\-]+__"),       # wildcard placeholders
    re.compile(r"\{\d+\$\$"),            # dynamic prompt syntax
)


def _looks_unresolved_prompt_text(value):
    """Return True when *value* still contains wildcard / dynamic prompt syntax."""
    text = _coerce_text_value(value)
    if not text:
        return False
    return any(pattern.search(text) for pattern in _UNRESOLVED_PROMPT_PATTERNS)


def _should_prefer_graph_prompt(current_value, graph_value, opposite_graph_value=None):
    """Decide whether graph routing is more reliable than a captured value."""
    current_text = _coerce_text_value(current_value)
    graph_text = _coerce_text_value(graph_value)
    opposite_text = _coerce_text_value(opposite_graph_value)
    if not graph_text or current_text == graph_text:
        return False
    if not current_text or _is_link(current_value):
        return True
    if opposite_text and current_text == opposite_text:
        return True
    return _looks_unresolved_prompt_text(current_text) and not _looks_unresolved_prompt_text(graph_text)


def _entries_by_node(entries):
    """Group metadata entries by source node while preserving occurrence order."""
    grouped = defaultdict(deque)
    for entry in entries or []:
        if len(entry) > 1:
            grouped[str(entry[0])].append(entry)
    return grouped


def _pair_entries_by_node(left_entries, right_entries):
    """Pair related metadata by node id instead of positional list index."""
    right_by_node = _entries_by_node(right_entries)
    for left in left_entries or []:
        matches = right_by_node.get(str(left[0]))
        if matches:
            yield left, matches.popleft()


def _resolve_number_from_graph(value, prompt, input_keys=(), _visited=None):
    """Resolve a numeric widget value through primitive/value-node links."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not _is_link(value):
        return None

    node_id = str(value[0])
    visited = set() if _visited is None else _visited
    if node_id in visited:
        return None
    visited = visited | {node_id}

    node = prompt.get(node_id)
    if node is None:
        return None
    node_inputs = node.get("inputs", {})
    keys = tuple(input_keys) + ("value", "float", "number", "scale_by", "upscale_by")
    for key in keys:
        if key not in node_inputs:
            continue
        resolved = _resolve_number_from_graph(node_inputs[key], prompt, input_keys, visited)
        if resolved is not None:
            return resolved
    return None


def _build_upscale_stages(inputs, prompt, active_trace_tree=None):
    """Build atomic upscale-stage records on the image path to Save Image."""
    names = _entries_by_node(inputs.get(MetaField.UPSCALE_MODEL_NAME, []))
    hashes = _entries_by_node(inputs.get(MetaField.UPSCALE_MODEL_HASH, []))
    scales = _entries_by_node(inputs.get(MetaField.UPSCALE_BY, []))
    active_ids = set(active_trace_tree) if active_trace_tree is not None else None
    sampling_inputs = {"model", "positive", "negative", "vae"}
    stages = []

    for consumer_id, consumer in prompt.items():
        if active_ids is not None and consumer_id not in active_ids:
            continue
        consumer_inputs = consumer.get("inputs", {})
        model_link = consumer_inputs.get("upscale_model")
        if not _is_link(model_link):
            continue

        loader_id = str(model_link[0])
        name_entries = names.get(loader_id)
        if not name_entries:
            continue

        class_type = consumer.get("class_type", "")
        class_lower = class_type.lower()
        is_diffusion = (
            "ultimatesdupscale" in class_lower
            or len(sampling_inputs & set(consumer_inputs)) >= 2
            or ("denoise" in consumer_inputs and "steps" in consumer_inputs)
        )

        name_entry = name_entries[0]
        hash_entries = hashes.get(loader_id)
        hash_entry = hash_entries[0] if hash_entries else None
        scale_value = None
        for key in ("upscale_by", "scale_by", "upscale_factor"):
            if key in consumer_inputs:
                scale_value = _resolve_number_from_graph(
                    consumer_inputs[key], prompt, (key,)
                )
                if scale_value is not None:
                    break

        # Some workflows separate model upscaling and explicit resizing into
        # ImageUpscaleWithModel -> ImageScaleBy. Associate a directly
        # downstream scale node with this same stage rather than selecting an
        # unrelated global scale value.
        if scale_value is None:
            for scale_node_id, scale_entries in scales.items():
                if active_ids is not None and scale_node_id not in active_ids:
                    continue
                scale_node = prompt.get(scale_node_id)
                if scale_node is None:
                    continue
                scale_inputs = scale_node.get("inputs", {})
                image_link = scale_inputs.get("image") or scale_inputs.get("images")
                if _is_link(image_link) and str(image_link[0]) == str(consumer_id):
                    scale_value = scale_entries[0][1]
                    break

        denoise_value = None
        for key in ("denoise", "denoise_strength"):
            if key in consumer_inputs:
                denoise_value = _resolve_number_from_graph(
                    consumer_inputs[key], prompt, (key,)
                )
                if denoise_value is not None:
                    break

        trace = active_trace_tree.get(consumer_id) if active_trace_tree else None
        consumer_distance = trace[0] if trace else (
            name_entry[2] if len(name_entry) > 2 else float("inf")
        )
        stages.append({
            "consumer_id": consumer_id,
            "loader_id": loader_id,
            "class_type": class_type,
            "is_diffusion": is_diffusion,
            "distance": consumer_distance,
            "name": name_entry[1],
            "hash": hash_entry[1] if hash_entry else None,
            "scale": scale_value,
            "denoise": denoise_value,
        })

    return stages


def _select_hires_upscale_stage(inputs, prompt, active_trace_tree=None):
    """Select one coherent hires stage, falling back to the nearest image stage."""
    stages = _build_upscale_stages(inputs, prompt, active_trace_tree)
    if not stages:
        return None
    diffusion_stages = [stage for stage in stages if stage["is_diffusion"]]
    candidates = diffusion_stages or stages
    return min(candidates, key=lambda stage: stage["distance"])


# ---------------------------------------------------------------------------
# Helpers to walk the prompt graph and extract raw text regardless of how
# many indirection levels (wired text nodes, concat nodes, etc.) there are.
# ---------------------------------------------------------------------------

# Node class names that are text-concatenation / joining nodes.
_CONCAT_CLASS_HINTS = [
    "concat", "join", "combine", "mixer",
    "string", "text",        # many "TextConcatenate", "StringJoin" nodes
]

# Input key names that carry text payloads inside concat-style nodes.
_TEXT_KEY_HINTS = [
    "text", "string", "input", "value", "prompt",
    "text1", "text2", "text_a", "text_b", "string_a", "string_b",
    "string1", "string2",
    "positive_prompt", "negative_prompt",
]

# Node class name fragments for dynamic text-generator nodes whose output
# text only exists at runtime.  We fall back to their best static input.
_DYNAMIC_TEXT_NODES = {
    # class_type_lower → list of input keys to try in order
    "wildcardmanager":      ["input_text"],
    "wildcard":             ["text", "input_text"],
    "dynamicprompt":        ["text", "template"],
    "randomlorafoldermodel":["extra_trigger_words"],  # string output slot 2
    "randomlora":           ["extra_trigger_words"],
}

# Custom conditioning nodes whose CONDITIONING outputs correspond to original
# text inputs. The output slot is significant: Z-Image NegPiP also exposes a
# compiled STRING on slot 3, but metadata should preserve the user's separate
# positive and negative prompts from slots 1 and 2.
_CONDITIONING_TEXT_INPUT_SLOT_MAP = {
    "ZImageNegPipPrompt": {
        1: "positive",
        2: "negative",
    },
}

# Conditioning routers whose output slots map one-to-one to named inputs.
# Resolve these before consulting runtime STRING output caches: multi-output
# context nodes can expose unrelated strings (sampler, scheduler, prompt text,
# etc.) on other slots. Those cache entries must not prevent a conditioning
# walk on slots 4/5 from reaching its real positive/negative source.
_CONDITIONING_PASSTHROUGH_SLOT_MAP = {
    # Context Big / Context (rgthree): slot 4 = positive, slot 5 = negative
    "Context Big (rgthree)": {4: "positive", 5: "negative"},
    "Context (rgthree)": {4: "positive", 5: "negative"},
    "Context Switch (rgthree)": {4: "positive", 5: "negative"},
    "Context Switch Big (rgthree)": {4: "positive", 5: "negative"},
    # ControlNet nodes: slot 0 = positive, slot 1 = negative
    "ControlNetApplyAdvanced": {0: "positive", 1: "negative"},
    "ControlNetApply": {0: "positive", 1: "negative"},
}


def _is_link(value):
    """Return True when *value* looks like a ComfyUI node-output link [node_id, index]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _resolve_text_from_graph(value, prompt, outputs, _visited=None, batch_index=0):
    """
    Recursively resolve *value* to a plain string by walking the prompt graph.

    *value* can be:
      - A plain string  → returned as-is.
      - A link          → follow to the source node and recurse.
      - None            → returns None.

    *batch_index* selects which entry to use when a cache slot holds a list
    of strings (i.e. when a list was fed into the node, generating one image
    per entry).  Pass the current image's position in the batch so each image
    gets its own prompt text rather than always the first one.

    The function tries (in order):
      1. The execution cache (already-evaluated output).
      2. A ``text`` / ``string`` / similar field on the source node's inputs
         (handles CLIPTextEncode with a wired-in text node).
      3. Concatenation / joining nodes whose text inputs are all resolved
         recursively and joined with the node's separator.

    *_visited* prevents infinite loops on cyclic graphs.
    """
    if _visited is None:
        _visited = set()

    if value is None:
        return None

    # Already a plain string – nothing to resolve.
    if isinstance(value, str):
        return value if value.strip() else None

    # Unwrap single-element lists that ComfyUI sometimes produces.
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _resolve_text_from_graph(value[0], prompt, outputs, _visited, batch_index)

    if not _is_link(value):
        return None

    node_id = str(value[0])
    out_slot = value[1] if len(value) > 1 else 0

    if node_id in _visited:
        return None
    _visited = _visited | {node_id}

    # ── 1. Runtime interception cache (populated by HierarchicalCache.set patch) ────────
    # Check slot-specific key first, then plain node_id key.
    slot_key = f"{node_id}:{out_slot}"
    cached_text = _coerce_text_value(
        _resolved_node_texts.get(slot_key) or _resolved_node_texts.get(node_id),
        batch_index=batch_index,
    )
    if cached_text:
        return cached_text

    # ── 2. Walk the graph node ───────────────────────────────────────────────
    node = prompt.get(node_id)
    if node is None:
        return None

    node_inputs = node.get("inputs", {})
    class_type = node.get("class_type", "").lower()

    # Direct text field on this node (e.g. CLIPTextEncode whose "text" is
    # a hard-coded string, or a primitive String node).
    for key in ("text", "string", "value", "val", "prompt",
                "positive_prompt", "negative_prompt"):
        raw = node_inputs.get(key)
        if raw is None:
            continue
        if isinstance(raw, str) and raw.strip():
            return raw
        if _is_link(raw):
            resolved = _resolve_text_from_graph(
                raw, prompt, outputs, _visited, batch_index
            )
            if resolved:
                return resolved

    # ── 3. Concatenation / joining nodes ────────────────────────────────────
    is_concat = any(hint in class_type for hint in _CONCAT_CLASS_HINTS)
    if is_concat:
        # Collect all text-like input keys in stable order.
        candidate_keys = sorted(
            (k for k in node_inputs if any(h in k.lower() for h in _TEXT_KEY_HINTS)),
            key=lambda k: (re.sub(r'\d+', '', k),
                           int(re.search(r'\d+', k).group()) if re.search(r'\d+', k) else 0)
        )
        parts = []
        for k in candidate_keys:
            resolved = _resolve_text_from_graph(node_inputs[k], prompt, outputs, _visited, batch_index)
            if resolved:
                parts.append(resolved)

        if parts:
            sep_raw = node_inputs.get("delimiter", node_inputs.get("separator", " "))
            sep = sep_raw.replace("\\n", "\n") if isinstance(sep_raw, str) else " "
            return sep.join(parts)

    # ── 4. Known dynamic text-generator nodes ───────────────────────────────
    # These nodes compute their output at execution time (wildcard expansion,
    # random LoRA selection, etc.).  We fall back to their best static input
    # as an approximation rather than returning nothing.
    for cls_hint, fallback_keys in _DYNAMIC_TEXT_NODES.items():
        if cls_hint in class_type:
            for fk in fallback_keys:
                raw = node_inputs.get(fk)
                if raw is None:
                    continue
                if isinstance(raw, str) and raw.strip():
                    return raw
                if _is_link(raw):
                    resolved = _resolve_text_from_graph(raw, prompt, outputs, _visited, batch_index)
                    if resolved:
                        return resolved
            break  # matched a dynamic node — don't fall through to generic scan

    # ── 5. Fallback: scan only text-hinted input keys, never model/clip/vae ──
    # IMPORTANT: skip nodes already matched as dynamic to prevent infinite loops
    # where e.g. RandomLoraFolderModel.extra_trigger_words links back upstream.
    _NON_TEXT_KEYS = {"model", "clip", "vae", "control_net", "image", "mask",
                      "latent", "latent_image", "samples", "upscale_model",
                      "positive", "negative", "conditioning"}
    is_dynamic = any(h in class_type for h in _DYNAMIC_TEXT_NODES)
    if not is_dynamic:
        for key, raw in node_inputs.items():
            if key.lower() in _NON_TEXT_KEYS:
                continue
            if _is_link(raw):
                # Only follow if the key name hints at text content
                if any(h in key.lower() for h in _TEXT_KEY_HINTS):
                    resolved = _resolve_text_from_graph(raw, prompt, outputs, _visited, batch_index)
                    if resolved:
                        return resolved

    return None


def _resolve_clip_text_encode_prompt(node_id, prompt, outputs, batch_index=0):
    """
    Given a CLIPTextEncode node's *node_id*, return its resolved text string.

    The CLIPTextEncode node has a single ``text`` input which may be:
      - A hard-coded string.
      - A link to another node (primitive, text node, concat node, …).
      - A list of strings when a list was wired in (one entry per batch image).
    """
    nid = str(node_id)

    # ── 1. Runtime-resolved text (populated by await get_input_data) ─────────
    # This is the only reliable source when "text" is wired from a dynamic
    # node (WildcardManager, StringConcatenate, etc.).
    cached_text = _coerce_text_value(_resolved_node_texts.get(nid), batch_index=batch_index)
    if cached_text:
        return cached_text

    # ── 2. Static graph walk (fallback for hardcoded text) ───────────────────
    node = prompt.get(nid)
    if node is None:
        return None
    raw = node.get("inputs", {}).get("text")
    if raw is None:
        return None
    # Hard-coded string directly in the node
    if isinstance(raw, str):
        return raw if raw.strip() else None
    # A link [node_id, output_index] — resolve through the graph
    if _is_link(raw):
        return _resolve_text_from_graph(raw, prompt, outputs, batch_index=batch_index)
    # A genuine list of strings (one per batch entry) — NOT a link
    if isinstance(raw, list) and raw and all(isinstance(x, str) for x in raw):
        idx = min(batch_index, len(raw) - 1)
        return raw[idx] if raw[idx].strip() else None
    return None


def _follow_conditioning_to_clip_text(cond_value, prompt, outputs, _depth=0, batch_index=0):
    """
    Follow a conditioning link chain until we reach a CLIPTextEncode and
    resolve its text.

    *batch_index* is forwarded all the way down so that when the text source
    is a list (one string per batch image), the correct entry is selected.
    """
    if _depth > 20:  # safety limit
        return None
    if not _is_link(cond_value):
        return None

    src_id = str(cond_value[0])
    src_node = prompt.get(src_id)
    if src_node is None:
        return None

    src_class = src_node.get("class_type", "")
    src_inputs = src_node.get("inputs", {})

    # ── Direct CLIPTextEncode ─────────────────────────────────────────────
    if src_class == "CLIPTextEncode":
        return _resolve_clip_text_encode_prompt(src_id, prompt, outputs, batch_index)

    # ── Custom conditioning node with slot-bound source text ─────────────
    # Resolve this before consulting the generic runtime-output cache. A node
    # may have a STRING output on another slot (e.g. NegPiP compiled_prompt),
    # which intentionally must not replace either original prompt.
    out_slot = cond_value[1] if len(cond_value) > 1 else 0
    text_input_map = _CONDITIONING_TEXT_INPUT_SLOT_MAP.get(src_class)
    if text_input_map is not None:
        input_key = text_input_map.get(out_slot)
        if input_key and input_key in src_inputs:
            resolved = _resolve_text_from_graph(
                src_inputs[input_key], prompt, outputs,
                batch_index=batch_index,
            )
            if resolved:
                return resolved

    # ── Known conditioning pass-through node ─────────────────────────────
    # This must run before the generic runtime-cache short-circuit below.
    # Context nodes can have cached STRING values on unrelated output slots;
    # their presence says nothing about the conditioning on slot 4 or 5.
    passthrough_slot_map = _CONDITIONING_PASSTHROUGH_SLOT_MAP.get(src_class)
    if passthrough_slot_map is not None:
        follow_key = passthrough_slot_map.get(out_slot)
        follow_value = src_inputs.get(follow_key) if follow_key else None
        if _is_link(follow_value):
            result = _follow_conditioning_to_clip_text(
                follow_value, prompt, outputs, _depth + 1, batch_index
            )
            if result:
                return result

    # ── Runtime-resolved text from any node that pushed it ────────────────
    # Nodes that compute their final text at runtime (LLM-based prompt
    # engineers, dynamic prompt expanders, etc.) can register their resolved
    # text via hook.record_resolved_text(node_id, text) or by writing
    # directly to current_resolved_texts. We honor that here so the metadata
    # reflects what was actually encoded, not the raw widget.
    #
    # Slot-aware lookup with cache-aware short-circuit:
    #
    # The cache can contain three kinds of entries per node:
    #   - "{nid}:{slot}" — explicit slot text (manual push, e.g. Z-Engineer
    #     pushing engineered text under "{nid}:0" only for its positive
    #     output)
    #   - "{nid}" — bare key, used by single-output wrappers (CLIPTextEncode
    #     runtime wrapper) AND auto-populated from runtime outputs cache
    #     for ANY string-yielding output of the node
    #   - Auto-populated "{nid}:{slot}" entries from string outputs
    #
    # The trap: a node with multiple outputs (e.g. Z-Engineer outputs
    # positive CONDITIONING at slot 0, negative CONDITIONING at slot 1,
    # and a STRING prompt at slot 2) has its slot 2 string auto-populated
    # to BOTH "{nid}:2" and the bare "{nid}". A naive lookup that falls
    # back on the bare key for slot 1's miss would return the slot 2 text
    # as the "negative" text — wrong.
    #
    # Fix: bare-key fallback is only valid if the node has NO explicit
    # slot entries. The presence of any "{nid}:*" entry marks the node
    # as slot-aware, and a missing slot is treated as "no text here".
    slot_key = f"{src_id}:{out_slot}"

    cached = _coerce_text_value(_resolved_node_texts.get(slot_key), batch_index=batch_index)
    if cached:
        return cached

    # Is this node slot-aware? (has any "{nid}:*" entries)
    has_slot_entries = any(
        k.startswith(f"{src_id}:") for k in _resolved_node_texts.keys()
    )

    if has_slot_entries:
        # Slot-aware node, but the queried slot has no text registered.
        # This is a deliberate "no text on this output" signal — do NOT
        # fall back to bare node_id key (which may be auto-populated from
        # a different output) or to widget text.
        return None

    # No slot-specific entries — try the bare node_id key (single-output
    # nodes like the CLIPTextEncode runtime wrapper). Fall through to
    # widget extraction below if this also misses.
    cached = _coerce_text_value(_resolved_node_texts.get(src_id), batch_index=batch_index)
    if cached:
        return cached

    # ── Node with its own text field (e.g. some conditioning wrappers) ───
    for k in ("text", "string", "prompt"):
        raw = src_inputs.get(k)
        if raw is not None:
            resolved = _resolve_text_from_graph(raw, prompt, outputs, batch_index=batch_index)
            if resolved:
                return resolved

    # ── Conditioning pass-through nodes (Context Big, ControlNetApply, …) ──
    # These nodes route conditioning through without changing it.  We use the
    # *output slot* we arrived from to pick the matching input ("positive" or
    # "negative") so positive/negative chains stay separate.
    #
    if (passthrough_slot_map is None
            and "positive" in src_inputs and "negative" in src_inputs):
        # Generic fallback for unknown pass-through nodes that have both
        # positive and negative inputs: try to match by ordered input position.
        # Build a list of input keys in definition order and find which slot
        # corresponds to "positive" vs "negative".
        ordered_keys = list(src_inputs.keys())
        cond_positions = {}
        for idx, k in enumerate(ordered_keys):
            if k in ("positive", "negative"):
                cond_positions[idx] = k
        # If the output slot matches a conditioning position, follow it.
        follow_key = cond_positions.get(out_slot)
        if follow_key and _is_link(src_inputs[follow_key]):
            result = _follow_conditioning_to_clip_text(
                src_inputs[follow_key], prompt, outputs, _depth + 1, batch_index
            )
            if result:
                return result

    # ── Conditioning passthrough: follow the *first* conditioning input ──
    PASSTHROUGH_KEYS = ("conditioning", "cond", "conditioning_1", "conditioning_2")
    for k in PASSTHROUGH_KEYS:
        if k in src_inputs:
            result = _follow_conditioning_to_clip_text(
                src_inputs[k], prompt, outputs, _depth + 1, batch_index
            )
            if result:
                return result

    # ── Last resort: any link-valued input that isn't a model/image slot ─
    _SKIP_KEYS = {"model", "clip", "vae", "image", "mask", "latent",
                  "latent_image", "samples", "positive", "negative"}
    for k, v in src_inputs.items():
        if k in _SKIP_KEYS:
            continue
        if _is_link(v):
            result = _follow_conditioning_to_clip_text(v, prompt, outputs, _depth + 1, batch_index)
            if result:
                return result

    return None


def _find_guider_node_with_conditioning(node_id, prompt):
    """
    Given a node_id, follow cfg_guider/guider links to find a node that has
    explicit conditioning inputs (e.g. CFGGuider or BasicGuider).
    Returns (node_id, node_dict) or (None, None).
    """
    visited = set()
    queue = [str(node_id)]
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = prompt.get(nid)
        if node is None:
            continue
        node_inputs = node.get("inputs", {})
        # Found a node that has conditioning slots. BasicGuider exposes its
        # single positive branch as `conditioning` instead of `positive`.
        if ("positive" in node_inputs or "negative" in node_inputs
                or (node.get("class_type") == "BasicGuider"
                    and "conditioning" in node_inputs)):
            return nid, node
        # Follow guider-type links deeper
        for k in ("cfg_guider", "guider", "positive_guider", "negative_guider",
                  "conditioning", "cond"):
            v = node_inputs.get(k)
            if _is_link(v):
                queue.append(str(v[0]))
    return None, None


def _find_prompt_texts(prompt, outputs, batch_index=0, sampler_node_id=None):
    """
    Walk the prompt graph to find the positive and negative prompt strings.

    Handles two major workflow topologies:

    Classic KSampler topology:
        KSampler(positive=COND, negative=COND, seed, steps, cfg, ...)

    SamplerCustomAdvanced topology (used by Flux / res_multistep_simple etc.):
        SamplerCustomAdvanced(noise=NOISE, cfg_guider=GUIDER, sampler=SAMPLER, sigmas=SIGMAS)
        CFGGuider(model, positive=COND, negative=COND, cfg)

    Both are detected and the conditioning chains are resolved independently
    to avoid swapping positive / negative.
    """
    SAMPLER_CLASSES = {
        "KSampler", "KSamplerAdvanced", "SamplerCustom",
        "KSamplerSelect", "KSampler_inspire",
        "KSamplerAdvancedPipe", "KSamplerPipe",
        "FluxKSampler", "FluxSampler",
        "SamplerCustomAdvanced",
    }
    # Inputs that indicate this node is a sampler even if the class name is unknown
    SAMPLER_HINT_KEYS = {"seed", "steps", "cfg", "sampler_name", "noise_seed", "denoise",
                         "cfg_guider", "noise", "sigmas"}
    # Nodes that hold conditioning but are NOT the sampler
    GUIDER_CLASSES = {"CFGGuider", "BasicGuider", "DualCFGGuider", "Guider"}

    # Prefer real sampler nodes over heuristic matches. Context Big and other
    # bundle nodes expose fields such as seed/steps/cfg too; treating whichever
    # one appears first in the prompt dict as a sampler can select an unrelated
    # conditioning branch. Only fall back to those hints for unknown sampler
    # implementations after all known samplers and guiders have been tried.
    if sampler_node_id is not None and str(sampler_node_id) in prompt:
        candidates = [(str(sampler_node_id), prompt[str(sampler_node_id)])]
        forced_sampler = True
    else:
        primary_candidates = []
        guider_candidates = []
        heuristic_candidates = []
        for item in prompt.items():
            _node_id, _node = item
            _class_type = _node.get("class_type", "")
            _inputs = _node.get("inputs", {})
            if _class_type in SAMPLER_CLASSES:
                primary_candidates.append(item)
            elif _class_type in GUIDER_CLASSES:
                guider_candidates.append(item)
            elif bool(SAMPLER_HINT_KEYS & set(_inputs.keys())):
                heuristic_candidates.append(item)
        candidates = primary_candidates + guider_candidates + heuristic_candidates
        forced_sampler = False

    for node_id, node in candidates:
        class_type = node.get("class_type", "")
        node_inputs = node.get("inputs", {})

        # ── Path A: classic node with positive+negative directly ─────────────
        has_pos_neg = "positive" in node_inputs and "negative" in node_inputs
        is_classic_sampler = has_pos_neg and (
            forced_sampler
            or class_type in SAMPLER_CLASSES
            or class_type in GUIDER_CLASSES
            or bool(SAMPLER_HINT_KEYS & set(node_inputs.keys()))
        )
        if is_classic_sampler:
            pos_text = _follow_conditioning_to_clip_text(
                node_inputs.get("positive"), prompt, outputs, batch_index=batch_index
            )
            neg_text = _follow_conditioning_to_clip_text(
                node_inputs.get("negative"), prompt, outputs, batch_index=batch_index
            )
            if pos_text or neg_text:
                return pos_text, neg_text

        # ── Path B: SamplerCustomAdvanced-style (cfg_guider link) ────────────
        if (forced_sampler
                or class_type in SAMPLER_CLASSES
                or bool(SAMPLER_HINT_KEYS & set(node_inputs.keys()))):
            for guider_key in ("cfg_guider", "guider"):
                guider_link = node_inputs.get(guider_key)
                if not _is_link(guider_link):
                    continue
                g_nid, g_node = _find_guider_node_with_conditioning(
                    str(guider_link[0]), prompt
                )
                if g_node is None:
                    continue
                g_inputs = g_node.get("inputs", {})
                g_class = g_node.get("class_type", "")
                positive_input = (
                    g_inputs.get("conditioning") if g_class == "BasicGuider"
                    else g_inputs.get("positive")
                )
                pos_text = _follow_conditioning_to_clip_text(
                    positive_input, prompt, outputs, batch_index=batch_index
                )
                # BasicGuider has only one positive conditioning input and no
                # negative branch. Treating its `conditioning` input as a
                # negative prompt duplicates positive into negative metadata.
                negative_input = (
                    None if g_class == "BasicGuider"
                    else g_inputs.get("negative") or g_inputs.get("conditioning")
                )
                neg_text = _follow_conditioning_to_clip_text(
                    negative_input,
                    prompt, outputs, batch_index=batch_index
                )
                if pos_text or neg_text:
                    return pos_text, neg_text

    return None, None


# ---------------------------------------------------------------------------
# Main Capture class (original logic preserved, prompt resolution patched)
# ---------------------------------------------------------------------------


def _get_outputs_cache():
    """
    The new HierarchicalCache (ComfyUI 0.3.68+) stores data under frozenset
    composite keys — there is no synchronous node_id -> output lookup.
    All async methods (.get, .get_cache) must not be called from sync code.

    We return None here so that all resolution falls through to the pure
    prompt-graph walk, which works correctly without any cache access.
    """
    return None


class Capture:
    @classmethod
    async def get_inputs(cls, batch_index=0):
        """
        Collect capturable field values from the active Save Image ancestry.

        Uses await get_input_data() per node — exactly like the original code —
        so that fully-resolved values (including wildcard-expanded text, LoRA
        trigger words, etc.) are available even when ComfyUI's execution cache
        is async (HierarchicalCache, ComfyUI 0.3.68+).

        The node's execute() method must also be async (see node.py) so that
        this coroutine can be awaited from the top-level save call.
        """
        from execution import get_input_data
        from comfy_execution.graph import DynamicPrompt

        _clear_resolved_texts()
        inputs = {}
        prompt = hook.current_prompt
        if not prompt:
            return inputs

        # Keep the executing save node selected by the hook. If a ComfyUI
        # version did not provide it, fall back to the first matching node.
        # Restrict expensive input/cache probing to that save node's ancestry;
        # unrelated workflow branches cannot contribute metadata anyway.
        save_node_id = hook.current_save_image_node_id
        if save_node_id not in prompt and str(save_node_id) in prompt:
            save_node_id = str(save_node_id)
        if (save_node_id not in prompt
                or prompt[save_node_id].get("class_type") != "SaveImageWithMetaData"):
            save_node_id = next(
                (
                    node_id for node_id, node in prompt.items()
                    if node.get("class_type") == "SaveImageWithMetaData"
                ),
                None,
            )
        hook.current_save_image_node_id = save_node_id if save_node_id is not None else -1
        active_node_ids = (
            set(Trace.trace(save_node_id, prompt)) if save_node_id is not None else None
        )

        extra_data = hook.current_extra_data

        # Pass the raw cache object directly to get_input_data — it knows how
        # to use HierarchicalCache (async) natively. OutputCacheCompat is only
        # used by our own sync graph-walking helpers (validate, selector, etc.).
        raw_outputs = None
        outputs = None   # sync-safe compat wrapper for validate/selector calls
        if hook.prompt_executer and hook.prompt_executer.caches:
            raw_outputs = hook.prompt_executer.caches.outputs
            # OutputCacheCompat for sync helpers only — NOT passed to get_input_data
            outputs = (
                raw_outputs
                if hasattr(raw_outputs, "get_output_cache")
                else OutputCacheCompat(raw_outputs)
            )

        # ── Bulk-populate _resolved_node_texts from HierarchicalCache ──────────
        # cache.get(node_id) returns a CacheEntry with .outputs — scan every
        # node now so _resolve_text_from_graph can find runtime values.
        if raw_outputs is not None:
            _gc = getattr(raw_outputs, "get", None)
            if _gc:
                cache_node_ids = active_node_ids or set(prompt.keys())
                for _nid in cache_node_ids:
                    try:
                        _cr = _gc(str(_nid))
                        if inspect.isawaitable(_cr):
                            _cr = await _cr
                        if _cr is None:
                            continue
                        _entry_outputs = getattr(_cr, "outputs", None)
                        if not isinstance(_entry_outputs, (list, tuple)):
                            continue
                        for _si, _sv in enumerate(_entry_outputs):
                            if _has_text_value(_sv):
                                _resolved_node_texts[f"{_nid}:{_si}"] = _sv
                                if str(_nid) not in _resolved_node_texts:
                                    _resolved_node_texts[str(_nid)] = _sv
                    except Exception:
                        pass

        for node_id, obj in prompt.items():
            if active_node_ids is not None and node_id not in active_node_ids:
                continue
            class_type = obj["class_type"]
            if class_type not in NODE_CLASS_MAPPINGS:
                continue
            obj_class = NODE_CLASS_MAPPINGS[class_type]
            node_inputs = obj.get("inputs", {})

            # get_input_data is async in ComfyUI 0.3.68+ — await it.
            # Pass execution_list=None: we don't have access to the ExecutionList
            # object ComfyUI uses internally, and passing CacheSet or
            # HierarchicalCache as execution_list causes get_input_data to call
            # execution_list.get_cache() which those types don't implement,
            # resulting in 'coroutine' object has no attribute 'outputs' in some
            # ComfyUI versions. With None, linked inputs are marked as missing and
            # we fall back to graph-walk resolution (our primary path anyway).
            try:
                import inspect as _inspect_mod
                input_data = get_input_data(
                    node_inputs, obj_class, node_id, None,
                    DynamicPrompt(prompt), extra_data
                )
                if _inspect_mod.isawaitable(input_data):
                    input_data = await input_data
                # input_data is now (input_data_all, missing_keys, v3_data) in
                # ComfyUI 0.3.x+.  input_data[0] is the resolved-inputs dict.
                _dbg = input_data[0] if isinstance(input_data, (list, tuple)) and input_data else {}
                if not isinstance(_dbg, dict):
                    _dbg = {}

            except Exception:
                input_data = ({}, {}, {})

            # For CLIPTextEncode with a linked text input, resolve via cache.get(node_id)
            # HierarchicalCache.get(node_id) returns a CacheEntry with .outputs list.
            if class_type == "CLIPTextEncode" and raw_outputs is not None:
                _dbg = input_data[0] if isinstance(input_data, (list,tuple)) and input_data else {}
                _txt = _dbg.get("text")
                _txt_missing = (_txt is None or _txt == (None,)
                                or (isinstance(_txt, (list,tuple)) and len(_txt) == 1 and _txt[0] is None))
                if _txt_missing:
                    _link = node_inputs.get("text")
                    if _is_link(_link):
                        _src_nid = str(_link[0])
                        _src_slot = int(_link[1])
                        try:
                            _gc = getattr(raw_outputs, "get", None)
                            if _gc:
                                _cr = _gc(_src_nid)
                                if asyncio.iscoroutine(_cr) or inspect.isawaitable(_cr):
                                    _cr = await _cr
                                if _cr is not None:
                                    _src_outputs = getattr(_cr, "outputs", None)
                                    if isinstance(_src_outputs, (list, tuple)) and len(_src_outputs) > _src_slot:
                                        _slot = _src_outputs[_src_slot]
                                        if isinstance(_slot, list) and len(_slot) == 1:
                                            _slot = _slot[0]
                                        if isinstance(_slot, str) and _slot.strip():
                                            _resolved_node_texts[str(node_id)] = _slot
                        except Exception:
                            pass

            # ── Store resolved text + probe async cache for this node ─────────
            _rid = str(node_id)

            # Scan CacheEntry objects that DO have ui.meta.node_id (display nodes).
            # Pure compute nodes (CLIPTextEncode etc.) have ui=None so we can't
            # identify them by cache key — we rely on get_input_data instead.
            if raw_outputs is not None and not _resolved_node_texts.get("__cache_scanned__"):
                _resolved_node_texts["__cache_scanned__"] = "1"
                _cache_dict = getattr(raw_outputs, "cache", {})
                for _entry in _cache_dict.values():
                    try:
                        _ui = getattr(_entry, "ui", None)
                        if not isinstance(_ui, dict):
                            continue
                        _entry_nid = str(_ui.get("meta", {}).get("node_id", "") or "")
                        if not _entry_nid:
                            continue
                        _entry_outputs = getattr(_entry, "outputs", None)
                        if not isinstance(_entry_outputs, (list, tuple)):
                            continue
                        for _si, _sv in enumerate(_entry_outputs):
                            if _has_text_value(_sv):
                                _resolved_node_texts[f"{_entry_nid}:{_si}"] = _sv
                                if _entry_nid not in _resolved_node_texts:
                                    _resolved_node_texts[_entry_nid] = _sv
                    except Exception:
                        pass

            # Fall back to get_input_data result for text fields
            if isinstance(input_data, (list, tuple)) and input_data:
                _rd = input_data[0] if isinstance(input_data[0], dict) else {}
                for _tkey in ("text", "string", "value", "prompt",
                              "positive_prompt", "negative_prompt"):
                    _tv = _coerce_text_value(
                        _rd.get(_tkey), batch_index=batch_index
                    )
                    if _tv:
                        if _rid not in _resolved_node_texts:
                            _resolved_node_texts[_rid] = _tv
                        break

            for node_class, metas in CAPTURE_FIELD_LIST.items():
                if class_type != node_class:
                    continue

                for meta, field_data in metas.items():
                    if field_data.get("validate") and not field_data["validate"](
                        node_id, obj, prompt, extra_data, outputs, input_data
                    ):
                        continue

                    if meta not in inputs:
                        inputs[meta] = []

                    value = field_data.get("value")
                    if value is not None:
                        inputs[meta].append((node_id, value))
                        continue

                    selector = field_data.get("selector")
                    if selector:
                        try:
                            v = selector(node_id, obj, prompt, extra_data, outputs, input_data)
                        except Exception:
                            v = None
                        cls._append_value(inputs, meta, node_id, v)
                        continue

                    field_name = field_data.get("field_name")
                    if not field_name:
                        continue

                    value = input_data[0].get(field_name) if isinstance(input_data, (list, tuple)) and input_data else None
                    if value is None:
                        continue

                    # If get_input_data returned a raw link instead of a resolved
                    # string (shouldn't happen with async await, but be safe)
                    if _is_link(value):
                        value = _resolve_text_from_graph(
                            value, prompt, _get_outputs_cache(),
                            batch_index=batch_index,
                        )
                    if value is None:
                        continue

                    format_func = field_data.get("format")
                    v = cls._apply_formatting(
                        value, input_data, format_func, batch_index=batch_index
                    )
                    cls._append_value(inputs, meta, node_id, v)

        return inputs


    @staticmethod
    def _apply_formatting(value, input_data, format_func, batch_index=0):
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            idx = min(max(batch_index, 0), len(value) - 1)
            value = value[idx]
        # ComfyUI can represent an unresolved linked input as ``(None,)`` or
        # ``[None]``. The outer container passes the earlier ``value is None``
        # guard, so check again after selecting the current batch item before
        # calling formatters that expect a string or number.
        if value is None:
            return None
        if format_func:
            value = format_func(value, input_data)
        return value

    @staticmethod
    def _append_value(inputs, meta, node_id, value):
        if isinstance(value, list):
            for x in value:
                inputs[meta].append((node_id, x))
        elif value is not None:
            inputs[meta].append((node_id, value))

    @classmethod
    def get_lora_strings_and_hashes(cls, inputs_before_sampler_node):

        def clean_name(n):
            return os.path.splitext(os.path.basename(n))[0].replace('\\', '_').replace('/', '_').replace(' ', '_').replace(':', '_')

        lora_assertion_re = re.compile(r"<(lora|lyco):([a-zA-Z0-9_\./\\-]+):([0-9.]+)>")

        prompt_texts = [
            val[1]
            for key in [MetaField.POSITIVE_PROMPT, MetaField.NEGATIVE_PROMPT]
            for val in inputs_before_sampler_node.get(key, [])
            if isinstance(val[1], str)
        ]
        prompt_joined = " ".join(prompt_texts).replace("\n", " ").replace("\r", " ").lower()

        lora_names = inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, [])
        lora_weights = inputs_before_sampler_node.get(MetaField.LORA_STRENGTH_MODEL, [])
        lora_hashes = inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, [])

        lora_names_from_prompt, lora_weights_from_prompt, lora_hashes_from_prompt = [], [], []
        if "<lora:" in prompt_joined:
            for text in prompt_texts:
                for _, name, weight in re.findall(lora_assertion_re, text.replace("\n", " ").replace("\r", " ")):
                    lora_names_from_prompt.append(("prompt_parse", name))
                    lora_weights_from_prompt.append(("prompt_parse", float(weight)))
                    h = calc_lora_hash(name)
                    # Preserve positional alignment even when a hash cannot be
                    # resolved; otherwise the next LoRA hash shifts left.
                    lora_hashes_from_prompt.append(("prompt_parse", h or ""))

        all_names = lora_names + lora_names_from_prompt
        all_weights = lora_weights + lora_weights_from_prompt
        all_hashes = lora_hashes + lora_hashes_from_prompt

        inputs_before_sampler_node[MetaField.LORA_MODEL_NAME] = all_names
        inputs_before_sampler_node[MetaField.LORA_STRENGTH_MODEL] = all_weights
        inputs_before_sampler_node[MetaField.LORA_MODEL_HASH] = all_hashes

        grouped = defaultdict(list)
        weights_by_node = _entries_by_node(all_weights)
        hashes_by_node = _entries_by_node(all_hashes)
        for name in all_names:
            node_key = str(name[0])
            weight_matches = weights_by_node.get(node_key)
            hash_matches = hashes_by_node.get(node_key)
            if not weight_matches or not hash_matches:
                continue
            weight = weight_matches.popleft()
            hsh = hash_matches.popleft()
            if not (name[1] and weight[1] is not None and hsh[1]):
                continue
            grouped[(hsh[1], weight[1])].append(clean_name(name[1]))

        hashes_in_prompt = {
            h[1].lower() for h in lora_hashes_from_prompt if h[1]
        }

        lora_strings, lora_hashes_list = [], []

        for (hsh, weight), names in grouped.items():
            canonical = min(names, key=len)
            present = hsh.lower() in hashes_in_prompt
            if not present:
                lora_strings.append(f"<lora:{canonical}:{weight}>")
            lora_hashes_list.append(f"{canonical}: {hsh}")

        updated_prompts = []
        if "<lora:" in prompt_joined:
            for text in prompt_texts:
                def replace(m):
                    tag, raw_name, weight = m.group(1), m.group(2), m.group(3)
                    return f"<{tag}:{clean_name(raw_name)}:{weight}>"
                updated_prompts.append(lora_assertion_re.sub(replace, text))
        else:
            updated_prompts = prompt_texts

        lora_hashes_string = ", ".join(lora_hashes_list)
        return lora_strings, lora_hashes_string, updated_prompts

    @classmethod
    def gen_pnginfo_dict(
        cls, inputs_before_sampler_node, inputs_before_this_node, prompt,
        save_civitai_sampler=True, batch_index=0, sampler_node_id=None,
        active_trace_tree=None,
    ):
        pnginfo = {}

        if not inputs_before_sampler_node:
            inputs_before_sampler_node = defaultdict(list)
            cls._collect_all_metadata(
                prompt, inputs_before_sampler_node,
                sampler_node_id=sampler_node_id,
                batch_index=batch_index,
            )

        hires_stage = _select_hires_upscale_stage(
            inputs_before_this_node, prompt, active_trace_tree
        )

        # ── PATCH: resolve prompts from graph when capture missed them ───────
        outputs = _get_outputs_cache()

        pos_list = inputs_before_sampler_node.get(MetaField.POSITIVE_PROMPT, [])
        neg_list = inputs_before_sampler_node.get(MetaField.NEGATIVE_PROMPT, [])
        current_positive = pos_list[0][1] if pos_list and len(pos_list[0]) > 1 else None
        current_negative = neg_list[0][1] if neg_list and len(neg_list[0]) > 1 else None

        # Resolve both roles through the actual sampler/guider path. Prefer the
        # graph when capture is missing/unresolved or when a value exactly
        # matches the opposite graph branch — the signature of branch leakage
        # seen with SamplerCustomAdvanced + CFGGuider + NegPip. Otherwise keep
        # the captured value, which may contain richer runtime-expanded text.
        graph_pos, graph_neg = _find_prompt_texts(
            prompt, outputs, batch_index=batch_index,
            sampler_node_id=sampler_node_id,
        )
        if _should_prefer_graph_prompt(current_positive, graph_pos, graph_neg):
            inputs_before_sampler_node[MetaField.POSITIVE_PROMPT] = [("graph", graph_pos)]
        if _should_prefer_graph_prompt(current_negative, graph_neg, graph_pos):
            inputs_before_sampler_node[MetaField.NEGATIVE_PROMPT] = [("graph", graph_neg)]
        # ─────────────────────────────────────────────────────────────────────

        def is_simple(value):
            return isinstance(value, (str, int, float, bool)) or value is None

        def extract(meta_key, label, source=inputs_before_sampler_node):
            val_list = source.get(meta_key, [])
            for link in val_list:
                if len(link) <= 1:
                    continue
                candidate = link[1]
                if candidate is None:
                    continue
                if isinstance(candidate, str):
                    if not candidate.strip():
                        continue
                elif not is_simple(candidate):
                    continue
                value = str(candidate)
                pnginfo[label] = value
                return value
            return None

        positive = extract(MetaField.POSITIVE_PROMPT, "Positive prompt") or ""
        if not positive.strip():
            print_warning("Positive prompt is empty!")

        negative = extract(MetaField.NEGATIVE_PROMPT, "Negative prompt") or ""
        lora_strings, lora_hashes, updated_prompts = cls.get_lora_strings_and_hashes(inputs_before_sampler_node)

        if updated_prompts and inputs_before_sampler_node.get(MetaField.POSITIVE_PROMPT):
            positive = updated_prompts[0]

        if lora_strings:
            positive += " " + " ".join(lora_strings)

        pnginfo["Positive prompt"] = positive.strip()
        pnginfo["Negative prompt"] = negative.strip()

        if not extract(MetaField.STEPS, "Steps"):
            # Fallback: read critical sampler fields directly from the prompt graph.
            # This handles the case where Trace found the sampler but CAPTURE_FIELD_LIST
            # didn't capture the fields (e.g. second run, async timing issue).
            # Prefer the sampler with denoise=1.0 (generation pass) over any
            # upscale/hires sampler with denoise < 1.0.
            _sampler_candidates = []
            for _nid, _node in prompt.items():
                _ni = _node.get("inputs", {})
                if "steps" in _ni and "sampler_name" in _ni and "cfg" in _ni:
                    if isinstance(_ni.get("steps"), (int, float)):
                        _denoise = _ni.get("denoise", 1.0)
                        if isinstance(_denoise, (int, float)):
                            _sampler_candidates.append((_nid, _ni, _denoise))
            # Sort: denoise=1.0 first (primary gen), then by steps descending
            _sampler_candidates.sort(
                key=lambda x: (-x[2] if x[2] >= 1.0 else x[2], -x[1].get("steps", 0))
            )
            if _sampler_candidates:
                _nid, _ni, _ = _sampler_candidates[0]
                inputs_before_sampler_node[MetaField.STEPS] = [(_nid, _ni["steps"])]
                if not inputs_before_sampler_node.get(MetaField.SAMPLER_NAME):
                    inputs_before_sampler_node[MetaField.SAMPLER_NAME] = [(_nid, _ni["sampler_name"])]
                if not inputs_before_sampler_node.get(MetaField.SCHEDULER):
                    inputs_before_sampler_node[MetaField.SCHEDULER] = [(_nid, _ni.get("scheduler", "normal"))]
                if not inputs_before_sampler_node.get(MetaField.CFG):
                    inputs_before_sampler_node[MetaField.CFG] = [(_nid, _ni["cfg"])]
                _seed = _ni.get("seed")
                if not inputs_before_sampler_node.get(MetaField.SEED):
                    if not _is_link(_seed):
                        inputs_before_sampler_node[MetaField.SEED] = [(_nid, _seed)]
                    else:
                        # Follow seed link chain to resolve actual value
                        _cur = _seed
                        _visited_seed = set()
                        while _is_link(_cur) and str(_cur[0]) not in _visited_seed:
                            _visited_seed.add(str(_cur[0]))
                            _src = prompt.get(str(_cur[0]))
                            if _src is None:
                                break
                            _si = _src.get("inputs", {})
                            for _sk in ("seed", "noise_seed", "value"):
                                _sv = _si.get(_sk)
                                if isinstance(_sv, (int, float)):
                                    inputs_before_sampler_node[MetaField.SEED] = [(_nid, int(_sv))]
                                    _cur = None
                                    break
                                elif _is_link(_sv):
                                    _cur = _sv
                                    break
                            else:
                                break
            if not extract(MetaField.STEPS, "Steps"):
                print_warning("Steps are empty, full metadata won't be added!")
                return {}

        # ── Sampler + Schedule type (Forge Neo splits them) ──────────────────
        samplers = inputs_before_sampler_node.get(MetaField.SAMPLER_NAME)
        schedulers = inputs_before_sampler_node.get(MetaField.SCHEDULER)
        sampler_pretty, schedule_pretty = cls.get_forge_sampler_and_schedule(
            samplers, schedulers
        )
        if sampler_pretty:
            pnginfo["Sampler"] = sampler_pretty
        if schedule_pretty:
            pnginfo["Schedule type"] = schedule_pretty

        # ── CFG scale (format as int when whole) ─────────────────────────────
        extract(MetaField.CFG, "CFG scale")
        cfg_val = pnginfo.get("CFG scale")
        if cfg_val is not None:
            try:
                f = float(cfg_val)
                pnginfo["CFG scale"] = str(int(f)) if f.is_integer() else str(f)
            except (ValueError, TypeError):
                pass

        # ── Seed ─────────────────────────────────────────────────────────────
        extract(MetaField.SEED, "Seed")

        # If seed is still missing, resolve it by following links through the graph
        if "Seed" not in pnginfo:
            _seed_resolved = None
            # Find the primary KSampler and follow its seed link
            for _nid, _node in prompt.items():
                _ni = _node.get("inputs", {})
                if _node.get("class_type") == "KSampler" and "steps" in _ni:
                    _seed_val = _ni.get("seed")
                    if isinstance(_seed_val, (int, float)):
                        _denoise = _ni.get("denoise", 1.0)
                        if isinstance(_denoise, (int, float)) and _denoise >= 1.0:
                            _seed_resolved = int(_seed_val)
                            break
                    elif _is_link(_seed_val):
                        # Follow the link chain to find the actual seed value
                        _visited_seed = set()
                        _cur = _seed_val
                        while _is_link(_cur) and str(_cur[0]) not in _visited_seed:
                            _visited_seed.add(str(_cur[0]))
                            _src_node = prompt.get(str(_cur[0]))
                            if _src_node is None:
                                break
                            _src_inputs = _src_node.get("inputs", {})
                            # Check for direct seed value on this node
                            for _sk in ("seed", "noise_seed", "value"):
                                _sv = _src_inputs.get(_sk)
                                if isinstance(_sv, (int, float)):
                                    _seed_resolved = int(_sv)
                                    break
                                elif _is_link(_sv):
                                    _cur = _sv
                                    break
                            else:
                                break
                            if _seed_resolved is not None:
                                break
                    if _seed_resolved is not None:
                        _denoise = _ni.get("denoise", 1.0)
                        if isinstance(_denoise, (int, float)) and _denoise >= 1.0:
                            break
            if _seed_resolved is not None:
                pnginfo["Seed"] = str(_seed_resolved)

        # ── Size (extracted before Model so order matches Forge Neo) ─────────
        image_width_data = inputs_before_sampler_node.get(MetaField.IMAGE_WIDTH, [[None]])
        image_height_data = inputs_before_sampler_node.get(MetaField.IMAGE_HEIGHT, [[None]])

        def extract_dimension(data):
            return data[0][1] if data and len(data[0]) > 1 and isinstance(data[0][1], int) else None

        width = extract_dimension(image_width_data)
        height = extract_dimension(image_height_data)
        if width and height:
            pnginfo["Size"] = f"{width}x{height}"

        # ── Model hash BEFORE Model (Forge Neo order) ────────────────────────
        extract(MetaField.MODEL_HASH, "Model hash")
        extract(MetaField.MODEL_NAME, "Model")
        model_name_val = pnginfo.get("Model")
        if model_name_val:
            pnginfo["Model"] = os.path.splitext(os.path.basename(model_name_val))[0]

        # ── VAE hash BEFORE VAE, strip extension ─────────────────────────────
        extract(MetaField.VAE_HASH, "VAE hash", inputs_before_this_node)
        extract(MetaField.VAE_NAME, "VAE", inputs_before_this_node)
        vae_name_val = pnginfo.get("VAE")
        if vae_name_val:
            pnginfo["VAE"] = os.path.splitext(os.path.basename(vae_name_val))[0]

        # ── Denoising strength ───────────────────────────────────────────────
        denoise = inputs_before_sampler_node.get(MetaField.DENOISE)
        dval = hires_stage.get("denoise") if hires_stage else None
        if dval is None:
            dval = denoise[0][1] if denoise else None
        if dval and 0 < float(dval) < 1:
            pnginfo["Denoising strength"] = float(dval)

        # ── Clip skip AFTER Denoising strength (Forge Neo order) ─────────────
        clip_skip = extract(MetaField.CLIP_SKIP, "Clip skip")
        if clip_skip is None:
            pnginfo["Clip skip"] = "1"

        # ── Hires fix ────────────────────────────────────────────────────────
        if hires_stage:
            if hires_stage.get("scale") is not None:
                pnginfo["Hires upscale"] = str(hires_stage["scale"])
            else:
                extract(MetaField.UPSCALE_BY, "Hires upscale", inputs_before_this_node)
            if hires_stage.get("name"):
                pnginfo["Hires upscaler"] = str(hires_stage["name"])
        else:
            extract(MetaField.UPSCALE_BY, "Hires upscale", inputs_before_this_node)
            extract(MetaField.UPSCALE_MODEL_NAME, "Hires upscaler", inputs_before_this_node)

        # ── LoRAs / embeddings ───────────────────────────────────────────────
        if lora_hashes:
            pnginfo["Lora hashes"] = f'"{lora_hashes}"'

        pnginfo.update(cls.gen_loras(inputs_before_sampler_node))
        pnginfo.update(cls.gen_embeddings(inputs_before_sampler_node))

        # ── Version signature (Forge Neo always writes a Version field) ──────
        pnginfo["Version"] = "ComfyUI"

        # NOTE: The Civitai-style "Hashes: {...}" JSON blob is intentionally
        # omitted here so the infotext matches Forge Neo byte-for-byte.
        # Model hash, VAE hash and Lora hashes are still present as individual
        # fields, which is what Civitai actually parses.

        return pnginfo

    @classmethod
    def _collect_all_metadata(
        cls, prompt, result_dict, sampler_node_id=None, batch_index=0
    ):
        # ── PATCH: use the graph-walk resolver for prompt texts ───────────────
        outputs = _get_outputs_cache()

        def _append_metadata(meta, node_id, value):
            if value is not None:
                result_dict[meta].append((node_id, value, 0))

        selected_sampler = (
            prompt.get(str(sampler_node_id)) if sampler_node_id is not None else None
        )
        sampler_with_fields = None
        if selected_sampler is not None:
            sampler_inputs = selected_sampler.get("inputs", {})
            if {"seed", "steps", "cfg", "sampler_name", "scheduler"} & set(sampler_inputs):
                sampler_with_fields = (str(sampler_node_id), selected_sampler)

        resolved = {
            "prompt": Trace.find_node_with_fields(prompt, {"positive", "negative"}),
            "denoise": Trace.find_node_with_fields(prompt, {"denoise"}),
            "sampler": sampler_with_fields or Trace.find_node_with_fields(
                prompt, {"seed", "steps", "cfg", "sampler_name", "scheduler"}
            ),
            "size": Trace.find_node_with_fields(prompt, {"width", "height"}),
            "model": Trace.find_node_with_fields(prompt, {"ckpt_name"}),
        }

        for node_id, node in Trace.find_all_nodes_with_fields(prompt, {"lora_name", "strength_model"}):
            if node is not None:
                inputs = node.get("inputs", {})
                name = inputs.get("lora_name")
                strength = inputs.get("strength_model")
                _append_metadata(MetaField.LORA_MODEL_NAME, node_id, name)
                _append_metadata(MetaField.LORA_MODEL_HASH, node_id, calc_lora_hash(name) if name else None)
                _append_metadata(MetaField.LORA_STRENGTH_MODEL, node_id, strength)

        model_node = resolved.get("model")
        if model_node and model_node[1] is not None:
            node_id, node = model_node
            inputs = node.get("inputs", {})
            name = inputs.get("ckpt_name")
            _append_metadata(MetaField.MODEL_NAME, node_id, name)
            _append_metadata(MetaField.MODEL_HASH, node_id, calc_model_hash(name) if name else None)

        denoise_node = resolved.get("denoise")
        if denoise_node and denoise_node[1] is not None:
            node_id, node = denoise_node
            val = node.get("inputs", {}).get("denoise")
            _append_metadata(MetaField.DENOISE, node_id, val)

        sampler_node = resolved.get("sampler")
        if sampler_node and sampler_node[1] is not None:
            node_id, node = sampler_node
            inputs = node.get("inputs", {})
            for key, meta in {
                "sampler_name": MetaField.SAMPLER_NAME,
                "scheduler": MetaField.SCHEDULER,
                "seed": MetaField.SEED,
                "steps": MetaField.STEPS,
                "cfg": MetaField.CFG,
            }.items():
                _append_metadata(meta, node_id, inputs.get(key))
        else:
            # ── SamplerCustomAdvanced topology ────────────────────────────────
            # Find any node that links to a GUIDER (cfg_guider input) and
            # has NOISE / SIGMAS / SAMPLER links — that is the top-level sampler.
            # Then trace its sub-nodes to gather seed, steps, cfg, sampler_name.
            custom_candidates = (
                [(str(sampler_node_id), selected_sampler)]
                if selected_sampler is not None else prompt.items()
            )
            for nid, node in custom_candidates:
                ni = node.get("inputs", {})
                if not (_is_link(ni.get("cfg_guider")) or _is_link(ni.get("guider"))):
                    continue
                # Found a SamplerCustomAdvanced-style node
                # Seed: follow noise link -> RandomNoise node
                noise_link = ni.get("noise")
                if _is_link(noise_link):
                    noise_node = prompt.get(str(noise_link[0]))
                    if noise_node:
                        seed = noise_node.get("inputs", {}).get("noise_seed")                                or noise_node.get("inputs", {}).get("seed")
                        _append_metadata(MetaField.SEED, str(noise_link[0]), seed)
                # Steps + scheduler: follow sigmas link -> BasicScheduler etc.
                sigmas_link = ni.get("sigmas")
                if _is_link(sigmas_link):
                    sig_node = prompt.get(str(sigmas_link[0]))
                    if sig_node:
                        sig_in = sig_node.get("inputs", {})
                        _append_metadata(MetaField.STEPS, str(sigmas_link[0]),
                                         sig_in.get("steps"))
                        _append_metadata(MetaField.SCHEDULER, str(sigmas_link[0]),
                                         sig_in.get("scheduler"))
                        _append_metadata(MetaField.DENOISE, str(sigmas_link[0]),
                                         sig_in.get("denoise"))
                # Sampler name: follow sampler link -> KSamplerSelect etc.
                sampler_link = ni.get("sampler")
                if _is_link(sampler_link):
                    samp_node = prompt.get(str(sampler_link[0]))
                    if samp_node:
                        samp_in = samp_node.get("inputs", {})
                        _append_metadata(MetaField.SAMPLER_NAME, str(sampler_link[0]),
                                         samp_in.get("sampler_name"))
                # CFG: follow cfg_guider link -> CFGGuider etc.
                guider_link = ni.get("cfg_guider") or ni.get("guider")
                if _is_link(guider_link):
                    g_node = prompt.get(str(guider_link[0]))
                    if g_node:
                        g_in = g_node.get("inputs", {})
                        _append_metadata(MetaField.CFG, str(guider_link[0]),
                                         g_in.get("cfg"))
                break  # Only process the first SamplerCustomAdvanced-style node

        size_node = resolved.get("size")
        if size_node and size_node[1] is not None:
            node_id, node = size_node
            inputs = node.get("inputs", {})
            for key, meta in {
                "width": MetaField.IMAGE_WIDTH,
                "height": MetaField.IMAGE_HEIGHT,
            }.items():
                _append_metadata(meta, node_id, inputs.get(key))

        # ── PATCHED prompt resolution ─────────────────────────────────────────
        # Uses _find_prompt_texts which handles both classic KSampler topology
        # (positive/negative on sampler) and SamplerCustomAdvanced topology
        # (positive/negative on CFGGuider, linked via cfg_guider).
        pos_text, neg_text = _find_prompt_texts(
            prompt, outputs, batch_index=batch_index,
            sampler_node_id=sampler_node_id,
        )
        found_prompts = bool(pos_text or neg_text)
        if pos_text:
            _append_metadata(MetaField.POSITIVE_PROMPT, "graph", pos_text)
        if neg_text:
            _append_metadata(MetaField.NEGATIVE_PROMPT, "graph", neg_text)
        for text in (pos_text, neg_text):
            if not text:
                continue
            for emb_name, emb_hash in zip(
                extract_embedding_names(text), extract_embedding_hashes(text)
            ):
                _append_metadata(MetaField.EMBEDDING_NAME, "graph", emb_name)
                _append_metadata(MetaField.EMBEDDING_HASH, "graph", emb_hash)

        # Final fallback – old behaviour preserved for edge-cases
        if not found_prompts:
            for node_id, node in Trace.find_all_nodes_with_fields(prompt, {"positive", "negative"}):
                if node is None:
                    continue
                inputs = node.get("inputs", {})
                pos_ref = inputs.get("positive", [None])[0]
                neg_ref = inputs.get("negative", [None])[0]

                def resolve_text(ref):
                    if isinstance(ref, list):
                        ref = ref[0]
                    if not isinstance(ref, str):
                        return None
                    n = prompt.get(ref)
                    if n is None:
                        return None
                    raw = n.get("inputs", {}).get("text")
                    if isinstance(raw, str):
                        return raw
                    return _resolve_text_from_graph(raw, prompt, outputs)

                pos_text = resolve_text(pos_ref)
                neg_text = resolve_text(neg_ref)
                _append_metadata(MetaField.POSITIVE_PROMPT, pos_ref, pos_text)
                _append_metadata(MetaField.NEGATIVE_PROMPT, neg_ref, neg_text)

                for text in (pos_text, neg_text):
                    if not text:
                        continue
                    for name, h in zip(extract_embedding_names(text), extract_embedding_hashes(text)):
                        _append_metadata(MetaField.EMBEDDING_NAME, node_id, name)
                        _append_metadata(MetaField.EMBEDDING_HASH, node_id, h)

    @classmethod
    def extract_model_info(cls, inputs, meta_field_name, prefix):
        model_info_dict = {}
        model_names = inputs.get(meta_field_name, [])
        hash_field = {
            MetaField.LORA_MODEL_NAME: MetaField.LORA_MODEL_HASH,
            MetaField.EMBEDDING_NAME: MetaField.EMBEDDING_HASH,
        }.get(meta_field_name)
        model_hashes = inputs.get(hash_field, []) if hash_field is not None else []

        for index, (model_name, model_hash) in enumerate(
            _pair_entries_by_node(model_names, model_hashes)
        ):
            field_prefix = f"{prefix}_{index}"
            model_info_dict[f"{field_prefix} name"] = os.path.splitext(os.path.basename(model_name[1]))[0]
            model_info_dict[f"{field_prefix} hash"] = model_hash[1]

        return model_info_dict

    @classmethod
    def gen_loras(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.LORA_MODEL_NAME, "Lora")

    @classmethod
    def gen_embeddings(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.EMBEDDING_NAME, "Embedding")

    @classmethod
    def gen_parameters_str(cls, pnginfo_dict):
        if not pnginfo_dict or not isinstance(pnginfo_dict, dict):
            return ""

        def clean_value(value):
            if value is None:
                return ""
            return str(value).strip().replace("\n", " ")

        def strip_embedding_prefix(text):
            return text.replace("embedding:", "")

        cleaned_dict = {k: clean_value(v) for k, v in pnginfo_dict.items()}

        pos = strip_embedding_prefix(cleaned_dict.get("Positive prompt", ""))
        neg = strip_embedding_prefix(cleaned_dict.get("Negative prompt", ""))

        result = [pos]
        if neg:
            result.append(f"Negative prompt: {neg}")

        s_list = [
            f"{k}: {v}"
            for k, v in cleaned_dict.items()
            if k not in {"Positive prompt", "Negative prompt"} and v not in {None, ""}
        ]

        result.append(", ".join(s_list))
        return "\n".join(result)

    @classmethod
    def get_hashes_for_civitai(cls, inputs_before_sampler_node, inputs_before_this_node):
        def extract_single(inputs, key):
            items = inputs.get(key, [])
            return items[0][1] if items and len(items[0]) > 1 else None

        def extract_named_hashes(names, hashes, prefix):
            result = {}
            for name, h in _pair_entries_by_node(names, hashes):
                if not h[1]:
                    continue
                base_name = os.path.splitext(os.path.basename(name[1]))[0]
                result[f"{prefix}:{base_name}"] = h[1]
            return result

        resource_hashes = {}

        model = extract_single(inputs_before_sampler_node, MetaField.MODEL_HASH)
        if model:
            resource_hashes["model"] = model

        vae = extract_single(inputs_before_this_node, MetaField.VAE_HASH)
        if vae:
            resource_hashes["vae"] = vae

        upscaler_hash = extract_single(inputs_before_this_node, MetaField.UPSCALE_MODEL_HASH)
        if upscaler_hash:
            resource_hashes["upscaler"] = upscaler_hash

        resource_hashes.update(extract_named_hashes(
            inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, []),
            inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, []),
            "lora"
        ))

        resource_hashes.update(extract_named_hashes(
            inputs_before_sampler_node.get(MetaField.EMBEDDING_NAME, []),
            inputs_before_sampler_node.get(MetaField.EMBEDDING_HASH, []),
            "embed"
        ))

        return resource_hashes

    # Pretty display names for ComfyUI scheduler enum values,
    # matching the "Schedule type" dropdown shown in Forge / Forge Neo.
    SCHEDULER_PRETTY = {
        "normal": "Normal",
        "karras": "Karras",
        "exponential": "Exponential",
        "sgm_uniform": "SGM Uniform",
        "simple": "Simple",
        "ddim_uniform": "DDIM",
        "beta": "Beta",
        "linear_quadratic": "Linear Quadratic",
        "kl_optimal": "KL Optimal",
        "polyexponential": "Polyexponential",
    }

    # Pretty display names for samplers (Civitai / A1111 naming).
    SAMPLER_PRETTY = {
        'euler': 'Euler',
        'euler_ancestral': 'Euler a',
        'heun': 'Heun',
        'dpm_2': 'DPM2',
        'dpm_2_ancestral': 'DPM2 a',
        'lms': 'LMS',
        'dpm_fast': 'DPM fast',
        'dpm_adaptive': 'DPM adaptive',
        'dpmpp_2s_ancestral': 'DPM++ 2S a',
        'dpmpp_sde': 'DPM++ SDE',
        'dpmpp_sde_gpu': 'DPM++ SDE',
        'dpmpp_2m': 'DPM++ 2M',
        'dpmpp_2m_sde': 'DPM++ 2M SDE',
        'dpmpp_2m_sde_gpu': 'DPM++ 2M SDE',
        'dpmpp_3m_sde': 'DPM++ 3M SDE',
        'dpmpp_3m_sde_gpu': 'DPM++ 3M SDE',
        'ddim': 'DDIM',
        'plms': 'PLMS',
        'uni_pc': 'UniPC',
        'uni_pc_bh2': 'UniPC',
        'lcm': 'LCM',
    }

    @classmethod
    def _pretty_scheduler(cls, scheduler):
        if not scheduler:
            return None
        if scheduler in cls.SCHEDULER_PRETTY:
            return cls.SCHEDULER_PRETTY[scheduler]
        # Fallback: turn snake_case into Title Case ("some_thing" -> "Some Thing")
        return " ".join(part.capitalize() for part in str(scheduler).split("_"))

    @classmethod
    def _pretty_sampler(cls, sampler):
        if not sampler:
            return None
        return cls.SAMPLER_PRETTY.get(sampler, sampler)

    @classmethod
    def get_forge_sampler_and_schedule(cls, sampler_names, schedulers):
        """
        Return (sampler_pretty, schedule_type_pretty) as two separate strings,
        matching Forge / Forge Neo's infotext format where Sampler and
        "Schedule type" are distinct fields.
        """
        sampler = None
        scheduler = None
        if sampler_names and len(sampler_names) > 0:
            sampler = sampler_names[0][1]
        if schedulers and len(schedulers) > 0:
            scheduler = schedulers[0][1]
        return cls._pretty_sampler(sampler), cls._pretty_scheduler(scheduler)

    @classmethod
    def get_sampler_for_civitai(cls, sampler_names, schedulers):
        """
        Legacy combined sampler name (sampler + scheduler merged), kept for
        backward compatibility with any caller that still expects it.
        """
        sampler_pretty, schedule_pretty = cls.get_forge_sampler_and_schedule(
            sampler_names, schedulers
        )
        if not sampler_pretty:
            return None
        if not schedule_pretty or schedule_pretty == "Normal":
            return sampler_pretty
        if schedule_pretty in ("Karras", "Exponential"):
            return f"{sampler_pretty} {schedule_pretty}"
        return f"{sampler_pretty}_{schedule_pretty.lower().replace(' ', '_')}"
