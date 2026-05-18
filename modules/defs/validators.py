from collections import deque

from .samplers import SAMPLERS


# Class types that fully zero/neutralize a conditioning tensor. When the
# graph walk encounters one of these while looking upstream from a sampler's
# negative input, we must NOT keep walking — the text of any CLIPTextEncode
# above it is irrelevant because the tensor is zeroed at this point.
# Treating them as walk terminators results in an empty negative prompt,
# which matches what actually gets sampled.
ZERO_OUT_CLASS_TYPES = {
    "ConditioningZeroOut",
    "ConditioningZeroedOut",  # alternate spelling seen in some forks
}


# Passthrough wrapper nodes that bundle many inputs into many outputs. For
# these we MUST follow only the input that corresponds to the output slot we
# came in on — otherwise a "negative" walk through e.g. Context Big would
# happily step into the "positive" input and report the wrong text.
#
# Format: class_type -> {output_slot_index: input_name}
# Only the conditioning-relevant slots need to be listed; other slots fall
# back to the default "expand all inputs" behavior.
PASSTHROUGH_SLOT_TO_INPUT = {
    "Context Big (rgthree)": {
        1: "model",
        2: "clip",
        3: "vae",
        4: "positive",
        5: "negative",
        6: "latent",
    },
    "Context (rgthree)": {
        1: "model",
        2: "clip",
        3: "vae",
        4: "positive",
        5: "negative",
        6: "latent",
    },
    "Context Switch (rgthree)": {
        1: "model",
        2: "clip",
        3: "vae",
        4: "positive",
        5: "negative",
        6: "latent",
    },
    "Context Switch Big (rgthree)": {
        1: "model",
        2: "clip",
        3: "vae",
        4: "positive",
        5: "negative",
        6: "latent",
    },
    # ControlNet nodes pass conditioning through: output 0 = positive, 1 = negative
    "ControlNetApplyAdvanced": {
        0: "positive",
        1: "negative",
    },
    "ControlNetApply": {
        0: "positive",
        1: "negative",
    },
}


def is_positive_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    return node_id in _get_node_id_list(prompt, "positive")


def is_negative_prompt(node_id, obj, prompt, extra_data, outputs, input_data_all):
    return node_id in _get_node_id_list(prompt, "negative")


def _resolve_passthrough(prompt, node_id, slot, max_hops=10):
    """Follow passthrough/context wrapper nodes by output slot.

    Returns (resolved_node_id, resolved_slot). If the node isn't a known
    passthrough, returns (node_id, slot) unchanged. Handles chained wrappers
    up to max_hops deep.
    """
    for _ in range(max_hops):
        if node_id not in prompt:
            return node_id, slot
        class_type = prompt[node_id].get("class_type")
        slot_map = PASSTHROUGH_SLOT_TO_INPUT.get(class_type)
        if not slot_map or slot not in slot_map:
            return node_id, slot
        input_name = slot_map[slot]
        inputs = prompt[node_id].get("inputs", {})
        if input_name not in inputs:
            return node_id, slot
        link = inputs[input_name]
        if not isinstance(link, list) or len(link) < 2:
            return node_id, slot
        node_id, slot = link[0], link[1]
    return node_id, slot


def _get_node_id_list(prompt, field_name):
    node_id_list = {}
    for nid, node in prompt.items():
        for sampler_type, field_map in SAMPLERS.items():
            if node["class_type"] != sampler_type:
                continue
            if field_name not in field_map or field_map[field_name] not in node["inputs"]:
                continue

            src = node["inputs"][field_map[field_name]]
            if not isinstance(src, list) or len(src) < 2:
                continue

            # Resolve through passthrough wrappers (e.g. Context Big) using
            # the output slot, so we actually follow the negative branch.
            start_node, _start_slot = _resolve_passthrough(prompt, src[0], src[1])

            d = deque()
            d.append(start_node)
            visited = set()
            while d:
                nid2 = d.popleft()
                if nid2 in visited or nid2 not in prompt:
                    continue
                visited.add(nid2)
                class_type = prompt[nid2]["class_type"]

                # Zero-out terminates this branch without registering anything.
                if class_type in ZERO_OUT_CLASS_TYPES:
                    continue

                if "CLIPTextEncode" in class_type:
                    node_id_list[nid] = nid2
                    break

                # For intermediate passthrough wrappers encountered mid-walk
                # (rare, but possible), follow only the matching slot if the
                # parent link is known. Otherwise fall back to expanding all
                # conditioning-like inputs.
                inputs = prompt[nid2].get("inputs", {})
                for k, v in inputs.items():
                    if isinstance(v, list) and v:
                        d.append(v[0])

    return list(node_id_list.values())
