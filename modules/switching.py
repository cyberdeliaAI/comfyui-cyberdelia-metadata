"""Follow metadata-relevant routes without executing workflow nodes."""

from .composer_metadata import COMPOSER_CLASS


def resolve_boolean(value, prompt, visited=frozenset()):
    if isinstance(value, bool):
        return value
    if not (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], (str, int)) and not isinstance(value[0], bool)
            and isinstance(value[1], int) and value[1] == 0 and not isinstance(value[1], bool)):
        return None
    node_id = value[0] if value[0] in prompt else str(value[0])
    if node_id in visited:
        return None
    node = prompt.get(node_id, {})
    if node.get('class_type') == 'PrimitiveBoolean':
        return resolve_boolean(node.get('inputs', {}).get('value'), prompt, visited | {node_id})
    return None


def selected_switch_input(node, prompt):
    if node.get('class_type') != 'ComfySwitchNode':
        return None
    selected = resolve_boolean(node.get('inputs', {}).get('switch'), prompt)
    return ('on_true' if selected else 'on_false') if selected is not None else None


def active_inputs(node, prompt):
    inputs = node.get('inputs', {})
    selected = selected_switch_input(node, prompt)
    if selected is None:
        return inputs
    return {key: value for key, value in inputs.items()
            if key not in {'on_false', 'on_true'} or key == selected}


def metadata_source_link(value, prompt, visited=frozenset()):
    """Composer LATENT is a pass-through, not a prompt/model contribution.

    Masks and the diagnostic report do not apply character conditioning or
    LoRAs either. Keep normal traversal for the model and conditioning outputs.
    """
    if not (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], (str, int)) and not isinstance(value[0], bool)
            and isinstance(value[1], int) and not isinstance(value[1], bool)):
        return value
    node_id = value[0] if value[0] in prompt else str(value[0])
    node = prompt.get(node_id, {})
    if node.get('class_type') != COMPOSER_CLASS:
        return value
    slot = value[1]
    if slot in (3, 5):
        return None
    if slot != 2:
        return value
    if node_id in visited:
        return None
    return metadata_source_link(node.get('inputs', {}).get('latent'), prompt, visited | {node_id})
