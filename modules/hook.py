from .nodes.node import SaveImageWithMetaData

current_prompt = {}
current_extra_data = {}
prompt_executer = None
current_save_image_node_id = -1
current_resolved_texts = {}


def record_resolved_text(node_id, text, list_index=None):
    global current_resolved_texts

    if not isinstance(text, str) or not text.strip():
        return

    node_key = str(node_id)
    if list_index is None:
        current_resolved_texts[node_key] = text
        return

    existing = current_resolved_texts.get(node_key)
    if isinstance(existing, list):
        values = existing
    elif isinstance(existing, str):
        values = [existing]
    else:
        values = []

    while len(values) <= list_index:
        values.append("")
    values[list_index] = text
    current_resolved_texts[node_key] = values


def pre_execute(self, prompt, prompt_id, extra_data, execute_outputs):
    global current_prompt
    global current_extra_data
    global prompt_executer
    global current_resolved_texts

    current_prompt = prompt
    current_extra_data = extra_data
    prompt_executer = self
    current_resolved_texts = {}

    # Clear the trace cache so stale data from a previous generation
    # (different workflow / different prompt) doesn't leak through.
    from .trace import Trace
    Trace._trace_cache.clear()


def pre_get_input_data(inputs, class_def, unique_id, *args):
    global current_save_image_node_id

    if class_def == SaveImageWithMetaData:
        current_save_image_node_id = unique_id
