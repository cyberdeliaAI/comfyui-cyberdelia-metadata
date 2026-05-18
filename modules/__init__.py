import asyncio
import functools
import importlib
from comfy_execution.utils import get_executing_context

from . import hook
from .hook import pre_execute, pre_get_input_data
import execution


_comfy_nodes = importlib.import_module("nodes")


def _make_async_safe_wrapper(function, prefunction):
    if asyncio.iscoroutinefunction(function):
        @functools.wraps(function)
        async def async_run(*args, **kwargs):
            try:
                prefunction(*args, **kwargs)
            except Exception:
                pass
            return await function(*args, **kwargs)
        return async_run
    else:
        @functools.wraps(function)
        def sync_run(*args, **kwargs):
            try:
                prefunction(*args, **kwargs)
            except Exception:
                pass
            return function(*args, **kwargs)
        return sync_run


# Wrap PromptExecutor.execute so pre_execute fires before each generation.
execution.PromptExecutor.execute = _make_async_safe_wrapper(
    execution.PromptExecutor.execute, pre_execute
)

# Wrap get_input_data so pre_get_input_data fires (sets current_save_image_node_id).
if hasattr(execution, "get_input_data"):
    execution.get_input_data = _make_async_safe_wrapper(
        execution.get_input_data, pre_get_input_data
    )


# ---------------------------------------------------------------------------
# Intercept CLIPTextEncode.encode at runtime to capture the *resolved* prompt
# text (after wildcard expansion, dynamic prompt generation, etc.).
# This is the most reliable source for the actual text that was encoded.
# ---------------------------------------------------------------------------
def _wrap_clip_text_encode():
    clip_text_encode = getattr(_comfy_nodes, "CLIPTextEncode", None)
    original_encode = getattr(clip_text_encode, "encode", None)
    if original_encode is None or getattr(original_encode, "__metadata_capture_wrapped__", False):
        return

    @functools.wraps(original_encode)
    def wrapped_encode(self, clip, text):
        try:
            context = get_executing_context()
            if context is not None:
                hook.record_resolved_text(context.node_id, text, context.list_index)
        except Exception:
            pass
        return original_encode(self, clip, text)

    wrapped_encode.__metadata_capture_wrapped__ = True
    clip_text_encode.encode = wrapped_encode


_wrap_clip_text_encode()