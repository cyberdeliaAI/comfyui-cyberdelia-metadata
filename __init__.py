from .modules.nodes.node import SaveImageWithMetaData, CreateExtraMetaData

node_definitions = [
    ("SaveImageWithMetaData", SaveImageWithMetaData, "Save Image With MetaData"),
    ("CreateExtraMetaData", CreateExtraMetaData, "Create Extra MetaData"),
]

NODE_CLASS_MAPPINGS = {f"{b}": c for b, c, _ in node_definitions}
NODE_DISPLAY_NAME_MAPPINGS = {f"{b}": f"{d}" for b, _, d in node_definitions}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]