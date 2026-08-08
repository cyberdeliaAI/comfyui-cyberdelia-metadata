from .modules.nodes.node import SaveImageWithMetaData, CreateExtraMetaData

NODE_CLASS_MAPPINGS = {
    "CyberdeliaSaveImageWithMetaData": SaveImageWithMetaData,
    "CyberdeliaCreateExtraMetaData": CreateExtraMetaData,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CyberdeliaSaveImageWithMetaData": "Save Image With Metadata (Cyberdelia)",
    "CyberdeliaCreateExtraMetaData": "Create Extra Metadata (Cyberdelia)",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
