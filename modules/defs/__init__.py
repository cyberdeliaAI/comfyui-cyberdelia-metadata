import os
from .captures import CAPTURE_FIELD_LIST
from .samplers import SAMPLERS
from .loader import load_extensions

# load CAPTURE_FIELD_LIST and SAMPLERS in ext folder
ext_dir = os.path.join(os.path.dirname(__file__), "ext")
load_extensions(ext_dir, __package__, CAPTURE_FIELD_LIST, SAMPLERS)
