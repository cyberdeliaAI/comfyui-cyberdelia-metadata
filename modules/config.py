import os

# Define the cache directory relative to the root directory of the module
MODULES_ROOT_DIR = os.path.dirname(__file__)
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(MODULES_ROOT_DIR, '..'))  # Move one level up
NODE_CACHE_DIR = os.path.join(PROJECT_ROOT_DIR, ".cache")

# Make sure the cache directory exists
os.makedirs(NODE_CACHE_DIR, exist_ok=True)
