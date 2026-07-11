import hashlib
import json
import os
import threading
from collections import OrderedDict

from ..config import NODE_CACHE_DIR
from .log import print_warning, print_error


CACHE_FILE = os.path.join(NODE_CACHE_DIR, "model_hash_cache.json")
CACHE_SIZE_LIMIT = 100
HASH_READ_CHUNK_SIZE = 1024 * 1024


# Keep the public memory cache values as hashes for backwards compatibility.
# Signatures are held separately so every memory hit can still be validated.
cache_model_hash = OrderedDict()
_memory_cache_signatures = {}
_disk_cache = {}
_disk_cache_dirty = False
_cache_lock = threading.Lock()


# Load cache from file on startup.
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            loaded_cache = json.load(f)
        if isinstance(loaded_cache, dict):
            _disk_cache = loaded_cache
        else:
            print_warning(f"Ignoring invalid cache data in {CACHE_FILE}")
    except Exception as e:
        print_error(f"Failed to load cache file {CACHE_FILE}: {e}")
        _disk_cache = {}


def _canonical_path(path):
    """Return the stable full-path identity used by both caches."""
    return os.path.normcase(
        os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path))))
    )


def get_file_mod_time(path):
    """Return the current mtime without caching stale filesystem state."""
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0


def _get_file_signature(path):
    stat_result = os.stat(path)
    return stat_result.st_mtime_ns, stat_result.st_size


def _record_matches(record, signature, allow_legacy=False):
    if not isinstance(record, dict) or not isinstance(record.get("file_hash"), str):
        return False

    mtime_ns, size = signature
    try:
        record_mtime_ns = record.get("file_modification_time_ns")
        record_size = record.get("file_size")
        if record_mtime_ns is not None and record_size is not None:
            return int(record_mtime_ns) == mtime_ns and int(record_size) == size

        # Old records only contain a floating-point mtime. They are safe to
        # migrate when they were already keyed by this exact canonical path.
        if allow_legacy and "file_modification_date" in record:
            return float(record["file_modification_date"]) == mtime_ns / 1_000_000_000
    except (TypeError, ValueError, OverflowError):
        return False

    return False


def _make_disk_record(file_hash, path, signature):
    mtime_ns, size = signature
    return {
        "file_hash": file_hash,
        # Retain this field so older plugin versions can still read new caches.
        "file_modification_date": get_file_mod_time(path),
        "file_modification_time_ns": mtime_ns,
        "file_size": size,
    }


def _remember_in_memory(key, file_hash, signature):
    cache_model_hash[key] = file_hash
    _memory_cache_signatures[key] = signature
    cache_model_hash.move_to_end(key)

    while len(cache_model_hash) > CACHE_SIZE_LIMIT:
        stale_key, _ = cache_model_hash.popitem(last=False)
        _memory_cache_signatures.pop(stale_key, None)


def trim_disk_cache():
    global _disk_cache
    if len(_disk_cache) > CACHE_SIZE_LIMIT:
        _disk_cache = dict(list(_disk_cache.items())[-CACHE_SIZE_LIMIT:])


def save_disk_cache():
    global _disk_cache_dirty
    if not _disk_cache_dirty:
        return
    try:
        trim_disk_cache()
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        temp_file = CACHE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(_disk_cache, f, indent=2)
        os.replace(temp_file, CACHE_FILE)
        _disk_cache_dirty = False
    except Exception as e:
        print_error(f"Failed to write cache to {CACHE_FILE}: {e}")


def _hash_file(path):
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(HASH_READ_CHUNK_SIZE), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:10]


def _hash_stable_file(path, initial_signature):
    """Avoid caching a mixed hash if a file changes while it is being read."""
    signature = initial_signature
    for _ in range(3):
        file_hash = _hash_file(path)
        current_signature = _get_file_signature(path)
        if current_signature == signature:
            return file_hash, current_signature
        signature = current_signature
    raise OSError(f"File changed repeatedly while hashing: {path}")


def calc_hash(filename, use_only_filename=True):
    """Calculate a short SHA-256 hash with a path- and stat-aware cache.

    ``use_only_filename`` is retained for API compatibility. Cache identities
    are always canonical full paths because basenames are not unique.
    """
    del use_only_filename
    global _disk_cache_dirty

    if not filename:
        print_warning(f"calc_hash: File not found or invalid path: {filename}")
        return ""

    try:
        key = _canonical_path(filename)
        if not os.path.isfile(key):
            raise FileNotFoundError(key)
        current_signature = _get_file_signature(key)
    except (OSError, TypeError, ValueError):
        print_warning(f"calc_hash: File not found or invalid path: {filename}")
        return ""

    with _cache_lock:
        if key in cache_model_hash:
            if _memory_cache_signatures.get(key) == current_signature:
                cache_model_hash.move_to_end(key)
                return cache_model_hash[key]
            cache_model_hash.pop(key, None)
            _memory_cache_signatures.pop(key, None)

        record = _disk_cache.get(key)
        if _record_matches(record, current_signature, allow_legacy=True):
            file_hash = record["file_hash"]
            _remember_in_memory(key, file_hash, current_signature)

            # Transparently upgrade exact-path records created by old versions.
            if "file_modification_time_ns" not in record or "file_size" not in record:
                _disk_cache[key] = _make_disk_record(file_hash, key, current_signature)
                _disk_cache_dirty = True
                save_disk_cache()
            return file_hash

    try:
        model_hash, stable_signature = _hash_stable_file(key, current_signature)

        with _cache_lock:
            _remember_in_memory(key, model_hash, stable_signature)
            new_record = _make_disk_record(model_hash, key, stable_signature)
            if _disk_cache.get(key) != new_record:
                _disk_cache[key] = new_record
                _disk_cache_dirty = True
            save_disk_cache()

        return model_hash
    except Exception as e:
        print_error(f"Failed to calculate hash for {filename}: {e}")
        return ""
