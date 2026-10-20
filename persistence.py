import os
import json
import hashlib

CACHE_PATH = "knowledge_base_cache.json"


def _docs_fingerprint(folder_path, extra_tag=""):
    """
    Compute a fingerprint of the docs/ folder's current contents PLUS an
    optional extra_tag (e.g. which chunking strategy was used), so the
    cache is invalidated both when documents change AND when the chunking
    method changes -- otherwise switching chunkers would silently keep
    serving chunks built with the old one.
    """
    fingerprint_parts = [extra_tag]
    for filename in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, filename)
        stat = os.stat(full_path)
        fingerprint_parts.append(f"{filename}:{stat.st_size}:{stat.st_mtime}")
    combined = "|".join(fingerprint_parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def save_cache(all_chunks, docs_folder, extra_tag="", cache_path=CACHE_PATH):
    """
    Save chunks (with their embeddings) to a JSON cache file, tagged with
    a fingerprint of the docs/ folder AND extra_tag.
    """
    data = {
        "fingerprint": _docs_fingerprint(docs_folder, extra_tag),
        "chunks": all_chunks,
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)


def load_cache(docs_folder, extra_tag="", cache_path=CACHE_PATH):
    """
    Load cached chunks+embeddings if the cache file exists AND both the
    docs/ folder and extra_tag match what the cache was built with.

    Returns:
        list[dict] | None: cached chunks if valid, otherwise None.
    """
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "r") as f:
        data = json.load(f)

    current_fingerprint = _docs_fingerprint(docs_folder, extra_tag)
    if data.get("fingerprint") != current_fingerprint:
        return None

    return data["chunks"]