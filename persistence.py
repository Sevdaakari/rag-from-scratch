import os
import json
import hashlib

CACHE_PATH = "knowledge_base_cache.json"


def _docs_fingerprint(folder_path):
    """
    Compute a fingerprint of the docs/ folder's current contents, so we can
    tell whether a cached knowledge base is still valid or needs recomputing.

    Based on each file's name, size, and last-modified time -- cheap to
    compute, and it changes whenever a document is added, removed, or edited.
    """
    fingerprint_parts = []
    for filename in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, filename)
        stat = os.stat(full_path)
        fingerprint_parts.append(f"{filename}:{stat.st_size}:{stat.st_mtime}")
    combined = "|".join(fingerprint_parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def save_cache(all_chunks, docs_folder, cache_path=CACHE_PATH):
    """
    Save chunks (with their embeddings) to a JSON cache file, tagged with
    a fingerprint of the docs/ folder so a later run can tell whether the
    cache is still valid.

    Args:
        all_chunks (list[dict]): chunks with 'source', 'text', 'embedding'
        docs_folder (str): path to the docs/ folder this cache was built from
        cache_path (str): where to write the cache file
    """
    data = {
        "fingerprint": _docs_fingerprint(docs_folder),
        "chunks": all_chunks,
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)


def load_cache(docs_folder, cache_path=CACHE_PATH):
    """
    Load cached chunks+embeddings if the cache file exists AND the docs/
    folder hasn't changed since the cache was built.

    Args:
        docs_folder (str): path to the docs/ folder to validate against
        cache_path (str): where to look for the cache file

    Returns:
        list[dict] | None: cached chunks if the cache is valid, otherwise
        None (meaning the caller should recompute from scratch).
    """
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "r") as f:
        data = json.load(f)

    current_fingerprint = _docs_fingerprint(docs_folder)
    if data.get("fingerprint") != current_fingerprint:
        return None  # docs/ changed since the cache was built

    return data["chunks"]