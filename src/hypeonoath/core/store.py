"""
The Chroma store contract shared by indexing (writes) and serving (reads).

  * get_client(): PersistentClient on disk, telemetry off -- a plain folder,
    so design doc Section 9.1 (Option A) can commit it and redeploy on push.
  * Metadata rules, kept side by side because they are each other's inverse:
      flatten_metadata()  Chunk -> Chroma-safe scalars (str/int/float/bool):
                          every list field -> a JSON string (no separator can
                          collide with a heading or symbol name), None -> key
                          omitted.
      restore_chunk()     the stored row -> the original Chunk, exactly.

Lives in core (not in indexing) so the serving side can read the index
without importing any indexing module or its heavy dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hypeonoath.core import config
from hypeonoath.core.records import Chunk, json_dumps_stable

# List-valued fields, stored as JSON strings.
JSON_FIELDS = ("symbols", "heading_path", "symbol_spans")


def get_client(chroma_dir: Path | None = None):
    chroma_dir = config.CHROMA_DIR if chroma_dir is None else chroma_dir
    import chromadb
    from chromadb.config import Settings

    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))


def flatten_metadata(chunk: Chunk) -> dict[str, Any]:
    """Every Chunk field except `text`, as str/int/float/bool only."""
    flat: dict[str, Any] = {}
    for key, value in chunk.to_dict().items():
        if key == "text" or value is None:
            continue
        if key in JSON_FIELDS:
            if value:
                flat[key] = json_dumps_stable(value)
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            raise TypeError(f"Unflattenable metadata {key}={value!r} on chunk {chunk.chunk_id}")
    return flat


def restore_chunk(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    """Inverse of flatten_metadata."""
    fields = dict(metadata)
    for key in JSON_FIELDS:
        fields[key] = json.loads(fields[key]) if fields.get(key) else []
    known = set(Chunk.__dataclass_fields__)
    fields = {k: v for k, v in fields.items() if k in known}
    fields.update(chunk_id=chunk_id, text=text)
    return Chunk(**fields)
