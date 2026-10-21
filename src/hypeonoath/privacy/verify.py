"""
Post-chunk verification = find-and-fix (privacy plan v10, P-D46; was P-D13).

Defence in depth after block redaction: every final chunk is matched with
the SAME matcher (core.private_values, plain text matching -- no Presidio,
no spaCy) -- its text and every metadata string that becomes Chroma metadata
or can reach the prompt (heading path, project name / description, symbols).
A hit can only be a value the chunker re-joined from blocks that redaction
saw in pieces; it is redacted in place and counted. The build never fails
here. A chunk whose text changed gets a new id (ids hash the chunk text).

Returns labels and counts only -- never values.
"""
from __future__ import annotations

from collections import Counter

from hypeonoath.core import private_values as pv
from hypeonoath.core.records import Chunk, sha256_hex


def fix_chunks(chunks: list[Chunk], values: list[pv.PrivateValue]) -> Counter:
    """Redact every private value left in the chunks, in place. Returns label -> occurrences fixed."""
    fixed: Counter = Counter()
    if not values:
        return fixed

    def redact(text: str) -> str:
        redacted, spans = pv.redact(text, values)
        fixed.update(span.label for span in spans)
        return redacted

    for chunk in chunks:
        text = redact(chunk.text)
        if text != chunk.text:
            chunk.text = text
            chunk.chunk_id = sha256_hex(f"{chunk.chunk_id}:{sha256_hex(text)}")[:16]
        chunk.heading_path = [redact(h) for h in chunk.heading_path]
        chunk.symbols = [redact(s) for s in chunk.symbols]
        if chunk.project_name:
            chunk.project_name = redact(chunk.project_name)
        if chunk.project_description:
            chunk.project_description = redact(chunk.project_description)
    return fixed
