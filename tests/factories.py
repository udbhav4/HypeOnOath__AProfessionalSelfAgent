"""Small Chunk builders shared by the store and indexer tests."""
from __future__ import annotations

from hypeonoath.core.records import Chunk


def code_chunk(**overrides):
    base = dict(
        chunk_id="c1", text="# hdr\ncode", source_document="corpus/code/a.py", content_type="code",
        project_name="P", date=None, line_start=1, line_end=2, project_key="p", project_description="d",
        language="python", symbols=["A.run", "load"],
        symbol_spans=[{"name": "A.run", "line_start": 1, "line_end": 2, "part": 1, "total_parts": 2}],
        part=1, total_parts=2, group_id="g", chunk_method="ast", start_byte=0, end_byte=10,
    )
    base.update(overrides)
    return Chunk(**base)


def doc_chunk():
    return Chunk("d1", "ABOUT ME\ntext", "corpus/converted/cv.md", "prose", heading_path=["ABOUT ME", "Sub"])
