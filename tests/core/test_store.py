"""core.store: Chroma-safe metadata flattening and its exact inverse (3A.8, Section 5)."""
from __future__ import annotations

import pytest

from hypeonoath.core import store
from hypeonoath.core.store import restore_chunk

from tests.factories import code_chunk, doc_chunk


def test_flatten_is_scalar_only_and_omits_none():
    for chunk in (code_chunk(), doc_chunk()):
        flat = store.flatten_metadata(chunk)
        assert all(isinstance(v, (str, int, float, bool)) for v in flat.values())
        assert None not in flat.values() and "text" not in flat
    flat = store.flatten_metadata(code_chunk())
    assert flat["symbols"] == '["A.run", "load"]' and "date" not in flat
    assert store.flatten_metadata(doc_chunk())["heading_path"] == '["ABOUT ME", "Sub"]'


@pytest.mark.parametrize("chunk", [code_chunk(), doc_chunk()])
def test_restore_round_trip(chunk):
    restored = restore_chunk(chunk.chunk_id, chunk.text, store.flatten_metadata(chunk))
    assert restored == chunk


def test_unflattenable_value_raises():
    with pytest.raises(TypeError):
        store.flatten_metadata(code_chunk(language={"bad": 1}))
