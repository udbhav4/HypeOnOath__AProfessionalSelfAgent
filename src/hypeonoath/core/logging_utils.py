"""
Shared JSON-Lines logging helper for router.py and converter.py.

Per the guideline: Step 3.2 and Step 3.3 each write to their own log file,
both in JSON Lines format (one JSON object per line). This module is the
single place that formats and appends those lines, so both steps stay
consistent without duplicating the append/format logic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_json_line(log_path: Path, **fields: Any) -> None:
    """
    Append one JSON object as a line to log_path, stamped with the current
    UTC time. Creates the parent directory if it doesn't exist yet.
    """
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **fields}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
