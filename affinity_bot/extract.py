"""Parse research notes from a variety of input shapes.

Supported inputs
----------------
* ``.txt`` / ``.md``  — one note per non-empty line, with bullet markers stripped
* ``.csv``            — one column of notes; column auto-detected or chosen with --column
* ``.json``           — list of strings, OR list of objects with ``text`` key
* In-memory list[str] — passed directly to :func:`notes_from_strings`

Notes are normalised but not aggressively edited: trailing whitespace,
common bullet markers, and surrounding quotes are stripped, and the
result is deduplicated on the normalised text. Order is preserved.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from affinity_bot.models import Note

_BULLET_RE = re.compile(r"^\s*(?:[-*•·▪▫►‣–—]|\d+[\.\)])\s+")
_WS_RE = re.compile(r"\s+")
_NOTE_ID_FMT = "n{:04d}"


def _normalise_text(raw: str) -> str:
    """Trim whitespace, strip a leading bullet, collapse internal runs."""
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    text = _BULLET_RE.sub("", text)
    # Strip a single pair of enclosing quotes (common in CSV exports)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "“", "”"}:
        text = text[1:-1].strip()
    return _WS_RE.sub(" ", text)


def notes_from_strings(
    texts: Iterable[str], source: Optional[str] = None
) -> list[Note]:
    """Build a deduplicated, indexed list of :class:`Note` from raw strings."""
    seen: set[str] = set()
    notes: list[Note] = []
    counter = 1
    for raw in texts:
        text = _normalise_text(raw)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        notes.append(Note(id=_NOTE_ID_FMT.format(counter), text=text, source=source))
        counter += 1
    return notes


def notes_from_text(text: str, source: Optional[str] = None) -> list[Note]:
    """Parse a plain-text blob — one note per non-empty line."""
    return notes_from_strings(text.splitlines(), source=source)


def notes_from_csv(
    data: str | Path,
    column: Optional[str] = None,
    source: Optional[str] = None,
) -> list[Note]:
    """Read a CSV string or file path. One column of notes is selected."""
    if isinstance(data, Path):
        content = data.read_text(encoding="utf-8")
        source = source or data.name
    else:
        content = data
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row")
    chosen = column or _pick_csv_column(reader.fieldnames)
    if chosen not in reader.fieldnames:
        raise ValueError(
            f"column {chosen!r} not in CSV header {list(reader.fieldnames)}"
        )
    return notes_from_strings(
        (row.get(chosen, "") for row in reader), source=source
    )


def _pick_csv_column(fieldnames: list[str]) -> str:
    """Heuristic: pick a sensible default note column."""
    candidates = [
        "note",
        "notes",
        "text",
        "observation",
        "observations",
        "quote",
        "quotes",
        "comment",
        "comments",
        "response",
        "open_end",
    ]
    lower_map = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return fieldnames[0]


def notes_from_json(
    data: str | Path, source: Optional[str] = None
) -> list[Note]:
    """Read a JSON file or string: a list of strings or list of objects."""
    if isinstance(data, Path):
        content = data.read_text(encoding="utf-8")
        source = source or data.name
    else:
        content = data
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError(
            "JSON input must be a list (of strings, or of objects with 'text')"
        )
    out: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            t = item.get("text") or item.get("note") or item.get("observation")
            if t:
                out.append(str(t))
    return notes_from_strings(out, source=source)


def notes_from_path(
    path: Path,
    column: Optional[str] = None,
) -> list[Note]:
    """Dispatch on file extension."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ""}:
        return notes_from_text(path.read_text(encoding="utf-8"), source=path.name)
    if suffix == ".csv":
        return notes_from_csv(path, column=column)
    if suffix == ".json":
        return notes_from_json(path)
    raise ValueError(f"Unsupported file type {suffix!r}; use .txt, .md, .csv, or .json")
