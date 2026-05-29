"""Tests for affinity_bot.extract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from affinity_bot.extract import (
    notes_from_csv,
    notes_from_json,
    notes_from_path,
    notes_from_strings,
    notes_from_text,
)


def test_strings_basic():
    notes = notes_from_strings(["one", "two", "three"])
    assert [n.text for n in notes] == ["one", "two", "three"]
    assert [n.id for n in notes] == ["n0001", "n0002", "n0003"]


def test_strings_dedupes_case_insensitive():
    notes = notes_from_strings(["Hello", "hello", "HELLO"])
    assert len(notes) == 1
    assert notes[0].text == "Hello"


def test_strings_skips_empty_and_whitespace():
    notes = notes_from_strings(["", "   ", "real", "\n\n"])
    assert [n.text for n in notes] == ["real"]


def test_text_with_bullets():
    text = "- onboarding broken\n* sign up too long\n1. nps awkward\n• mobile slow"
    notes = notes_from_text(text)
    assert [n.text for n in notes] == [
        "onboarding broken",
        "sign up too long",
        "nps awkward",
        "mobile slow",
    ]


def test_text_strips_enclosing_quotes():
    text = '"first note"\n\'second note\''
    notes = notes_from_text(text)
    assert [n.text for n in notes] == ["first note", "second note"]


def test_text_collapses_internal_whitespace():
    notes = notes_from_text("a    b\t\tc")
    assert notes[0].text == "a b c"


def test_csv_auto_detects_note_column():
    csv = "id,note,extra\n1,first note,foo\n2,second note,bar"
    notes = notes_from_csv(csv)
    assert [n.text for n in notes] == ["first note", "second note"]


def test_csv_explicit_column():
    csv = "id,a,b\n1,foo,bar\n2,baz,qux"
    notes = notes_from_csv(csv, column="b")
    assert [n.text for n in notes] == ["bar", "qux"]


def test_csv_falls_back_to_first_column_when_no_match():
    csv = "field1,field2\nfirst,foo\nsecond,bar"
    notes = notes_from_csv(csv)
    assert [n.text for n in notes] == ["first", "second"]


def test_csv_bad_column_raises():
    csv = "note\nhi"
    with pytest.raises(ValueError, match="not in CSV header"):
        notes_from_csv(csv, column="missing")


def test_csv_no_header_raises():
    with pytest.raises(ValueError, match="no header"):
        notes_from_csv("")


def test_json_list_of_strings():
    notes = notes_from_json(json.dumps(["a", "b", "c"]))
    assert [n.text for n in notes] == ["a", "b", "c"]


def test_json_list_of_objects_with_text():
    payload = json.dumps([{"text": "hi"}, {"note": "there"}, {"observation": "again"}])
    notes = notes_from_json(payload)
    assert [n.text for n in notes] == ["hi", "there", "again"]


def test_json_rejects_non_list():
    with pytest.raises(ValueError, match="must be a list"):
        notes_from_json(json.dumps({"text": "wrong"}))


def test_path_dispatch_txt(tmp_path):
    p: Path = tmp_path / "notes.txt"
    p.write_text("first\nsecond\n")
    notes = notes_from_path(p)
    assert len(notes) == 2
    assert notes[0].source == "notes.txt"


def test_path_dispatch_csv(tmp_path):
    p: Path = tmp_path / "data.csv"
    p.write_text("note\nhello\nworld\n")
    notes = notes_from_path(p)
    assert [n.text for n in notes] == ["hello", "world"]


def test_path_dispatch_json(tmp_path):
    p: Path = tmp_path / "data.json"
    p.write_text(json.dumps(["one", "two"]))
    notes = notes_from_path(p)
    assert [n.text for n in notes] == ["one", "two"]


def test_path_dispatch_unsupported(tmp_path):
    p: Path = tmp_path / "data.xls"
    p.write_text("garbage")
    with pytest.raises(ValueError, match="Unsupported"):
        notes_from_path(p)


def test_ids_have_stable_format():
    notes = notes_from_strings([f"note {i}" for i in range(15)])
    assert notes[0].id == "n0001"
    assert notes[14].id == "n0015"
    assert len(notes[0].id) == 5
