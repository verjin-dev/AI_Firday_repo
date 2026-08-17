"""Helper tests — JSON recovery, page mapping, heading detection, truncation."""

from __future__ import annotations

import pytest

from backend.utils.helpers import (
    dedupe_preserving_order,
    extract_json,
    is_heading,
    page_for_offset,
    sanitize_metadata,
    slugify,
    truncate_at_sentence,
)


# ----------------------------------------------------------------------
# extract_json — LLMs return JSON wrapped, prefixed, or cut off mid-write.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": [1, 2]}', {"a": [1, 2]}),
        ('```json\n{"a": ["x"]}\n```', {"a": ["x"]}),
        ('```\n{"a": ["x"]}\n```', {"a": ["x"]}),
        ('Sure, here it is:\n{"a": ["x"]}\nHope that helps.', {"a": ["x"]}),
        ('{"note": "a { brace", "ok": [1]}', {"note": "a { brace", "ok": [1]}),
        ('[{"a": 1}, {"b": 2}]', [{"a": 1}, {"b": 2}]),
    ],
)
def test_extract_json_wellformed(raw, expected):
    assert extract_json(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Cut mid-string: the completed members survive.
        (
            '{"people": ["Alice", "Bob"], "orgs": ["Acme Corp',
            {"people": ["Alice", "Bob"]},
        ),
        # Cut after a trailing comma inside an array.
        (
            '{"people": ["Alice", "Bob"], "orgs": ["Acme",',
            {"people": ["Alice", "Bob"], "orgs": ["Acme"]},
        ),
        # Missing only the closing brace.
        (
            '{"people": ["Alice"], "dates": ["2024-01-01"]',
            {"people": ["Alice"], "dates": ["2024-01-01"]},
        ),
        # Cut immediately after a key.
        ('{"people": ["Alice"], "orgs":', {"people": ["Alice"]}),
        # Truncated array of objects — the complete object survives.
        (
            '{"indicators": [{"type": "dup", "severity": "HIGH"}, {"type": "inc',
            {"indicators": [{"type": "dup", "severity": "HIGH"}]},
        ),
        ('[{"a": 1}, {"b":', [{"a": 1}]),
    ],
)
def test_extract_json_recovers_truncated_output(raw, expected):
    """A response cut off by max_tokens must not discard what was parsed."""
    assert extract_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "no json at all", "{{{{", None])
def test_extract_json_returns_none_when_unrecoverable(raw):
    assert extract_json(raw) is None


def test_extract_json_handles_escaped_quotes():
    assert extract_json(r'{"q": "he said \"hi\"", "n": [2]}') == {
        "q": 'he said "hi"',
        "n": [2],
    }


def test_extract_json_prefers_root_object_over_nested_array():
    """A nested array must never be returned in place of its parent object."""
    result = extract_json('{"people": ["Alice", "Bob"], "orgs": ["Acme')
    assert isinstance(result, dict), "must not return the inner array"


# ----------------------------------------------------------------------
def test_page_for_offset():
    text = "\n\n--- PAGE 1 ---\n\nalpha\n\n--- PAGE 2 ---\n\nbravo"
    assert page_for_offset(text, text.index("alpha")) == 1
    assert page_for_offset(text, text.index("bravo")) == 2


def test_page_for_offset_before_any_marker_defaults_to_one():
    assert page_for_offset("no markers here", 5) == 1


@pytest.mark.parametrize(
    "line,expected",
    [
        ("SECTION 31. PRIOR CLAIMS HISTORY", True),
        ("## Markdown heading", True),
        ("1.2 Definitions And Scope", True),
        ("ALL CAPS HEADING", True),
        ("This is an ordinary sentence.", False),
        ("", False),
    ],
)
def test_is_heading(line, expected):
    assert (is_heading(line) is not None) is expected


def test_truncate_at_sentence_prefers_boundary():
    text = "First sentence here. Second sentence here. Third one."
    assert truncate_at_sentence(text, 30).endswith(".")


def test_truncate_returns_input_when_short_enough():
    assert truncate_at_sentence("short", 100) == "short"


def test_dedupe_is_case_insensitive_and_order_preserving():
    assert dedupe_preserving_order(["Alice", "alice", "Bob", " Alice "]) == [
        "Alice",
        "Bob",
    ]


def test_sanitize_metadata_coerces_to_scalars():
    clean = sanitize_metadata(
        {"a": ["x", "y"], "b": None, "c": 3, "d": "s", "e": True, "f": {"k": "v"}}
    )
    assert clean["a"] == "x, y"
    assert "b" not in clean, "None values are dropped"
    assert all(isinstance(v, (str, int, float, bool)) for v in clean.values())


def test_slugify():
    assert slugify("Sample Insurance Claim!.txt") == "sample-insurance-claim-txt"
