"""Slash-style dates are read under ONE day/month rule for the whole file.
The judge infers the rule only when a value forces it, refuses an ambiguous
file unless the format is stated, and refuses a file that forces both rules.
It never parses two rows of one file by different conventions."""

from __future__ import annotations

import pandas as pd
import pytest

from judge.ingest import IngestError, load_trades

HEAD = "symbol,side,entry_time,exit_time,entry_price,exit_price"


def _csv(*rows: str) -> bytes:
    return (HEAD + "\n" + "\n".join(rows) + "\n").encode()


def _entries(tr: pd.DataFrame) -> list[str]:
    return [t.strftime("%Y-%m-%d %H:%M") for t in tr["entry_time"]]


# ── the acceptance case from the audit ───────────────────────────────────────

def test_acceptance_03_02_and_13_02_parse_under_the_same_rule():
    """Before: 03/02 -> 2 March and 13/02 -> 13 February in one file."""
    tr, warns = load_trades(_csv(
        "LINKUSDT,long,03/02/2026 04:00,03/02/2026 07:00,20,20.2",
        "LINKUSDT,short,13/02/2026 04:00,13/02/2026 07:00,20,19.8"))
    assert _entries(tr) == ["2026-02-03 04:00", "2026-02-13 04:00"]
    assert any("day-first" in w and "applied to every row" in w for w in warns)


# ── DD/MM and MM/DD, forced by a field > 12 ──────────────────────────────────

def test_day_first_is_inferred_from_a_day_above_12():
    tr, warns = load_trades(_csv(
        "ETHUSDT,long,13/01/2026 04:00,13/01/2026 07:00,100,101",
        "ETHUSDT,long,05/01/2026 04:00,05/01/2026 07:00,100,101"))
    assert _entries(tr) == ["2026-01-05 04:00", "2026-01-13 04:00"]
    assert any("DD/MM/YYYY" in w for w in warns)


def test_month_first_is_inferred_from_a_day_above_12():
    tr, warns = load_trades(_csv(
        "ETHUSDT,long,01/13/2026 04:00,01/13/2026 07:00,100,101",
        "ETHUSDT,long,01/05/2026 04:00,01/05/2026 07:00,100,101"))
    assert _entries(tr) == ["2026-01-05 04:00", "2026-01-13 04:00"]
    assert any("MM/DD/YYYY" in w for w in warns)


def test_the_rule_is_taken_from_any_time_column():
    """Only the exit column has a forcing value; the entry column follows."""
    tr, _ = load_trades(_csv(
        "ETHUSDT,long,03/02/2026 04:00,13/02/2026 07:00,100,101"))
    assert _entries(tr) == ["2026-02-03 04:00"]


def test_dotted_european_dates_follow_the_same_rule():
    tr, _ = load_trades(_csv(
        "ETHUSDT,long,13.02.2026 04:00,13.02.2026 07:00,100,101",
        "ETHUSDT,long,03.02.2026 04:00,03.02.2026 07:00,100,101"))
    assert _entries(tr) == ["2026-02-03 04:00", "2026-02-13 04:00"]


# ── ambiguous file: refuse, or obey an explicit format ───────────────────────

def test_all_ambiguous_dates_are_refused_without_a_format():
    with pytest.raises(IngestError, match="ambiguous"):
        load_trades(_csv(
            "ETHUSDT,long,03/02/2026 04:00,03/02/2026 07:00,100,101",
            "ETHUSDT,long,04/02/2026 04:00,04/02/2026 07:00,100,101"))


@pytest.mark.parametrize("fmt, expected", [
    ("DMY", "2026-02-03 04:00"),
    ("MDY", "2026-03-02 04:00"),
])
def test_explicit_format_resolves_an_ambiguous_file(fmt, expected):
    tr, warns = load_trades(_csv(
        "ETHUSDT,long,03/02/2026 04:00,03/02/2026 07:00,100,101"),
        date_format=fmt)
    assert _entries(tr) == [expected]
    assert any(f"date_format={fmt}" in w for w in warns)


def test_explicit_format_that_contradicts_the_data_is_refused():
    with pytest.raises(IngestError, match="MDY was requested"):
        load_trades(_csv(
            "ETHUSDT,long,13/02/2026 04:00,13/02/2026 07:00,100,101"),
            date_format="MDY")


def test_unknown_format_value_is_refused():
    with pytest.raises(IngestError, match="date_format must be one of"):
        load_trades(_csv(
            "ETHUSDT,long,2026-02-03 04:00,2026-02-03 07:00,100,101"),
            date_format="YMD")


# ── mixed file: both interpretations forced -> refuse, never choose ──────────

def test_a_file_that_forces_both_rules_is_refused():
    with pytest.raises(IngestError, match="do not follow one rule"):
        load_trades(_csv(
            "ETHUSDT,long,13/02/2026 04:00,13/02/2026 07:00,100,101",
            "ETHUSDT,long,02/13/2026 04:00,02/13/2026 07:00,100,101"))


def test_an_explicit_format_cannot_override_a_mixed_file():
    with pytest.raises(IngestError, match="do not follow one rule"):
        load_trades(_csv(
            "ETHUSDT,long,13/02/2026 04:00,13/02/2026 07:00,100,101",
            "ETHUSDT,long,02/13/2026 04:00,02/13/2026 07:00,100,101"),
            date_format="DMY")


# ── ISO and epoch are untouched by the rule ──────────────────────────────────

def test_iso_dates_need_no_rule_and_raise_no_slash_warning():
    tr, warns = load_trades(_csv(
        "ETHUSDT,long,2026-02-03 04:00,2026-02-03 07:00,100,101",
        "ETHUSDT,long,2026-02-13 04:00,2026-02-13 07:00,100,101"))
    assert _entries(tr) == ["2026-02-03 04:00", "2026-02-13 04:00"]
    assert not any("slash-style" in w for w in warns)


def test_iso_dates_ignore_an_explicit_slash_format():
    """A DMY flag must not distort unambiguous ISO input."""
    tr, _ = load_trades(_csv(
        "ETHUSDT,long,2026-02-03 04:00,2026-02-03 07:00,100,101"),
        date_format="DMY")
    assert _entries(tr) == ["2026-02-03 04:00"]


def test_iso_and_slash_mixed_in_one_file_share_the_slash_rule():
    tr, _ = load_trades(_csv(
        "ETHUSDT,long,2026-02-03 04:00,2026-02-03 07:00,100,101",
        "ETHUSDT,long,13/02/2026 04:00,13/02/2026 07:00,100,101",
        "ETHUSDT,long,03/03/2026 04:00,03/03/2026 07:00,100,101"))
    assert _entries(tr) == ["2026-02-03 04:00", "2026-02-13 04:00",
                            "2026-03-03 04:00"]
