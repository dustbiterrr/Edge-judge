"""C6 look-ahead: signal_time vs entry_time on the log's own timestamps.

Three states.  PASS: every entry is strictly after its signal and outside
the signal's 1h bar.  FAIL: any signal at/after entry, or any entry inside
the signal's bar - counted and named by CSV line.  UNVERIFIABLE: no
signal_time at all - reported in the verdict line, the summary and the
notes, never folded into a PASS."""

from __future__ import annotations

import pandas as pd
import pytest

from judge.checks import UNVERIFIABLE, audit
from judge.ingest import TEMPLATE_CSV, IngestError, load_trades
from judge.report import to_markdown, verdict_line, what_this_means

T0 = pd.Timestamp("2026-01-05 00:00", tz="UTC")
H = pd.Timedelta(hours=1)
M = pd.Timedelta(minutes=1)


def log(n: int = 120, signal=None, entry=None, header: str = "signal_time",
        with_signal: bool = True) -> bytes:
    """n non-overlapping ETHUSDT longs, +0.5% gross each, two hours apart.
    signal(i, entry_ts) / entry(i, base_ts) override the timing per trade.
    Default: entry at the top of an hour, signal one hour earlier."""
    cols = "symbol,side,entry_time,exit_time,entry_price,exit_price"
    if with_signal:
        cols = f"symbol,side,{header},entry_time,exit_time,entry_price,exit_price"
    rows = [cols]
    for i in range(n):
        base = T0 + 2 * i * H
        e = entry(i, base) if entry else base
        s = signal(i, e) if signal else e - H
        x = base + H + 30 * M                # exit after the entry bar
        f = "%Y-%m-%d %H:%M"
        cells = ["ETHUSDT", "long"]
        if with_signal:
            cells.append(s.strftime(f))
        cells += [e.strftime(f), x.strftime(f), "100.0", "100.5"]
        rows.append(",".join(cells))
    return ("\n".join(rows) + "\n").encode()


def c6(res):
    c = res.checks[5]
    assert c.code == "C6"
    return c


# ── UNVERIFIABLE: the state that must never look like a pass ─────────────────

def test_no_signal_time_is_unverifiable_not_pass(offline_klines):
    tr, _ = load_trades(log(with_signal=False))
    res = audit(tr, 0.11)
    c = c6(res)
    assert c.status == UNVERIFIABLE and not c.passed
    assert c.key_number == "no signal_time column"
    assert res.unverifiable == [c] and len(res.evaluated) == 5


def test_unverifiable_is_named_in_verdict_line_summary_and_notes(offline_klines):
    tr, _ = load_trades(log(with_signal=False))
    res = audit(tr, 0.11)
    assert res.verdict == "PASS"                     # C1-C5 do pass here
    line = verdict_line(res)
    assert line.startswith("PASS — 5 of 5 evaluated checks passed")
    assert "C6 look-ahead UNVERIFIABLE (no signal_time column)" in line
    text = what_this_means(res)
    assert "could NOT be checked" in text
    assert "coin flip does not explain" not in text  # no clean endorsement
    assert any("C6 look-ahead UNVERIFIABLE" in w for w in res.warnings)
    md = to_markdown(res)
    assert "| C6 | Look-ahead | **UNVERIFIABLE** |" in md
    assert "- none" not in md.split("## Notes")[1]


def test_unverifiable_also_rides_along_with_a_fail(offline_klines):
    tr, _ = load_trades(log(n=20, with_signal=False))      # C1 fails
    res = audit(tr, 0.11)
    assert res.verdict == "FAIL"
    line = verdict_line(res)
    assert line.startswith("FAIL — ") and "of 5 checks failed" in line
    assert "C6 look-ahead UNVERIFIABLE" in line
    assert "could NOT be checked" in what_this_means(res)


def test_partial_signal_time_column_is_refused_at_ingest():
    raw = log(n=3).decode().splitlines()
    cells = raw[2].split(",")
    cells[2] = ""                                            # blank signal
    raw[2] = ",".join(cells)
    with pytest.raises(IngestError, match="signal_time column must be complete"):
        load_trades(("\n".join(raw) + "\n").encode())


# ── PASS: entry after the signal, in a later bar ─────────────────────────────

def test_signal_one_bar_before_entry_passes(offline_klines):
    tr, _ = load_trades(log())
    res = audit(tr, 0.11)
    c = c6(res)
    assert c.status == "PASS" and c.passed and c.key_number == "0/120 trades"
    assert res.verdict == "PASS" and res.unverifiable == []
    assert verdict_line(res).startswith("PASS — all 6 checks passed")
    assert what_this_means(res).startswith("All six pre-registered checks passed")


def test_signal_mid_bar_and_entry_at_next_bar_open_passes(offline_klines):
    """Decision at 08:37, entry at 09:00 (open of the next bar): clean."""
    tr, _ = load_trades(log(signal=lambda i, e: e - 23 * M))
    assert c6(audit(tr, 0.11)).passed


def test_template_csv_carries_signal_time_and_is_clean(offline_klines):
    tr, _ = load_trades(TEMPLATE_CSV.encode())
    assert tr["signal_time"].notna().all()
    assert c6(audit(tr, 0.11)).passed


# ── FAIL: decision logged at/after entry, or entry inside the signal bar ─────

def test_signal_equal_to_entry_fails(offline_klines):
    tr, _ = load_trades(log(signal=lambda i, e: e))
    c = c6(audit(tr, 0.11))
    assert c.status == "FAIL" and c.key_number == "120/120 trades"
    assert "signal_time at or after entry_time" in c.detail
    assert "+0 min relative to entry" in c.detail


def test_signal_after_entry_fails_and_names_the_worst_case(offline_klines):
    def sig(i, e):
        return e + (45 * M if i == 7 else 5 * M)
    tr, _ = load_trades(log(signal=sig))
    c = c6(audit(tr, 0.11))
    assert not c.passed
    assert "worst: CSV line 9" in c.detail           # trade i=7 -> line 9
    assert "+45 min" in c.detail


def test_entry_inside_the_signal_bar_fails(offline_klines):
    """Signal at 08:00 (bar open), entry at 08:30: books the signal bar."""
    tr, _ = load_trades(log(entry=lambda i, b: b + 30 * M,
                            signal=lambda i, e: e - 30 * M))
    c = c6(audit(tr, 0.11))
    assert not c.passed and c.key_number == "120/120 trades"
    assert "inside the signal's own 1h bar" in c.detail


def test_only_the_offending_rows_are_counted_and_named(offline_klines):
    bad = {3, 15, 38}                                       # lines 5, 17, 40

    def sig(i, e):
        return e if i in bad else e - H
    tr, _ = load_trades(log(signal=sig))
    c = c6(audit(tr, 0.11))
    assert c.key_number == "3/120 trades"
    assert "CSV lines 5, 17, 40" in c.detail


def test_more_than_five_offenders_are_truncated_with_a_count(offline_klines):
    tr, _ = load_trades(log(signal=lambda i, e: e if i < 9 else e - H))
    c = c6(audit(tr, 0.11))
    assert c.key_number == "9/120 trades"
    assert "+4 more" in c.detail


def test_c6_alone_turns_a_passing_log_into_fail(offline_klines):
    tr, _ = load_trades(log(signal=lambda i, e: e + M))
    res = audit(tr, 0.11)
    assert all(c.passed for c in res.checks[:5])
    assert res.verdict == "FAIL"
    assert verdict_line(res).startswith(
        "FAIL — 1 of 6 checks failed (C6 look-ahead: 120/120 trades)")
    assert what_this_means(res).startswith("Entries are logged at or before")


# ── ingest aliases for the decision time ─────────────────────────────────────

@pytest.mark.parametrize("alias", ["decision_time", "signal_ts", "sig_time",
                                   "signal_at", "decided_at"])
def test_decision_time_aliases_are_recognised(alias):
    tr, _ = load_trades(log(n=3, header=alias))
    assert tr["signal_time"].notna().all()
    assert (tr["signal_time"] < tr["entry_time"]).all()


def test_a_bare_signal_column_is_not_taken_for_a_timestamp():
    """'signal' usually holds +1/-1; it must not become signal_time."""
    raw = log(n=3, with_signal=False).decode().splitlines()
    raw[0] = "signal," + raw[0]
    raw[1:] = ["1," + r for r in raw[1:]]
    tr, _ = load_trades(("\n".join(raw) + "\n").encode())
    assert tr["signal_time"].isna().all()
