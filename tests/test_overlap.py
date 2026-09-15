"""resolve_overlaps: the behaviour must equal its docstring - per symbol, in
entry_time order, an entry made while a previous trade is still open is
dropped; on tied entries the first input row wins."""

from __future__ import annotations

import pandas as pd

from judge.overlap import resolve_overlaps


def _trades(rows):
    df = pd.DataFrame(rows, columns=["symbol", "entry_time", "exit_time"])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["side"] = 1.0
    return df


def _hhmm(s: pd.Series) -> list[str]:
    return list(s.dt.strftime("%H:%M"))


def test_entry_while_open_is_dropped_entry_after_exit_is_kept():
    tr = _trades([("ETHUSDT", "2026-01-01 00:00", "2026-01-01 05:00"),
                  ("ETHUSDT", "2026-01-01 02:00", "2026-01-01 03:00"),
                  ("ETHUSDT", "2026-01-01 06:00", "2026-01-01 07:00")])
    kept, dropped = resolve_overlaps(tr)
    assert dropped == 1
    assert _hhmm(kept["entry_time"]) == ["00:00", "06:00"]


def test_entry_at_the_exact_exit_time_is_not_an_overlap():
    tr = _trades([("ETHUSDT", "2026-01-01 00:00", "2026-01-01 05:00"),
                  ("ETHUSDT", "2026-01-01 05:00", "2026-01-01 06:00")])
    kept, dropped = resolve_overlaps(tr)
    assert dropped == 0
    assert len(kept) == 2


def test_a_dropped_trade_does_not_extend_the_open_window():
    # 00:00-02:00 open; 01:00-10:00 dropped; 03:00 is after 02:00 -> kept
    tr = _trades([("ETHUSDT", "2026-01-01 00:00", "2026-01-01 02:00"),
                  ("ETHUSDT", "2026-01-01 01:00", "2026-01-01 10:00"),
                  ("ETHUSDT", "2026-01-01 03:00", "2026-01-01 04:00")])
    kept, dropped = resolve_overlaps(tr)
    assert dropped == 1
    assert _hhmm(kept["entry_time"]) == ["00:00", "03:00"]


def test_symbols_are_resolved_independently():
    tr = _trades([("ETHUSDT", "2026-01-01 00:00", "2026-01-01 05:00"),
                  ("SOLUSDT", "2026-01-01 01:00", "2026-01-01 02:00")])
    kept, dropped = resolve_overlaps(tr)
    assert dropped == 0
    assert len(kept) == 2


def test_tied_entries_first_input_row_wins_not_the_shortest():
    long_first = _trades([("ETHUSDT", "2026-01-01 00:00", "2026-01-01 10:00"),
                          ("ETHUSDT", "2026-01-01 00:00", "2026-01-01 01:00")])
    kept, dropped = resolve_overlaps(long_first)
    assert dropped == 1 and _hhmm(kept["exit_time"]) == ["10:00"]

    short_first = _trades([("ETHUSDT", "2026-01-01 00:00", "2026-01-01 01:00"),
                           ("ETHUSDT", "2026-01-01 00:00", "2026-01-01 10:00")])
    kept, dropped = resolve_overlaps(short_first)
    assert dropped == 1 and _hhmm(kept["exit_time"]) == ["01:00"]


def test_unsorted_input_is_taken_in_entry_time_order():
    tr = _trades([("ETHUSDT", "2026-01-01 06:00", "2026-01-01 07:00"),
                  ("ETHUSDT", "2026-01-01 02:00", "2026-01-01 03:00"),
                  ("ETHUSDT", "2026-01-01 00:00", "2026-01-01 05:00")])
    kept, dropped = resolve_overlaps(tr)
    assert dropped == 1
    assert sorted(_hhmm(kept["entry_time"])) == ["00:00", "06:00"]
    assert list(kept.index) == [0, 1]           # index reset, no gaps
