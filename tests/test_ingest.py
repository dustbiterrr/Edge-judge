"""ingest: epoch timestamps in seconds / milliseconds parse to the same UTC
instants as ISO strings, and numbers outside every epoch range are rejected
with a human message (never silent 1970 dates)."""

from __future__ import annotations

import pandas as pd
import pytest

from judge.ingest import IngestError, load_trades

E1 = pd.Timestamp("2026-01-10 00:00", tz="UTC")
X1 = pd.Timestamp("2026-01-10 03:00", tz="UTC")
E2 = pd.Timestamp("2026-01-11 18:00", tz="UTC")
X2 = pd.Timestamp("2026-01-11 21:30", tz="UTC")
HEAD = "symbol,side,entry_time,exit_time,entry_price,exit_price"


def _csv(*rows: str) -> bytes:
    return (HEAD + "\n" + "\n".join(rows) + "\n").encode()


def _log(fmt) -> bytes:
    return _csv(f"ETHUSDT,long,{fmt(E1)},{fmt(X1)},100,101",
                f"ETHUSDT,short,{fmt(E2)},{fmt(X2)},100,99")


def test_epoch_seconds_parse_to_utc():
    tr, warns = load_trades(_log(lambda t: int(t.timestamp())))
    assert list(tr["entry_time"]) == [E1, E2]
    assert list(tr["exit_time"]) == [X1, X2]
    assert any("epoch s" in w for w in warns)


def test_epoch_milliseconds_parse_to_utc():
    tr, warns = load_trades(_log(lambda t: int(t.timestamp() * 1000)))
    assert list(tr["entry_time"]) == [E1, E2]
    assert list(tr["exit_time"]) == [X1, X2]
    assert any("epoch ms" in w for w in warns)


def test_seconds_milliseconds_and_iso_agree():
    iso, _ = load_trades(_log(lambda t: f"{t:%Y-%m-%d %H:%M}"))
    sec, _ = load_trades(_log(lambda t: int(t.timestamp())))
    ms, _ = load_trades(_log(lambda t: int(t.timestamp() * 1000)))
    for col in ("entry_time", "exit_time"):
        assert list(sec[col]) == list(iso[col])
        assert list(ms[col]) == list(iso[col])


def test_numeric_outside_epoch_ranges_is_rejected():
    # 5e8 is below the seconds range: pandas would read it as nanoseconds
    # and hand back 1970 - the judge refuses instead.
    with pytest.raises(IngestError, match="epoch"):
        load_trades(_csv("ETHUSDT,long,500000000,500003600,100,101"))


def test_mixed_units_in_one_column_do_not_pass_silently():
    # median magnitude decides the unit; the row in the other unit then
    # fails validation (exit before entry) rather than yielding a bogus date
    e_s, x_s = int(E1.timestamp()), int(X1.timestamp())
    e_ms, x_ms = int(E2.timestamp() * 1000), int(X2.timestamp() * 1000)
    with pytest.raises(IngestError):
        load_trades(_csv(f"ETHUSDT,long,{e_s},{x_s},100,101",
                         f"ETHUSDT,long,{e_ms},{x_s},100,101",
                         f"ETHUSDT,long,{e_ms},{x_ms},100,101"))
