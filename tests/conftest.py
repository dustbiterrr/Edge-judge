"""Shared fixtures.  Every test here runs OFFLINE: network access is disabled
at the socket level, so anything that tries to reach data.binance.vision
fails loudly instead of silently downloading."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TF_MS = 3_600_000


class NetworkDisabled(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def blocked(*_a, **_k):
        raise NetworkDisabled("network access is disabled in tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)
    yield


def make_klines(t0_ms: int, n_bars: int, price: float = 100.0) -> pd.DataFrame:
    """Flat synthetic 1h OHLC series starting at t0_ms (24h drift = 0)."""
    t = t0_ms + TF_MS * np.arange(n_bars, dtype=np.int64)
    c = np.full(n_bars, price, dtype=float)
    return pd.DataFrame({"open_time": t, "open": c, "high": c, "low": c,
                         "close": c})


@pytest.fixture
def offline_klines(monkeypatch):
    """judge.checks.ensure_klines -> flat synthetic series covering
    [t0 - 48h, t1 + 48h].  No network, every trade lands in regime FLAT."""
    def fake(symbol, t0_ms, t1_ms, progress=None):
        n = (t1_ms - t0_ms) // TF_MS + 97
        return make_klines(t0_ms - 48 * TF_MS, int(n))
    monkeypatch.setattr("judge.checks.ensure_klines", fake)
    return fake


def trade_log(n: int = 120, entry: float = 100.0, exit_: float = 100.5,
              side: str = "long", with_prices: bool = True) -> bytes:
    """n non-overlapping ETHUSDT trades, one hour long, two hours apart,
    starting 2026-01-05 00:00 UTC."""
    t = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    head = "symbol,side,entry_time,exit_time"
    if with_prices:
        head += ",entry_price,exit_price,qty"
    rows = [head]
    for i in range(n):
        e = t + pd.Timedelta(hours=2 * i)
        x = e + pd.Timedelta(hours=1)
        row = f"ETHUSDT,{side},{e:%Y-%m-%d %H:%M},{x:%Y-%m-%d %H:%M}"
        if with_prices:
            row += f",{entry},{exit_},1"
        rows.append(row)
    return ("\n".join(rows) + "\n").encode()
