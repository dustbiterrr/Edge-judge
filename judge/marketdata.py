"""
marketdata.py — 1h klines for reconstruction and regime tagging.

Cache-first: data/native/{SYMBOL}/1h/bars.parquet (the campaign cache) is
used when it covers the requested span; only the missing months are fetched
from data.binance.vision (archive dumps, never REST for history) and stored
in data/native/{SYMBOL}/1h/klines_cache.parquet for next time.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VISION = "https://data.binance.vision/data/futures/um"
# TODO(packaging): _get / _parse_klines / _fetch_month duplicate the archive
# downloader in scripts/fetch_binance_native.py (http_get / parse_klines /
# fetch_kline_like).  Kept separate on purpose: the judge stays a small
# self-contained library and the probe script is frozen campaign tooling.
# Unify behind one module once the probes are versioned on their own.
TF_MS = 3_600_000
# anchored to the repo root, not the caller's CWD
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "native"
UA = {"User-Agent": "Mozilla/5.0 (edge-judge)"}
# Binance USDT-M futures have no data before this; also keeps
# datetime.fromtimestamp() away from pre-epoch values on Windows
ARCHIVE_T0_MS = 1_546_300_800_000          # 2019-01-01 UTC


class MarketDataError(Exception):
    pass


def _get(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise MarketDataError(f"Binance archive returned HTTP {e.code}")
    except Exception as e:
        raise MarketDataError(f"Network problem fetching market data: "
                              f"{type(e).__name__}")


def _parse_klines(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        df = pd.read_csv(zf.open(zf.namelist()[0]), header=None, dtype=str)
    if not str(df.iloc[0, 0]).replace(".", "").isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    t = df[0].astype(np.float64).values.astype(np.int64)
    t = np.where(t > 10 ** 14, t // 1000, t)          # us -> ms
    return pd.DataFrame({
        "open_time": t,
        "open": df[1].astype(np.float64),
        "high": df[2].astype(np.float64),
        "low": df[3].astype(np.float64),
        "close": df[4].astype(np.float64),
    })


def _months(t0: datetime, t1: datetime) -> list[str]:
    out, y, m = [], t0.year, t0.month
    while (y, m) <= (t1.year, t1.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _month_bounds_ms(ym: str) -> tuple[int, int]:
    """[start, next-month start) of a YYYY-MM month, in ms UTC."""
    y, m = map(int, ym.split("-"))
    a = datetime(y, m, 1, tzinfo=timezone.utc)
    b = datetime(y + 1, 1, 1, tzinfo=timezone.utc) if m == 12 else \
        datetime(y, m + 1, 1, tzinfo=timezone.utc)
    return int(a.timestamp() * 1000), int(b.timestamp() * 1000)


def _fetch_month(sym: str, ym: str, now: datetime,
                 progress=None) -> list[pd.DataFrame]:
    """Monthly dump, else daily dumps.  An empty result is NOT an error:
    the archive simply has nothing for that month (e.g. the symbol was not
    listed yet, or the month only exists as padding around the log)."""
    blob = _get(f"{VISION}/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip")
    if blob is not None:
        return [_parse_klines(blob)]
    out: list[pd.DataFrame] = []
    y, m = map(int, ym.split("-"))
    d = datetime(y, m, 1, tzinfo=timezone.utc)
    while d.month == m and d <= now - timedelta(days=1):
        b = _get(f"{VISION}/daily/klines/{sym}/1h/{sym}-1h-{d:%Y-%m-%d}.zip")
        if b is not None:
            out.append(_parse_klines(b))
        d += timedelta(days=1)
    return out


def ensure_klines(symbol: str, t0_ms: int, t1_ms: int,
                  progress=None) -> pd.DataFrame:
    """1h OHLC covering [t0_ms - 24h, t1_ms + 24h].

    Months inside the window that have no bar in the cache are (re)fetched —
    min/max coverage alone would leave internal holes unfilled.  A month the
    archive does not have (symbol listed later, padding before listing) is
    skipped, not fatal; MarketDataError is raised only when the WHOLE window
    comes back empty."""
    sym = symbol.upper()
    lo = max(t0_ms - 24 * TF_MS, ARCHIVE_T0_MS)
    hi = t1_ms + 24 * TF_MS
    frames: list[pd.DataFrame] = []

    campaign = DATA_DIR / sym / "1h" / "bars.parquet"
    cache_fp = DATA_DIR / sym / "1h" / "klines_cache.parquet"
    for fp in (campaign, cache_fp):
        if fp.is_file():
            d = pd.read_parquet(fp)[["open_time", "open", "high", "low",
                                     "close"]]
            frames.append(d)
    have = (pd.concat(frames).drop_duplicates("open_time")
            if frames else pd.DataFrame(columns=["open_time"]))

    now = datetime.now(timezone.utc)
    d0 = datetime.fromtimestamp(lo / 1000, timezone.utc)
    d1 = min(datetime.fromtimestamp(max(hi, lo) / 1000, timezone.utc), now)
    need: list[str] = []
    if d0 <= d1:
        ht = have["open_time"].values
        for ym in _months(d0, d1):
            m_lo, m_hi = _month_bounds_ms(ym)
            w0, w1 = max(lo, m_lo), min(hi, m_hi - TF_MS)
            if w0 > w1:
                continue
            if have.empty or not ((ht >= w0) & (ht <= w1)).any():
                need.append(ym)

    if need:
        fetched: list[pd.DataFrame] = []
        for i, ym in enumerate(need):
            if progress:
                progress(f"fetching {sym} {ym} ({i + 1}/{len(need)})")
            fetched += _fetch_month(sym, ym, now, progress)
        if fetched:
            new = pd.concat(fetched, ignore_index=True)
            all_ = (pd.concat([have, new]) if not have.empty else new)
            all_ = (all_.drop_duplicates("open_time")
                    .sort_values("open_time").reset_index(drop=True))
            cache_fp.parent.mkdir(parents=True, exist_ok=True)
            all_.to_parquet(cache_fp, index=False)
            have = all_

    out = have[(have["open_time"] >= lo) & (have["open_time"] <= hi)]
    if out.empty:
        raise MarketDataError(
            f"Binance archive has no 1h data for {sym} anywhere in the "
            f"log's window. Check the symbol — we support Binance USDT-perp "
            f"futures.")
    return out.sort_values("open_time").reset_index(drop=True)


def next_bar_open(kl: pd.DataFrame, ts_ms: int) -> tuple[float, bool]:
    """Execution price rule v1.0: open of the first 1h bar strictly after ts.
    Returns (price, gapped): gapped=True when that bar starts more than one
    bar-length after ts — a market-data hole; the price is then approximate
    and the caller must surface a warning instead of silently using it."""
    t = kl["open_time"].values
    i = np.searchsorted(t, ts_ms, side="right")
    if i >= len(t):
        return float("nan"), False
    return float(kl["open"].values[i]), bool(t[i] - ts_ms > TF_MS)


def drift_24h_at(kl: pd.DataFrame, ts_ms: int) -> float:
    """24h close-to-close drift ending at the last bar CLOSED at/before ts.

    v1.0.1: bar i is closed once open_time[i] + TF <= ts, so the anchor is
    searchsorted(ts - TF, side="right") - 1.  The previous "left - 1" form
    picked the last bar *opened* before ts, whose close can lie up to one
    bar after a mid-bar entry — a look-ahead.  Bar-aligned timestamps are
    unaffected (both forms give the same index there)."""
    t = kl["open_time"].values
    i = np.searchsorted(t, ts_ms - TF_MS, side="right") - 1
    if i < 24:
        return float("nan")
    c = kl["close"].values
    return float(c[i] / c[i - 24] - 1.0)
