#!/usr/bin/env python3
"""
fetch_binance_native.py — download NATIVE 15m/1h Binance USDT-M futures data
from data.binance.vision archive dumps (ZIP/CSV), NOT REST for history.

Sources (all archive unless noted):
  klines {15m,1h}        monthly + daily fallback for the trailing month
                         -> OHLCV, n_trades, taker_buy_volume
                         CVD per bar = taker_buy - (volume - taker_buy)
                                     = 2*taker_buy_volume - volume
                         (exact same quantity as summing signed aggTrades:
                          isBuyerMaker=false => aggressor BUY => +volume)
  premiumIndexKlines     monthly + daily fallback (close = premium at bar end)
  fundingRate            monthly dumps; trailing partial month via REST
                         /fapi/v1/fundingRate (marked in meta)
  metrics (daily dumps)  5m snapshots of sum_open_interest -> OI per bar
                         (last snapshot <= bar close, then ffill)
                         REST /futures/data/openInterestHist is only ~30d ->
                         used ONLY as fallback, marked "partial".

Output: data/native/{SYMBOL}/{tf}/bars.parquet with columns
  open_time(ms), open, high, low, close, volume, cvd_bar, n_trades,
  taker_buy_vol, oi, funding, premium
plus data/native/{SYMBOL}/meta.json coverage report.

Gaps: missing bars are left MISSING (no price interpolation); the probe must
gate forward windows on contiguity.

Usage:
    python scripts/fetch_binance_native.py --symbol ETHUSDT --months 6
    python scripts/fetch_binance_native.py --verify-cvd   # 1-day aggTrades cross-check
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

VISION = "https://data.binance.vision/data/futures/um"
# TODO(packaging): http_get / parse_klines / fetch_kline_like duplicate the
# 1h-kline downloader in judge/marketdata.py (_get / _parse_klines /
# _fetch_month).  Not refactored in the packaging wave - this script is frozen
# campaign tooling; the judge must not import from scripts/.
FAPI = "https://fapi.binance.com"
TF_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000}
UA = {"User-Agent": "edge-judge/1.0"}


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers (archive ZIPs, threaded, disk-cached)
# ─────────────────────────────────────────────────────────────────────────────

def http_get(url: str, timeout: int = 60, retries: int = 2) -> bytes | None:
    """GET url -> bytes; None on 404; retry on transient errors."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries:
                print(f"    [http {e.code}] {url}")
                return None
        except Exception as e:
            if attempt == retries:
                print(f"    [err {type(e).__name__}] {url}")
                return None
        time.sleep(1.0 + attempt)
    return None


def _dump_is_recent(url: str, days: int = 8) -> bool:
    """True if the URL's dated dump is recent enough that a past 404 may have
    since been published (dumps lag 1-3 days).  Daily YYYY-MM-DD -> within
    `days`; monthly YYYY-MM -> current or previous month."""
    now = datetime.now(timezone.utc)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", url)
    if m:
        d = datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
        return (now - d).days <= days
    m = re.search(r"(\d{4})-(\d{2})(?!\d)", url)
    if m:
        ym, cur = int(m[1]) * 12 + int(m[2]), now.year * 12 + now.month
        return 0 <= cur - ym <= 1
    return False


def fetch_cached(url: str, cache_dir: Path) -> bytes | None:
    """Download with a local file cache keyed by URL basename.  A 404 miss is
    cached to avoid re-hitting permanently-absent dumps — EXCEPT for recent
    dates, whose dumps publish with a lag: a stale miss there is refreshed so
    incremental fetch backfills days that appeared after the first attempt."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / url.rsplit("/", 1)[-1]
    miss_marker = fp.with_suffix(fp.suffix + ".404")
    if fp.is_file():
        return fp.read_bytes()
    if miss_marker.is_file():
        if _dump_is_recent(url):
            miss_marker.unlink()          # recent day may have published since
        else:
            return None
    data = http_get(url)
    if data is None:
        miss_marker.write_bytes(b"")
        return None
    fp.write_bytes(data)
    return data


def fetch_many(urls: list[str], cache_dir: Path, label: str) -> dict[str, bytes | None]:
    out: dict[str, bytes | None] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_cached, u, cache_dir): u for u in urls}
        for fut, u in futs.items():
            out[u] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(urls):
                ok = sum(1 for v in out.values() if v is not None)
                print(f"    [{label}] {done}/{len(urls)} fetched ({ok} present)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CSV parsing (dumps are headerless OR carry a header row — detect)
# ─────────────────────────────────────────────────────────────────────────────

def read_zip_csv(blob: bytes) -> pd.DataFrame | None:
    """Read the single CSV inside a dump ZIP, header auto-detected, as str."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = zf.namelist()[0]
            df = pd.read_csv(zf.open(name), header=None, dtype=str)
    except Exception as e:
        print(f"    [zip err] {type(e).__name__}")
        return None
    if df.empty:
        return None
    first = str(df.iloc[0, 0])
    if not first.replace(".", "", 1).replace("-", "", 1).isdigit():
        df = df.iloc[1:].reset_index(drop=True)   # drop header row
    return df


def ts_to_ms(v: np.ndarray) -> np.ndarray:
    """Binance switched some dumps to microseconds — normalize to ms."""
    v = v.astype(np.int64)
    return np.where(v > 10_000_000_000_000_0, v // 1000, v)  # >1e14 -> us


def parse_klines(blob: bytes) -> pd.DataFrame | None:
    """Futures kline CSV: 0 open_time 1 open 2 high 3 low 4 close 5 volume
    6 close_time 7 quote_vol 8 count 9 taker_buy_vol 10 taker_buy_quote 11 ign."""
    df = read_zip_csv(blob)
    if df is None or df.shape[1] < 10:
        return None
    out = pd.DataFrame({
        "open_time": ts_to_ms(df[0].astype(np.float64).values.astype(np.int64)),
        "open": df[1].astype(np.float64),
        "high": df[2].astype(np.float64),
        "low": df[3].astype(np.float64),
        "close": df[4].astype(np.float64),
        "volume": df[5].astype(np.float64),
        "n_trades": df[8].astype(np.float64),
        "taker_buy_vol": df[9].astype(np.float64),
    })
    return out


def parse_premium(blob: bytes) -> pd.DataFrame | None:
    """premiumIndexKlines CSV: kline format; close (col 4) = premium at bar end."""
    df = read_zip_csv(blob)
    if df is None or df.shape[1] < 5:
        return None
    return pd.DataFrame({
        "open_time": ts_to_ms(df[0].astype(np.float64).values.astype(np.int64)),
        "premium": df[4].astype(np.float64),
    })


def parse_funding(blob: bytes) -> pd.DataFrame | None:
    """fundingRate CSV: calc_time, funding_interval_hours, last_funding_rate."""
    df = read_zip_csv(blob)
    if df is None or df.shape[1] < 3:
        return None
    return pd.DataFrame({
        "time": ts_to_ms(df[0].astype(np.float64).values.astype(np.int64)),
        "funding": df[2].astype(np.float64),
    })


def parse_metrics(blob: bytes) -> pd.DataFrame | None:
    """metrics CSV (has header): create_time, symbol, sum_open_interest, ..."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            df = pd.read_csv(zf.open(zf.namelist()[0]))
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = cols.get("create_time")
    ocol = cols.get("sum_open_interest")
    if tcol is None or ocol is None:
        return None
    t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    # unit-proof ms conversion (pandas 3.0 may parse strings at us resolution)
    t_ms = (t - pd.Timestamp(0, tz="utc")) // pd.Timedelta(milliseconds=1)
    out = pd.DataFrame({
        "time": t_ms.astype("int64"),
        "oi": pd.to_numeric(df[ocol], errors="coerce"),
    }).dropna()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Date planning
# ─────────────────────────────────────────────────────────────────────────────

def month_list(months_back: int, now: datetime) -> list[str]:
    """['2026-01', ..., current month] covering months_back full months."""
    y, m = now.year, now.month
    res = []
    for k in range(months_back, -1, -1):
        yy, mm = y, m - k
        while mm <= 0:
            mm += 12
            yy -= 1
        res.append(f"{yy:04d}-{mm:02d}")
    return res


def days_of_month(ym: str, now: datetime) -> list[str]:
    """All dump-published days of month ym (up to yesterday UTC)."""
    y, m = map(int, ym.split("-"))
    d = datetime(y, m, 1, tzinfo=timezone.utc)
    last_ok = now - timedelta(days=1)
    days = []
    while d.month == m and d <= last_ok:
        days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days


# ─────────────────────────────────────────────────────────────────────────────
# Per-source fetch: monthly first, daily fallback for missing months
# ─────────────────────────────────────────────────────────────────────────────

def fetch_kline_like(kind: str, symbol: str, tf: str | None, months: list[str],
                     cache: Path, parser, now: datetime, label: str
                     ) -> tuple[pd.DataFrame | None, list[str]]:
    """kind in {'klines','premiumIndexKlines'}; returns (df, notes)."""
    sub = f"{symbol}/{tf}" if tf else symbol
    stem = f"{symbol}-{tf}" if tf else symbol
    # cache subdir per source kind: klines and premiumIndexKlines share
    # identical ZIP basenames — a flat cache would collide them
    cache = cache / f"{kind}-{tf or 'na'}"
    monthly_urls = [f"{VISION}/monthly/{kind}/{sub}/{stem}-{ym}.zip" for ym in months]
    got = fetch_many(monthly_urls, cache, f"{label} monthly")
    frames, notes = [], []
    for ym, url in zip(months, monthly_urls):
        blob = got[url]
        if blob is not None:
            df = parser(blob)
            if df is not None:
                frames.append(df)
                continue
        # daily fallback for this month (trailing month is never in monthly)
        days = days_of_month(ym, now)
        if not days:
            continue
        daily_urls = [f"{VISION}/daily/{kind}/{sub}/{stem}-{d}.zip" for d in days]
        dgot = fetch_many(daily_urls, cache, f"{label} daily {ym}")
        dfr = [parser(b) for b in dgot.values() if b is not None]
        dfr = [d for d in dfr if d is not None]
        if dfr:
            frames.append(pd.concat(dfr, ignore_index=True))
            notes.append(f"{label}: {ym} via daily dumps ({len(dfr)}/{len(days)} days)")
        else:
            notes.append(f"{label}: {ym} MISSING entirely")
    if not frames:
        return None, notes + [f"{label}: NO DATA"]
    df = (pd.concat(frames, ignore_index=True)
          .drop_duplicates("open_time").sort_values("open_time")
          .reset_index(drop=True))
    return df, notes


def fetch_funding(symbol: str, months: list[str], cache: Path, now: datetime
                  ) -> tuple[pd.DataFrame | None, list[str]]:
    urls = [f"{VISION}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
            for ym in months]
    got = fetch_many(urls, cache / "fundingRate", "funding monthly")
    frames, notes = [], []
    missing_months = []
    for ym, url in zip(months, urls):
        blob = got[url]
        df = parse_funding(blob) if blob is not None else None
        if df is not None:
            frames.append(df)
        else:
            missing_months.append(ym)
    # trailing / missing months via REST (funding has no daily dumps);
    # paginated: one 1000-event page covers ~333 days, longer holes need more
    if missing_months:
        start = datetime.strptime(min(missing_months) + "-01", "%Y-%m-%d")\
            .replace(tzinfo=timezone.utc)
        start_ms = int(start.timestamp() * 1000)
        events: list[dict] = []
        failed = False
        for _ in range(40):                      # hard cap: 40k events (~36y)
            url = (f"{FAPI}/fapi/v1/fundingRate?symbol={symbol}"
                   f"&startTime={start_ms}&limit=1000")
            blob = http_get(url)
            if blob is None:
                failed = not events
                break
            js = json.loads(blob)
            if not js:
                break
            events.extend(js)
            if len(js) < 1000:
                break
            start_ms = int(js[-1]["fundingTime"]) + 1
        if events:
            frames.append(pd.DataFrame({
                "time": [int(x["fundingTime"]) for x in events],
                "funding": [float(x["fundingRate"]) for x in events],
            }))
            notes.append(f"funding: months {missing_months} filled via REST "
                         f"fapi/v1/fundingRate ({len(events)} events)")
        if failed:
            notes.append(f"funding: months {missing_months} MISSING (REST failed)")
    if not frames:
        return None, notes + ["funding: NO DATA"]
    df = (pd.concat(frames, ignore_index=True)
          .drop_duplicates("time").sort_values("time").reset_index(drop=True))
    return df, notes


def fetch_oi(symbol: str, months: list[str], cache: Path, now: datetime
             ) -> tuple[pd.DataFrame | None, list[str]]:
    """OI from daily metrics dumps (5m snapshots, full history)."""
    days: list[str] = []
    for ym in months:
        days.extend(days_of_month(ym, now))
    urls = [f"{VISION}/daily/metrics/{symbol}/{symbol}-metrics-{d}.zip" for d in days]
    got = fetch_many(urls, cache / "metrics", "metrics(OI)")
    frames = [parse_metrics(b) for b in got.values() if b is not None]
    frames = [f for f in frames if f is not None]
    notes = []
    n_missing = sum(1 for v in got.values() if v is None)
    if n_missing:
        notes.append(f"metrics(OI): {n_missing}/{len(urls)} daily dumps missing")
    if not frames:
        # fallback: REST openInterestHist — only ~30 days deep, mark partial
        notes.append("metrics(OI): dumps unavailable -> trying REST "
                     "openInterestHist (PARTIAL ~30d)")
        rows = []
        end = int(now.timestamp() * 1000)
        for _ in range(80):
            url = (f"{FAPI}/futures/data/openInterestHist?symbol={symbol}"
                   f"&period=15m&limit=500&endTime={end}")
            blob = http_get(url)
            if not blob:
                break
            js = json.loads(blob)
            if not js:
                break
            rows.extend(js)
            end = int(js[0]["timestamp"]) - 1
            if len(js) < 500:
                break
        if rows:
            df = pd.DataFrame({
                "time": [int(x["timestamp"]) for x in rows],
                "oi": [float(x["sumOpenInterest"]) for x in rows],
            })
            return (df.drop_duplicates("time").sort_values("time")
                    .reset_index(drop=True)), notes
        return None, notes + ["OI: NO DATA"]
    df = (pd.concat(frames, ignore_index=True)
          .drop_duplicates("time").sort_values("time").reset_index(drop=True))
    return df, notes


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_tf(kl: pd.DataFrame, prem: pd.DataFrame | None,
                fund: pd.DataFrame | None, oi: pd.DataFrame | None,
                tf: str) -> pd.DataFrame:
    tf_ms = TF_MS[tf]
    df = kl.copy()
    # CVD per bar from taker aggressor volume (exact aggTrades identity)
    df["cvd_bar"] = 2.0 * df["taker_buy_vol"] - df["volume"]
    df["close_time"] = df["open_time"] + tf_ms - 1

    if prem is not None:
        df = df.merge(prem, on="open_time", how="left")
    else:
        df["premium"] = np.nan

    # funding: last PAST event at/before bar close — ffill only, no interpolation
    if fund is not None:
        df = pd.merge_asof(df.sort_values("close_time"),
                           fund.rename(columns={"time": "close_time"})
                               .sort_values("close_time"),
                           on="close_time", direction="backward")
    else:
        df["funding"] = np.nan

    # OI: last 5m snapshot at/before bar close (backward asof == ffill of past)
    if oi is not None:
        df = pd.merge_asof(df.sort_values("close_time"),
                           oi.rename(columns={"time": "close_time"})
                             .sort_values("close_time"),
                           on="close_time", direction="backward")
    else:
        df["oi"] = np.nan

    df = df.sort_values("open_time").reset_index(drop=True)
    cols = ["open_time", "open", "high", "low", "close", "volume", "cvd_bar",
            "n_trades", "taker_buy_vol", "oi", "funding", "premium"]
    return df[cols]


def coverage(df: pd.DataFrame, tf: str) -> dict:
    tf_ms = TF_MS[tf]
    t = df["open_time"].values
    expected = int((t[-1] - t[0]) // tf_ms) + 1
    gaps = int(np.sum(np.diff(t) != tf_ms))
    return {
        "bars": int(len(df)),
        "expected": expected,
        "missing_bars": int(expected - len(df)),
        "gap_points": gaps,
        "t0_utc": datetime.fromtimestamp(t[0] / 1000, timezone.utc).isoformat(),
        "t1_utc": datetime.fromtimestamp(t[-1] / 1000, timezone.utc).isoformat(),
        "oi_nan_frac": float(df["oi"].isna().mean()),
        "funding_nan_frac": float(df["funding"].isna().mean()),
        "premium_nan_frac": float(df["premium"].isna().mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional: 1-day aggTrades cross-check of klines-derived CVD
# ─────────────────────────────────────────────────────────────────────────────

def verify_cvd(symbol: str, cache: Path, now: datetime) -> None:
    day = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    print(f"\n[verify-cvd] cross-checking klines CVD vs aggTrades for {day}")
    kb = fetch_cached(f"{VISION}/daily/klines/{symbol}/15m/{symbol}-15m-{day}.zip",
                      cache / "klines-15m")
    ab = fetch_cached(f"{VISION}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip",
                      cache / "aggTrades")
    if kb is None or ab is None:
        print("  [skip] one of the sources unavailable")
        return
    kl = parse_klines(kb)
    with zipfile.ZipFile(io.BytesIO(ab)) as zf:
        at = pd.read_csv(zf.open(zf.namelist()[0]), header=None, dtype=str)
    if not str(at.iloc[0, 0]).isdigit():
        at = at.iloc[1:].reset_index(drop=True)
    # aggTrades cols: 0 aggId 1 price 2 qty 3 firstId 4 lastId 5 ts 6 isBuyerMaker
    ts = ts_to_ms(at[5].astype(np.float64).values.astype(np.int64))
    qty = at[2].astype(np.float64).values
    maker = at[6].astype(str).str.strip().str.lower().isin(["true", "1"]).values
    signed = np.where(maker, -qty, qty)     # m=false -> aggressor buy -> +vol
    bar = (ts // TF_MS["15m"]) * TF_MS["15m"]
    agg = pd.DataFrame({"open_time": bar, "cvd_at": signed}).groupby("open_time").sum()
    m = kl.set_index("open_time").join(agg, how="inner")
    m["cvd_kl"] = 2.0 * m["taker_buy_vol"] - m["volume"]
    diff = (m["cvd_kl"] - m["cvd_at"]).abs()
    rel = diff / m["volume"].clip(lower=1e-9)
    print(f"  bars compared: {len(m)}   max|diff|={diff.max():.6f} "
          f"(rel to volume: {rel.max():.2e})")
    # liquidation/ADL prints are absent from aggTrades but counted in klines —
    # sub-0.1%-of-volume discrepancy is expected and sign-irrelevant
    if rel.max() < 1e-3:
        print("  -> MATCH: klines-derived CVD == aggTrades CVD "
              "(within liquidation-print tolerance <0.1% of volume)")
    else:
        print("  -> MISMATCH above tolerance — investigate before trusting CVD")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def validate_bars(df: pd.DataFrame, cov: dict) -> tuple[bool, list[str]]:
    """Hard gates — a symbol failing any gate is EXCLUDED with a note, not fixed."""
    reasons: list[str] = []
    if cov["expected"] > 0 and cov["missing_bars"] / cov["expected"] > 0.02:
        reasons.append(f"gaps: {cov['missing_bars']}/{cov['expected']} bars missing")
    cvd_alive = float((df["cvd_bar"] != 0).mean())
    if cvd_alive < 0.5:
        reasons.append(f"cvd dead (nonzero frac {cvd_alive:.2f})")
    oi_nu = int(df["oi"].nunique())
    if oi_nu < 100 or df["oi"].isna().mean() > 0.2:
        reasons.append(f"oi not live (nunique={oi_nu}, "
                       f"nan={df['oi'].isna().mean():.2f})")
    f_nu = int(df["funding"].nunique())
    if f_nu < 5 or df["funding"].isna().mean() > 0.2:
        reasons.append(f"funding not live (nunique={f_nu})")
    p_nan = float(df["premium"].isna().mean())
    p_max = (float(np.nanmax(np.abs(df["premium"].values)))
             if p_nan < 1.0 else float("nan"))
    if p_nan > 0.2 or not (p_max < 0.05):
        reasons.append(f"premium invalid (nan={p_nan:.2f}, max|p|={p_max})")
    return len(reasons) == 0, reasons


def fetch_symbol(symbol: str, months: list[str], tfs: list[str],
                 data_dir: Path, now: datetime, verify: bool) -> dict:
    base = data_dir / symbol
    cache = base / "_cache"
    print(f"\n{'=' * 72}\n  {symbol}   months={months[0]}..{months[-1]}  "
          f"tfs={tfs}\n{'=' * 72}")

    all_notes: list[str] = []
    meta: dict = {"symbol": symbol, "fetched_utc": now.isoformat(),
                  "months": months, "sources": {}, "valid": {}}

    print("[1/4] funding (monthly dumps + REST tail)")
    fund, notes = fetch_funding(symbol, months, cache, now)
    all_notes += notes
    print("[2/4] open interest (daily metrics dumps, 5m snapshots)")
    oi, notes = fetch_oi(symbol, months, cache, now)
    all_notes += notes

    for tf in tfs:
        print(f"[3/4] klines {tf}")
        kl, notes = fetch_kline_like("klines", symbol, tf, months, cache,
                                     parse_klines, now, f"klines-{tf}")
        all_notes += notes
        if kl is None:
            print(f"  [fatal] no klines for {tf}")
            meta["valid"][tf] = {"ok": False, "reasons": ["no klines at all"]}
            continue
        print(f"[4/4] premiumIndexKlines {tf}")
        prem, notes = fetch_kline_like("premiumIndexKlines", symbol, tf, months,
                                       cache, parse_premium, now, f"premium-{tf}")
        all_notes += notes

        df = assemble_tf(kl, prem, fund, oi, tf)
        cov = coverage(df, tf)
        ok, reasons = validate_bars(df, cov)
        outdir = base / tf
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outdir / "bars.parquet", index=False)
        meta["sources"][tf] = cov
        meta["valid"][tf] = {"ok": ok, "reasons": reasons}
        status = "VALID" if ok else "EXCLUDED: " + "; ".join(reasons)
        print(f"  -> {outdir / 'bars.parquet'}  bars={cov['bars']} "
              f"(missing {cov['missing_bars']})  span {cov['t0_utc'][:10]}.."
              f"{cov['t1_utc'][:10]}  [{status}]")

    meta["notes"] = all_notes
    (base / "meta.json").write_text(json.dumps(meta, indent=2))
    for n in all_notes:
        print(f"  note: {n}")
    if verify:
        verify_cvd(symbol, cache, now)
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--symbols", default="",
                   help="comma list; overrides --symbol "
                        "(e.g. SOLUSDT,DOGEUSDT,AVAXUSDT)")
    p.add_argument("--months", type=int, default=6)
    p.add_argument("--data-dir", default="data/native")
    p.add_argument("--tfs", default="15m,1h")
    p.add_argument("--verify-cvd", action="store_true",
                   help="1-day aggTrades cross-check of klines-derived CVD")
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               or [args.symbol.upper()])
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    months = month_list(args.months, now)

    summary: list[tuple] = []
    for sym in symbols:
        try:
            meta = fetch_symbol(sym, months, tfs, Path(args.data_dir), now,
                                args.verify_cvd)
            v = meta.get("valid", {})
            ok = bool(tfs) and all(v.get(tf, {}).get("ok", False) for tf in tfs)
            summary.append((sym, ok, v))
        except Exception as e:
            print(f"  [fatal] {sym}: {type(e).__name__}: {e}")
            summary.append((sym, False, {"error": str(e)}))

    print(f"\n{'=' * 72}\n  BASKET SUMMARY\n{'=' * 72}")
    for sym, ok, v in summary:
        if ok:
            print(f"  VALID     {sym}")
        else:
            why = "; ".join(
                f"{tf}: {', '.join(d.get('reasons', ['?']))}"
                for tf, d in v.items()
                if isinstance(d, dict) and not d.get("ok", True)) or str(v)
            print(f"  EXCLUDED  {sym}  ({why})")
    n_ok = sum(1 for _, ok, _ in summary if ok)
    print(f"\n  {n_ok}/{len(summary)} symbols valid for the probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
