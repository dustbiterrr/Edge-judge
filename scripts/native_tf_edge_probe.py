#!/usr/bin/env python3
"""
native_tf_edge_probe.py — model-free directional-edge probe on NATIVE Binance
15m/1h futures bars (fetched by fetch_binance_native.py). No RL, no ML.

METHODOLOGY (pre-registered):
  1. NO LOOK-AHEAD: signal at bar T uses only data closed by end of T.
     Entry at open(T+1); exit at open(T+1+H). Never close(T).
  2. WALK-FORWARD: history split in half by time. Any selection (combo
     members, "best" cells) is done on IN half only; verdict read on OUT.
  3. PRE-REGISTERED THRESHOLD: PASS requires, with n>=100 on BOTH halves:
        net expectancy OUT > +0.22%/trade   (= 2x taker round-trip)
        net expectancy IN  > 0              (same sign both halves)
     0 < net_OUT <= 0.22% -> "marginal, probably noise".
  4. Guard: any cell with n<100 on a half -> "insufficient".
  5. Funding/OI are ffilled PAST values only (done at fetch); price gaps are
     NOT interpolated — forward windows crossing a gap are dropped here.

Expectancy metric (A): mean( sign(signal_T) * log_ret(open_{T+1} -> open_{T+1+H}) )
minus taker round-trip 0.11%. Hit-rate (B): P(sign(fwd)==sign(signal)).
NOTE: entries are per-bar and overlap for H>1 — this measures per-signal-bar
expectancy (information), not a non-overlapping equity curve.

Usage:
    python scripts/native_tf_edge_probe.py --symbol ETHUSDT
    python scripts/native_tf_edge_probe.py --tfs 15m --horizons 1,4
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

TAKER_RT = 0.0011          # 0.055%/side * 2
PASS_NET = 0.0022          # pre-registered: net OUT expectancy > 2x taker RT
MIN_N = 100
TF_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000}


# ─────────────────────────────────────────────────────────────────────────────
# Forward returns with explicit gap gating
# ─────────────────────────────────────────────────────────────────────────────

def forward_returns(t_ms: np.ndarray, open_: np.ndarray, H: int,
                    tf_ms: int) -> np.ndarray:
    """fwd[T] = log(open[T+1+H]/open[T+1]); NaN if bars T..T+1+H not contiguous."""
    n = len(t_ms)
    fwd = np.full(n, np.nan)
    k = 1 + H
    if n <= k:
        return fwd
    lo = np.log(np.clip(open_, 1e-12, None))
    seg = lo[k:] - lo[1:n - k + 1]
    # contiguity: open_time must advance exactly (1+H)*tf between T and T+1+H
    contig = (t_ms[k:] - t_ms[:n - k]) == k * tf_ms
    fwd[:n - k] = np.where(contig, seg, np.nan)
    return fwd


# ─────────────────────────────────────────────────────────────────────────────
# Signal families — every input series ends at close of bar T (no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values
    cvd = df["cvd_bar"].values
    oi = df["oi"].values
    prem = df["premium"].values
    logc = np.log(np.clip(close, 1e-12, None))

    sig: dict[str, np.ndarray] = {}

    def ret_n(N: int) -> np.ndarray:
        r = np.full_like(logc, np.nan)
        r[N:] = logc[N:] - logc[:-N]
        return r

    def rollsum(x: np.ndarray, N: int) -> np.ndarray:
        s = pd.Series(x).rolling(N, min_periods=N).sum().values
        return s

    # S1 PRICE momentum
    for N in (4, 8, 24):
        sig[f"S1 ret_{N}"] = np.sign(np.nan_to_num(ret_n(N)))

    # S2 FLOW (cvd_bar is per-bar increment by construction) + inversions
    for N in (1, 4, 8):
        c = rollsum(cvd, N)
        s = np.sign(np.where(np.isnan(c), 0.0, c))
        sig[f"S2 +cvd_{N}"] = s
        sig[f"S2 -cvd_{N}"] = -s

    # S3 DIVERGENCE: cvd_4 against ret_4 — trade cvd sign only on disagreement
    c4 = np.sign(np.nan_to_num(rollsum(cvd, 4)))
    r4 = np.sign(np.nan_to_num(ret_n(4)))
    sig["S3 cvd4_vs_ret4"] = np.where((c4 != 0) & (r4 != 0) & (c4 != r4), c4, 0.0)

    # S4 MTF context: close vs rolling VWAP-24; breakout vs prior 96-bar range
    pv = pd.Series(close * vol).rolling(24, min_periods=24).sum().values
    vv = pd.Series(vol).rolling(24, min_periods=24).sum().values
    vwap24 = np.where(vv > 0, pv / np.where(vv > 0, vv, 1.0), np.nan)
    d = close - vwap24
    sig["S4 close-vwap24"] = np.sign(np.where(np.isnan(d), 0.0, d))
    # prior-96 extremes EXCLUDING current bar (close can't exceed own high)
    hh = pd.Series(high).shift(1).rolling(96, min_periods=96).max().values
    ll = pd.Series(low).shift(1).rolling(96, min_periods=96).min().values
    bo = np.zeros_like(close)
    bo[np.nan_to_num(close - hh, nan=-1) > 0] = 1.0
    bo[np.nan_to_num(ll - close, nan=-1) > 0] = -1.0
    sig["S4 breakout96"] = bo

    # S5 OI/FUNDING
    if not np.all(np.isnan(oi)):
        doi = np.full_like(close, np.nan)
        doi[4:] = oi[4:] - oi[:-4]
        r4v = ret_n(4)
        # rising OI confirms the price move (new positioning drives it);
        # falling OI = position closing -> no signal
        s = np.where((doi > 0) & ~np.isnan(r4v), np.sign(r4v), 0.0)
        sig["S5 dOI+ x ret4"] = np.nan_to_num(s)
    if not np.all(np.isnan(prem)):
        pm = pd.Series(prem)
        mu = pm.rolling(96, min_periods=96).mean()
        sd = pm.rolling(96, min_periods=96).std()
        z = ((pm - mu) / sd.replace(0.0, np.nan)).values
        # mean-revert: extreme positive premium -> short, extreme negative -> long
        sig["S5 prem_z>2 MR"] = np.where(np.abs(np.nan_to_num(z)) > 2.0,
                                         -np.sign(np.nan_to_num(z)), 0.0)
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def cell(sig: np.ndarray, fwd: np.ndarray, half: np.ndarray) -> dict:
    m = (sig != 0) & ~np.isnan(fwd) & half
    n = int(m.sum())
    if n < MIN_N:
        return {"n": n, "ok": False}
    d = sig[m]
    f = fwd[m]
    gross = float(np.mean(d * f))
    hit = float(np.mean(np.sign(f) == d))
    return {"n": n, "ok": True, "net": (gross - TAKER_RT) * 100.0,
            "gross": gross * 100.0, "hit": hit * 100.0}


def fmt_cell(c: dict) -> str:
    if not c["ok"]:
        return f"n={c['n']:<6d} insufficient        "
    return f"n={c['n']:<6d} net={c['net']:+7.3f}% hit={c['hit']:5.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--data-dir", default="data/native")
    p.add_argument("--tfs", default="15m,1h")
    p.add_argument("--horizons", default="1,4,8")
    args = p.parse_args()

    symbol = args.symbol.upper()
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    print("=" * 100)
    print(f"  NATIVE-TF DIRECTIONAL EDGE PROBE — {symbol}   "
          f"(entry open(T+1), exit open(T+1+H), taker RT {TAKER_RT*100:.2f}%)")
    print(f"  PASS: net_OUT > +{PASS_NET*100:.2f}%  AND  net_IN > 0 (same sign)  "
          f"AND  n>={MIN_N} both halves   |   0<net_OUT<=+0.22% = marginal")
    print("=" * 100)

    passed: list[tuple] = []
    marginal: list[tuple] = []
    results: dict[str, dict] = {}

    for tf in tfs:
        fp = Path(args.data_dir) / symbol / tf / "bars.parquet"
        if not fp.is_file():
            print(f"\n[{tf}] MISSING {fp} — run fetch_binance_native.py first")
            continue
        df = pd.read_parquet(fp)
        t_ms = df["open_time"].values.astype(np.int64)
        n = len(df)
        mid = n // 2
        half_in = np.zeros(n, bool); half_in[:mid] = True
        half_out = ~half_in
        t0 = datetime.fromtimestamp(t_ms[0] / 1000, timezone.utc)
        tm = datetime.fromtimestamp(t_ms[mid] / 1000, timezone.utc)
        t1 = datetime.fromtimestamp(t_ms[-1] / 1000, timezone.utc)
        print(f"\n{'-' * 100}")
        print(f"  TF {tf}: {n} bars   IN  {t0:%Y-%m-%d} .. {tm:%Y-%m-%d}   "
              f"OUT {tm:%Y-%m-%d} .. {t1:%Y-%m-%d}")
        print(f"{'-' * 100}")

        fwds = {H: forward_returns(t_ms, df["open"].values, H, TF_MS[tf])
                for H in horizons}
        sigs = build_signals(df)
        results[tf] = {"sigs": sigs, "fwds": fwds,
                       "half_in": half_in, "half_out": half_out}

        hdr = f"  {'signal':<18s} {'H':>2s}   {'IN half':<38s} {'OUT half':<38s}"
        print(hdr)
        for name, s in sigs.items():
            for H in horizons:
                ci = cell(s, fwds[H], half_in)
                co = cell(s, fwds[H], half_out)
                print(f"  {name:<18s} {H:>2d}   {fmt_cell(ci):<38s} {fmt_cell(co):<38s}")
                if ci["ok"] and co["ok"]:
                    if co["net"] > PASS_NET * 100 and ci["net"] > 0:
                        passed.append((tf, name, H, ci, co))
                    elif 0 < co["net"] <= PASS_NET * 100 and ci["net"] > 0:
                        marginal.append((tf, name, H, ci, co))

        # COMBO — members chosen on IN half ONLY (walk-forward clean):
        # best S1 cell and best S2 cell by IN net expectancy; trade agreement.
        def best_family(prefix: str) -> tuple | None:
            cand = []
            for name, s in sigs.items():
                if not name.startswith(prefix):
                    continue
                for H in horizons:
                    ci = cell(s, fwds[H], half_in)
                    if ci["ok"]:
                        cand.append((ci["net"], name, H))
            return max(cand) if cand else None

        b1, b2 = best_family("S1"), best_family("S2")
        if b1 and b2:
            _, n1, h1 = b1
            _, n2, h2 = b2
            s1, s2 = sigs[n1], sigs[n2]
            combo = np.where((s1 != 0) & (s1 == s2), s1, 0.0)
            print(f"\n  COMBO (IN-selected: [{n1}] & [{n2}], agreement only):")
            for H in horizons:
                ci = cell(combo, fwds[H], half_in)
                co = cell(combo, fwds[H], half_out)
                print(f"  {'S1&S2 agree':<18s} {H:>2d}   {fmt_cell(ci):<38s} "
                      f"{fmt_cell(co):<38s}")
                if ci["ok"] and co["ok"]:
                    if co["net"] > PASS_NET * 100 and ci["net"] > 0:
                        passed.append((tf, "COMBO S1&S2", H, ci, co))
                    elif 0 < co["net"] <= PASS_NET * 100 and ci["net"] > 0:
                        marginal.append((tf, "COMBO S1&S2", H, ci, co))

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  VERDICT (pre-registered threshold, OUT half only)")
    print("=" * 100)
    if passed:
        passed.sort(key=lambda x: -x[4]["net"])
        print(f"  PASSED ({len(passed)} cells):")
        for tf, name, H, ci, co in passed[:3]:
            print(f"    {tf} [{name}] H={H}: OUT net {co['net']:+.3f}% "
                  f"hit {co['hit']:.1f}% n={co['n']}  |  IN net {ci['net']:+.3f}% "
                  f"n={ci['n']}")
    else:
        print("  PASSED: none.")
    if marginal:
        marginal.sort(key=lambda x: -x[4]["net"])
        print(f"  MARGINAL (0 < net_OUT <= +0.22%, probably noise): "
              f"{len(marginal)} cells; top 3:")
        for tf, name, H, ci, co in marginal[:3]:
            print(f"    {tf} [{name}] H={H}: OUT net {co['net']:+.3f}% "
                  f"hit {co['hit']:.1f}% n={co['n']}  |  IN net {ci['net']:+.3f}%")
    if not passed:
        print("\n  -> directional edge on 15m/1h NOT FOUND across all 5 signal")
        print("     families under the pre-registered threshold — closed honestly.")
    else:
        print("\n  -> candidate edge(s) above; verify stability before ANY RL:")
        print("     re-run on a different symbol and a shifted half-split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
