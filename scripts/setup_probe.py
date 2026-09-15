#!/usr/bin/env python3
"""
setup_probe.py — walk-forward judge for the 13 composite scalper setups
(setup_library.py) over a multi-symbol basket of native 15m/1h bars.

REUSES the judging core of native_tf_edge_probe.py (forward_returns, cell,
thresholds) — same executable definition of a trade as every previous probe:
entry open(T+1), exit open(T+1+H), taker RT fee, gap-gated windows.

Pre-registered rules (identical to the generic probe, no post-hoc softening):
  PASS      net_OUT > +0.22%/trade AND net_IN > 0 (same sign) AND n>=100
            on both halves
  MARGINAL  net_IN > 0 and 0 < net_OUT <= +0.22%
  else dead / insufficient (n<100 on a half)

Multiple-comparisons control: for every evaluable cell we estimate
P(false pass | zero edge) with a normal approximation using the cell's own
per-trade sigma:  p = P(gross_OUT > 0.33%) * P(gross_IN > 0.11%),
and print  E[false passes] = sum(p)  next to the actual pass count.
CAVEAT (printed): entries overlap for H>1, so effective n is lower than
counted and E[false] is an UNDERestimate — read passes conservatively.

Usage:
    python scripts/setup_probe.py
    python scripts/setup_probe.py --symbols ETHUSDT,SOLUSDT --tfs 15m
"""

from __future__ import annotations

import argparse
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_tf_edge_probe import (MIN_N, PASS_NET, TAKER_RT, TF_MS,  # noqa: E402
                                  cell, forward_returns)
from setup_library import SETUPS  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

DEFAULT_SYMBOLS = ("ETHUSDT,SOLUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,NEARUSDT,"
                   "APTUSDT,ARBUSDT,OPUSDT,SUIUSDT,ADAUSDT,XRPUSDT,DOTUSDT")
GROSS_PASS = PASS_NET + TAKER_RT      # net_OUT>0.22% <=> gross_OUT>0.33%


def phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def quick_validate(df: pd.DataFrame, tf: str) -> list[str]:
    """Independent re-check of the fetch-time gates (belt and suspenders)."""
    t = df["open_time"].values.astype(np.int64)
    reasons = []
    if len(df) < 1000:
        reasons.append(f"too few bars ({len(df)})")
        return reasons
    expected = int((t[-1] - t[0]) // TF_MS[tf]) + 1
    if (expected - len(df)) / expected > 0.02:
        reasons.append(f"gaps {expected - len(df)}/{expected}")
    if float((df["cvd_bar"] != 0).mean()) < 0.5:
        reasons.append("cvd dead")
    if df["oi"].nunique() < 100:
        reasons.append("oi not live")
    if df["funding"].nunique() < 5:
        reasons.append("funding not live")
    return reasons


def cell_sd(sig: np.ndarray, fwd: np.ndarray, half: np.ndarray) -> float:
    m = (sig != 0) & ~np.isnan(fwd) & half
    if int(m.sum()) < 2:
        return float("nan")
    return float(np.std(sig[m] * fwd[m]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    p.add_argument("--tfs", default="15m,1h")
    p.add_argument("--horizons", default="4,8,16")
    p.add_argument("--data-dir", default="data/native")
    p.add_argument("--out", default="results/setup_probe_full.csv")
    p.add_argument("--split", type=float, default=0.5,
                   help="IN-half fraction (robustness: try 0.4 / 0.6)")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    print("=" * 100)
    print(f"  COMPOSITE-SETUP WALK-FORWARD PROBE — {len(symbols)} symbols x "
          f"{len(tfs)} TF x {len(SETUPS)} setups x H={horizons}")
    print(f"  judge: entry open(T+1), exit open(T+1+H), taker RT "
          f"{TAKER_RT*100:.2f}%  |  PASS: net_OUT>+{PASS_NET*100:.2f}%, "
          f"net_IN>0, n>={MIN_N} both halves")
    print("=" * 100)

    rows: list[dict] = []
    excluded: list[str] = []
    e_false = 0.0
    n_evaluable = 0

    for sym in symbols:
        for tf in tfs:
            fp = Path(args.data_dir) / sym / tf / "bars.parquet"
            if not fp.is_file():
                excluded.append(f"{sym}/{tf}: file missing")
                continue
            df = pd.read_parquet(fp)
            bad = quick_validate(df, tf)
            if bad:
                excluded.append(f"{sym}/{tf}: {', '.join(bad)}")
                continue

            t_ms = df["open_time"].values.astype(np.int64)
            n = len(df)
            mid = int(n * args.split)
            half_in = np.zeros(n, bool)
            half_in[:mid] = True
            half_out = ~half_in
            fwds = {H: forward_returns(t_ms, df["open"].values, H, TF_MS[tf])
                    for H in horizons}

            for name, fn in SETUPS.items():
                sig = np.nan_to_num(np.asarray(fn(df, half_in), dtype=np.float64))
                for H in horizons:
                    ci = cell(sig, fwds[H], half_in)
                    co = cell(sig, fwds[H], half_out)
                    status = "insufficient"
                    if ci["ok"] and co["ok"]:
                        n_evaluable += 1
                        if co["net"] > PASS_NET * 100 and ci["net"] > 0:
                            status = "PASS"
                        elif ci["net"] > 0 and 0 < co["net"] <= PASS_NET * 100:
                            status = "marginal"
                        else:
                            status = "dead"
                        sd_o = cell_sd(sig, fwds[H], half_out)
                        sd_i = cell_sd(sig, fwds[H], half_in)
                        if np.isfinite(sd_o) and sd_o > 0 and \
                           np.isfinite(sd_i) and sd_i > 0:
                            p_o = 1 - phi(GROSS_PASS / (sd_o / sqrt(co["n"])))
                            p_i = 1 - phi(TAKER_RT / (sd_i / sqrt(ci["n"])))
                            e_false += p_o * p_i
                    rows.append({
                        "symbol": sym, "tf": tf, "setup": name, "H": H,
                        "n_in": ci["n"], "net_in": ci.get("net", np.nan),
                        "hit_in": ci.get("hit", np.nan),
                        "n_out": co["n"], "net_out": co.get("net", np.nan),
                        "hit_out": co.get("hit", np.nan),
                        "status": status,
                    })
            print(f"  [done] {sym} {tf}  ({len(SETUPS)*len(horizons)} cells)")

    res = pd.DataFrame(rows)
    outfp = Path(args.out)
    outfp.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(outfp, index=False, float_format="%.4f")

    # ── terminal summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    if excluded:
        print("  excluded from probe:")
        for e in excluded:
            print(f"    - {e}")

    ev = res[res["status"].isin(["PASS", "marginal", "dead"])]
    passed = res[res["status"] == "PASS"].sort_values("net_out", ascending=False)
    marg = res[res["status"] == "marginal"].sort_values("net_out", ascending=False)

    print(f"\n  evaluable cells (n>={MIN_N} both halves): {len(ev)} "
          f"of {len(res)} total")
    print(f"  PASSED: {len(passed)}   |   E[false passes | zero edge] ~ "
          f"{e_false:.2f}  (normal approx, per-cell sigma;")
    print("          UNDERestimate — overlapping entries for H>1 inflate "
          "effective n)")
    if len(passed):
        print("\n  PASSED cells:")
        for _, r in passed.iterrows():
            print(f"    {r['symbol']:<9s} {r['tf']:<4s} {r['setup']:<21s} "
                  f"H={r['H']:<3d} OUT net={r['net_out']:+.3f}% "
                  f"hit={r['hit_out']:.1f}% n={r['n_out']}  |  "
                  f"IN net={r['net_in']:+.3f}% n={r['n_in']}")
    print(f"\n  top-10 MARGINAL (0 < net_OUT <= +{PASS_NET*100:.2f}%):")
    for _, r in marg.head(10).iterrows():
        print(f"    {r['symbol']:<9s} {r['tf']:<4s} {r['setup']:<21s} "
              f"H={r['H']:<3d} OUT net={r['net_out']:+.3f}% "
              f"hit={r['hit_out']:.1f}% n={r['n_out']}  |  "
              f"IN net={r['net_in']:+.3f}%")
    if marg.empty:
        print("    (none)")

    # ── per-setup transferability: median OUT expectancy across symbols ─────
    print("\n  PER-SETUP TRANSFERABILITY (median net_OUT across symbols; "
          "best {tf,H} per setup):")
    print(f"    {'setup':<21s} {'tf':<4s} {'H':>3s} {'med net_OUT':>12s} "
          f"{'pos/n symbols':>14s}")
    agg_rows = []
    for (name, tf, H), g in ev.groupby(["setup", "tf", "H"]):
        if len(g) < 3:
            continue
        agg_rows.append({"setup": name, "tf": tf, "H": H,
                         "med": g["net_out"].median(),
                         "pos": int((g["net_out"] > 0).sum()), "n": len(g)})
    agg = pd.DataFrame(agg_rows)
    if not agg.empty:
        best = (agg.sort_values("med", ascending=False)
                .groupby("setup", as_index=False).first()
                .sort_values("med", ascending=False))
        for _, r in best.iterrows():
            print(f"    {r['setup']:<21s} {r['tf']:<4s} {r['H']:>3d} "
                  f"{r['med']:>+11.3f}% {r['pos']:>7d}/{r['n']}")

    # ── final verdict paragraph ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    print(f"  {len(passed)} cell(s) passed the pre-registered threshold "
          f"against ~{e_false:.2f} expected false passes under zero edge.")
    if len(passed) == 0:
        print("  -> No composite setup carries a fee-clearing directional "
              "edge on this basket under the")
        print("     pre-registered judge. Same verdict as the generic probe "
              "— closed honestly.")
    elif len(passed) <= max(1.0, 2 * e_false):
        print("  -> Pass count is within the expected false-positive band — "
              "indistinguishable from noise.")
        print("     Do NOT promote these cells without out-of-basket "
              "confirmation.")
    else:
        print("  -> Pass count EXCEEDS the false-positive expectation. "
              "Candidate edges above; before ANY RL:")
        print("     confirm on a shifted half-split and an untouched symbol.")
    print(f"\n  full table -> {outfp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
