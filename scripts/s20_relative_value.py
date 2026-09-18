#!/usr/bin/env python3
"""
s20_relative_value.py — S20 RELATIVE VALUE: cross-sectional, dollar-neutral
long/short over the 13-symbol universe.  A DIFFERENT strategy class with its
OWN pre-registered protocol — not a 20th single-asset setup.

Rules (the S-C rotation lesson, -56%/yr churn, is baked in):
  * rebalance EXACTLY once per 24h: decision at the close of the 23:00 UTC
    bar, execution at the open of the 00:00 UTC bar; min-hold 24h by cadence;
  * ranking signals, tested separately:
      (a) momentum: ret_24h = close/close[-24] - 1
      (b) flow:     z-score over 168h of (cvd_bar / volume)  — volume-
                    normalized, otherwise cross-symbol comparison is
                    meaningless (coin units differ);
  * portfolios: k=1 (long top-1 / short bottom-1) and k=2, equal dollar legs;
  * costs: initial open 0.055%/leg; each slot change = 0.11% RT on that leg
    (0.055 exit of the old + 0.055 entry of the new); final close 0.055%/leg;
    only genuinely changed slots pay;
  * capital: deployed = sum of BOTH legs' notionals (2k legs).

PRE-REGISTERED THRESHOLD (before the run): S20 passes iff
  (1) mean net_OUT > +0.44% per pair-cycle (= 2x the double RT),
  (2) sign(mean net_IN) == sign(mean net_OUT),
  (3) >= 50 pair-cycles on OUT.
HONEST NOTE (printed): 6 months of daily rebalances is a SMALL sample
(~90 cycles per half) — the result, either way, is preliminary.
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

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
           "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
           "ADAUSDT", "XRPUSDT", "DOTUSDT"]
TF_MS = 3_600_000
RT = 0.11                 # % round trip per leg swap
HALF_RT = 0.055
PASS_CYCLE = 0.44         # % net per pair-cycle, pre-registered
MIN_CYCLES = 50
RES = Path("out")                 # results/ is the immutable evidence base
EVIDENCE = Path("results")


def load() -> dict:
    op, cl, cv, vol = {}, {}, {}, {}
    grid = None
    for s in SYMBOLS:
        df = pd.read_parquet(f"data/native/{s}/1h/bars.parquet")
        df = df.set_index("open_time")
        op[s], cl[s] = df["open"], df["close"]
        cv[s], vol[s] = df["cvd_bar"], df["volume"]
        grid = df.index if grid is None else grid.union(df.index)
    O = pd.DataFrame({s: op[s].reindex(grid) for s in SYMBOLS})
    C = pd.DataFrame({s: cl[s].reindex(grid) for s in SYMBOLS})
    X = pd.DataFrame({s: (cv[s] / vol[s].replace(0, np.nan)).reindex(grid)
                      for s in SYMBOLS})
    return {"t": np.array(grid, dtype=np.int64), "O": O, "C": C, "X": X}


def signals(C: pd.DataFrame, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mom = C / C.shift(24) - 1.0
    mu = X.rolling(168, min_periods=168).mean()
    sd = X.rolling(168, min_periods=168).std()
    flowz = (X - mu) / sd.replace(0.0, np.nan)
    return {"momentum": mom, "flow_z": flowz}


def run_config(t, O, sigdf, k: int, mid: int) -> pd.DataFrame:
    """Daily cycles; returns one row per pair-cycle with net pnl % deployed."""
    hour = (t // TF_MS) % 24
    dec = np.where(hour == 23)[0]           # decision bars (close 00:00 UTC)
    dec = dec[(dec + 1) < len(t)]
    deployed = 2 * k
    longs: set = set()
    shorts: set = set()
    cycles = []
    for j in range(len(dec) - 1):
        T, Tn = dec[j], dec[j + 1]
        e, en = T + 1, Tn + 1               # execution bars (00:00 opens)
        row = sigdf.iloc[T]
        ranks = row.dropna()
        if len(ranks) < 2 * k:
            continue
        new_long = set(ranks.nlargest(k).index)
        new_short = set(ranks.nsmallest(k).index)
        if new_long & new_short:
            continue
        # price the cycle FIRST; a skipped (unpriceable) cycle must leave
        # holdings AND costs untouched — the old order updated holdings
        # before the NaN check, ghosting the swap fees (v1.0.1 fix,
        # post-dates the shipped CSVs)
        pnl = 0.0
        ok = True
        for s in new_long:
            a, b = O[s].iloc[e], O[s].iloc[en]
            if np.isnan(a) or np.isnan(b) or a <= 0:
                ok = False
                break
            pnl += (b / a - 1.0) * 100.0
        for s in new_short:
            a, b = O[s].iloc[e], O[s].iloc[en]
            if np.isnan(a) or np.isnan(b) or a <= 0:
                ok = False
                break
            pnl -= (b / a - 1.0) * 100.0
        if not ok:
            continue
        # costs: initial open / slot swaps
        cost = 0.0
        if not longs and not shorts:
            cost += HALF_RT * 2 * k
        else:
            swaps = (len(new_long - longs) + len(new_short - shorts))
            cost += RT * swaps
        longs, shorts = new_long, new_short
        cycles.append({"dec_bar": T, "t_ms": int(t[e]),
                       "gross_pct": pnl / deployed,
                       "cost_pct": cost / deployed,
                       "net_pct": (pnl - cost) / deployed,
                       "half": "IN" if T < mid else "OUT"})
    if cycles:                              # final close costs on last cycle
        cycles[-1]["cost_pct"] += HALF_RT * 2 * k / deployed
        cycles[-1]["net_pct"] -= HALF_RT * 2 * k / deployed
    return pd.DataFrame(cycles)


def main() -> int:
    global RES
    ap = argparse.ArgumentParser(prog="s20_relative_value")
    ap.add_argument("--outdir", default=str(RES),
                    help="where s20_*.csv are written (default: out/). "
                         "'results' overwrites the campaign evidence base - "
                         "full reproduction only.")
    args = ap.parse_args()
    RES = Path(args.outdir)
    if RES.resolve() == EVIDENCE.resolve():
        print("  !! --outdir results: OVERWRITING the campaign evidence base. "
              "Only meaningful with all 13 symbols fetched for the campaign "
              "window.")
    missing = [s for s in SYMBOLS
               if not Path(f"data/native/{s}/1h/bars.parquet").is_file()]
    if missing:
        print("  nothing evaluated: S20 is cross-sectional and needs all 13 "
              f"symbols; missing data/native/<SYM>/1h/bars.parquet for "
              f"{', '.join(missing)}. Nothing written.")
        return 2
    RES.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("  S20 RELATIVE VALUE — cross-sectional dollar-neutral, separate "
          "pre-registered protocol")
    print(f"  pass iff: mean net_OUT > +{PASS_CYCLE}%/pair-cycle, sign IN==OUT,"
          f" >={MIN_CYCLES} OUT cycles")
    print("  HONEST NOTE: ~90 daily cycles per half = SMALL sample; either "
          "outcome is preliminary.")
    print("=" * 96)

    M = load()
    t, O, C, X = M["t"], M["O"], M["C"], M["X"]
    mid = len(t) // 2
    sigs = signals(C, X)

    all_rows = []
    eq_cols = {}
    verdicts = []
    for sname, sdf in sigs.items():
        for k in (1, 2):
            cyc = run_config(t, O, sdf, k, mid)
            tag = f"{sname}_k{k}"
            if cyc.empty:
                print(f"  [warn] {tag}: no cycles")
                continue
            cyc.to_csv(RES / f"s20_cycles_{tag}.csv", index=False,
                       float_format="%.4f")
            eq = cyc.set_index("t_ms")["net_pct"].cumsum()
            eq_cols[tag] = eq
            for half in ("IN", "OUT"):
                g = cyc[cyc["half"] == half]
                if len(g) == 0:
                    continue
                e = g["net_pct"].cumsum()
                dd = float((e.cummax() - e).max())
                ann = g["net_pct"].mean() * 365.0
                all_rows.append({
                    "config": tag, "half": half, "cycles": len(g),
                    "gross_mean": g["gross_pct"].mean(),
                    "cost_mean": g["cost_pct"].mean(),
                    "net_mean": g["net_pct"].mean(),
                    "net_total": g["net_pct"].sum(),
                    "hit": (g["net_pct"] > 0).mean() * 100,
                    "ann_dep": ann, "maxdd": dd})
            gi = cyc[cyc["half"] == "IN"]["net_pct"]
            go = cyc[cyc["half"] == "OUT"]["net_pct"]
            c1 = go.mean() > PASS_CYCLE
            c2 = len(gi) > 0 and len(go) > 0 and \
                np.sign(gi.mean()) == np.sign(go.mean()) != 0
            c3 = len(go) >= MIN_CYCLES
            verdicts.append((tag, go.mean() if len(go) else np.nan,
                             gi.mean() if len(gi) else np.nan,
                             len(go), c1, c2, c3))

    res = pd.DataFrame(all_rows)
    res.to_csv(RES / "s20_summary.csv", index=False, float_format="%.4f")
    eqdf = pd.DataFrame(eq_cols).sort_index().ffill()
    eqdf.insert(0, "time_utc", [
        datetime.fromtimestamp(m / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
        for m in eqdf.index])
    eqdf.to_csv(RES / "s20_equity.csv", float_format="%.4f")

    print("\n  SUMMARY (% on dollar-neutral deployed = both legs):")
    print(f"  {'config':<14s} {'half':<4s} {'cyc':>4s} {'gross/cyc':>10s} "
          f"{'cost/cyc':>9s} {'net/cyc':>8s} {'total%':>8s} {'hit%':>6s} "
          f"{'ann%':>8s} {'maxDD%':>7s}")
    for _, r in res.iterrows():
        print(f"  {r['config']:<14s} {r['half']:<4s} {r['cycles']:>4d} "
              f"{r['gross_mean']:>+10.3f} {r['cost_mean']:>9.3f} "
              f"{r['net_mean']:>+8.3f} {r['net_total']:>+8.2f} "
              f"{r['hit']:>6.1f} {r['ann_dep']:>+8.1f} {r['maxdd']:>7.2f}")

    print("\n" + "=" * 96)
    print("  VERDICT — S20 (own protocol; small-sample caveat applies)")
    print("=" * 96)
    any_pass = False
    for tag, mo, mi, no, c1, c2, c3 in verdicts:
        ok = c1 and c2 and c3
        any_pass |= ok
        print(f"  {tag:<14s} net_OUT/cyc={mo:+.3f}% (bar +{PASS_CYCLE}%) "
              f"{'PASS' if c1 else 'FAIL'} | IN {mi:+.3f}% sign "
              f"{'PASS' if c2 else 'FAIL'} | n_OUT={no} "
              f"{'PASS' if c3 else 'FAIL'}  => "
              f"{'CONFIRMED(prelim)' if ok else 'not confirmed'}")
    print()
    if not any_pass:
        print("  => No relative-value config clears its pre-registered bar "
              "on this (small) sample.")
    else:
        print("  => Survivor(s) above are PRELIMINARY (small sample) — "
              "require a longer window before any capital.")
    print(f"\n  cycles -> {RES}/s20_cycles_<config>.csv   summary -> "
          f"{RES}/s20_summary.csv   equity -> {RES}/s20_equity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
