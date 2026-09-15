#!/usr/bin/env python3
"""
s02_nonoverlap_backtest.py — JUDGE (not optimizer) for the S02 VWAP-FADE
survivor: does the edge survive honest one-position-at-a-time execution?

Setup rules are FROZEN: imported verbatim from setup_library.s02_vwap_fade
(q80 stretch threshold fitted on the IN half, rolling VWAP-24).  No new
parameters, no sweeps beyond the pre-existing H in {8, 16}.

EXECUTION (pre-registered):
  * signal on bar T (data <= close(T)) -> enter at open(T+1), taker;
  * hold; exit at close(T+H), taker; RT fee 0.11% per trade;
  * while a position is open ALL new signals are ignored;
  * next tradable signal: strictly AFTER the exit bar (T' >= T+H+1) —
    a signal forming on the exit bar is simultaneous with the exit, not after;
  * trade window T..T+H must be time-contiguous, else the signal is untradable
    (no price interpolation, ever);
  * a trade belongs to IN or OUT by its SIGNAL bar (decision time).

PRE-REGISTERED CONFIRMATION CRITERIA (per H; S02 confirmed if >=1 H passes all;
family multiplicity x2 is printed with E[false]):
  (a) median over 13 symbols of per-symbol mean net_OUT > +0.11%
  (b) >= 8/13 symbols with mean net_OUT > 0
  (c) total OUT non-overlapping trades >= 150
  (d) sign of the 13-symbol median of mean net_IN matches OUT

Usage:
    python scripts/s02_nonoverlap_backtest.py                   # -> out/
    python scripts/s02_nonoverlap_backtest.py --outdir results  # full repro:
        OVERWRITES the campaign evidence in results/ - needs all 13 symbols
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from math import erf, exp, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_library import s02_vwap_fade  # noqa: E402  (frozen rules)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
           "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
           "ADAUSDT", "XRPUSDT", "DOTUSDT"]
TF, TF_MS = "1h", 3_600_000
HS = (8, 16)
FEE_RT = 0.11                     # % per round trip, both legs taker
DATA_DIR = Path("data/native")
OUT_DEFAULT = Path("out")         # results/ is the immutable evidence base
EVIDENCE = Path("results")


def phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(
        "%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# Non-overlapping simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, sig: np.ndarray, H: int,
             mid: int, tf_ms: int = TF_MS) -> pd.DataFrame:
    t = df["open_time"].values.astype(np.int64)
    op = df["open"].values
    cl = df["close"].values
    n = len(df)
    trades = []
    next_ok = 0
    # v1.0.1: range(n - H), not range(n - H - 1) — the final signal at
    # T = n-H-1 (exit on the last bar) was silently skipped.  Post-dates the
    # shipped CSVs; adds at most one trade at the very end of a series.
    for T in range(n - H):
        if T < next_ok or sig[T] == 0:
            continue
        if t[T + H] - t[T] != H * tf_ms:      # gap inside window -> untradable
            continue
        d = float(sig[T])
        entry, exit_ = op[T + 1], cl[T + H]
        if entry <= 0:
            continue
        net = d * (exit_ / entry - 1.0) * 100.0 - FEE_RT
        trades.append({
            "signal_time": iso(t[T]), "entry_time": iso(t[T + 1]),
            "exit_time": iso(t[T + H]), "dir": int(d),
            "entry_px": entry, "exit_px": exit_,
            "net_pnl_pct": net, "half": "IN" if T < mid else "OUT",
            "exit_ms": int(t[T + H]),
        })
        next_ok = T + H + 1                   # strictly after the exit bar
    cols = ["signal_time", "entry_time", "exit_time", "dir", "entry_px",
            "exit_px", "net_pnl_pct", "half", "exit_ms"]
    return pd.DataFrame(trades, columns=cols)


def half_stats(tr: pd.DataFrame, half: str) -> dict:
    g = tr[tr["half"] == half]["net_pnl_pct"]
    if len(g) == 0:
        return {"n": 0, "mean": np.nan, "med": np.nan, "hit": np.nan,
                "maxdd": np.nan}
    eq = g.cumsum()
    dd = float((eq.cummax() - eq).max())
    return {"n": int(len(g)), "mean": float(g.mean()), "med": float(g.median()),
            "hit": float((g > 0).mean() * 100), "maxdd": dd}


def run_lengths(sig: np.ndarray) -> list[int]:
    runs, cur = [], 0
    for v in sig:
        if v != 0:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="s02_nonoverlap_backtest",
        description="S02 VWAP-fade one-position-at-a-time judge.")
    ap.add_argument("--outdir", default=str(OUT_DEFAULT),
                    help="where trade logs and equity are written "
                         "(default: out/). 'results' overwrites the campaign "
                         "evidence base - full reproduction only.")
    args = ap.parse_args()
    res_dir = Path(args.outdir)
    res_dir.mkdir(parents=True, exist_ok=True)
    if res_dir.resolve() == EVIDENCE.resolve():
        print("  !! --outdir results: OVERWRITING the campaign evidence base. "
              "Only meaningful with all 13 symbols fetched for the campaign "
              "window.")
    print("=" * 96)
    print("  S02 VWAP-FADE — NON-OVERLAPPING JUDGE  (frozen rules, "
          f"entry open(T+1), exit close(T+H), taker RT {FEE_RT:.2f}%)")
    print("  confirm iff (a) med13 mean net_OUT > +0.11%  (b) >=8/13 net_OUT>0"
          "  (c) sum n_OUT >= 150  (d) IN sign matches")
    print("=" * 96)

    rows = []
    all_trades: dict[int, list[pd.DataFrame]] = {H: [] for H in HS}
    all_runs: list[int] = []

    for sym in SYMBOLS:
        fp = DATA_DIR / sym / TF / "bars.parquet"
        if not fp.is_file():
            print(f"  [skip] {sym}: {fp} missing")
            continue
        df = pd.read_parquet(fp)
        n = len(df)
        mid = n // 2                            # same 50/50 as the main probe
        half_in = np.zeros(n, bool)
        half_in[:mid] = True
        sig = np.nan_to_num(np.asarray(
            s02_vwap_fade(df, half_in), dtype=np.float64))
        all_runs += run_lengths(sig)

        for H in HS:
            tr = simulate(df, sig, H, mid)
            tr.drop(columns=["exit_ms"]).to_csv(
                res_dir / f"s02_trades_{sym}_{H}.csv", index=False,
                float_format="%.6f")
            all_trades[H].append(tr.assign(symbol=sym))
            si, so = half_stats(tr, "IN"), half_stats(tr, "OUT")
            rows.append({"symbol": sym, "H": H,
                         "n_in": si["n"], "mean_in": si["mean"],
                         "med_in": si["med"], "hit_in": si["hit"],
                         "dd_in": si["maxdd"],
                         "n_out": so["n"], "mean_out": so["mean"],
                         "med_out": so["med"], "hit_out": so["hit"],
                         "dd_out": so["maxdd"]})
        print(f"  [done] {sym}")

    n_eval = len({r["symbol"] for r in rows})
    print(f"\n  {n_eval}/{len(SYMBOLS)} symbols evaluated "
          f"({len(SYMBOLS) - n_eval} skipped: no local data)")

    res = pd.DataFrame(rows)
    if res.empty:
        print("\n  no symbols with data — run scripts/fetch_binance_native.py "
              "first (data/native/<SYMBOL>/1h/bars.parquet).")
        return 0

    # ── summary table ────────────────────────────────────────────────────────
    print("\n  SUMMARY (net %/trade, fee included; DD = cum-PnL peak-to-trough "
          "of the trade sequence)")
    print(f"  {'symbol':<9s} {'H':>2s} | {'n_IN':>4s} {'mean_IN':>8s} "
          f"{'hit_IN':>7s} {'DD_IN':>6s} | {'n_OUT':>5s} {'mean_OUT':>9s} "
          f"{'med_OUT':>8s} {'hit_OUT':>8s} {'DD_OUT':>7s}")
    for _, r in res.iterrows():
        print(f"  {r['symbol']:<9s} {r['H']:>2d} | {r['n_in']:>4d} "
              f"{r['mean_in']:>+8.3f} {r['hit_in']:>6.1f}% {r['dd_in']:>6.2f} "
              f"| {r['n_out']:>5d} {r['mean_out']:>+9.3f} {r['med_out']:>+8.3f} "
              f"{r['hit_out']:>7.1f}% {r['dd_out']:>7.2f}")

    # ── overlap diagnostic ───────────────────────────────────────────────────
    ar = np.array(all_runs)
    print("\n  OVERLAP DIAGNOSTIC — stretch-episode length (consecutive "
          "signal-active bars), all symbols:")
    if len(ar):
        print(f"    episodes={len(ar)}  median={np.median(ar):.0f}  "
              f"p75={np.quantile(ar, .75):.0f}  p90={np.quantile(ar, .90):.0f}  "
              f"max={ar.max()}  mean={ar.mean():.1f}")
        print("    -> in the per-bar probe one episode was counted "
              "~mean-length times; here it is ONE trade.")
    else:
        print("    no signal episodes on the available data.")

    # ── portfolio equity ─────────────────────────────────────────────────────
    eq_frames = {}
    for H in HS:
        tr = pd.concat(all_trades[H], ignore_index=True)
        pnl = tr.groupby("exit_ms")["net_pnl_pct"].sum().sort_index()
        eq = pnl.cumsum() / len(SYMBOLS)        # equal capital split, % terms
        eq_frames[H] = eq
        # OUT-segment DD: from the earliest OUT signal time
        out_ms = tr[tr["half"] == "OUT"]["exit_ms"]
        if len(out_ms):
            eo = eq[eq.index >= out_ms.min()]
            dd_full = float((eq.cummax() - eq).max())
            dd_out = float((eo.cummax() - eo).max()) if len(eo) else np.nan
            print(f"\n  PORTFOLIO H={H}: final equity {eq.iloc[-1]:+.2f}%  "
                  f"maxDD full {dd_full:.2f}%  maxDD OUT-segment {dd_out:.2f}%"
                  f"  trades {len(tr)}")
    grid = sorted(set().union(*[set(e.index) for e in eq_frames.values()]))
    eq_csv = pd.DataFrame(index=pd.Index(grid, name="exit_ms"))
    for H in HS:
        eq_csv[f"equity_H{H}_pct"] = eq_frames[H].reindex(grid).ffill().fillna(0)
    eq_csv.insert(0, "time_utc", [iso(int(m)) for m in grid])
    eq_csv.to_csv(res_dir / "s02_equity.csv", index=True, float_format="%.4f")

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("  VERDICT (pre-registered criteria, per H)")
    print("=" * 96)
    confirmed_any = False
    for H in HS:
        g = res[res["H"] == H]
        med_out = float(g["mean_out"].median())
        med_in = float(g["mean_in"].median())
        n_pos = int((g["mean_out"] > 0).sum())
        n_tot = int(g["n_out"].sum())
        a = med_out > 0.11
        b = n_pos >= 8
        c = n_tot >= 150
        d = np.sign(med_in) == np.sign(med_out) and med_out != 0

        # E[false] under zero edge (null per-trade mean = -FEE_RT):
        p_a_list, p_b_list = [], []
        tr = pd.concat(all_trades[H], ignore_index=True)   # once, not per symbol
        for sym in g["symbol"]:
            gs = tr[(tr["symbol"] == sym) & (tr["half"] == "OUT")]["net_pnl_pct"]
            if len(gs) < 5:
                continue
            se = gs.std() / sqrt(len(gs))
            p_a_list.append(1 - phi((0.11 - (-FEE_RT)) / se))
            p_b_list.append(1 - phi((0.0 - (-FEE_RT)) / se))
        lam_a, lam_b = sum(p_a_list), sum(p_b_list)
        # P(>=7 of 13 beyond +0.11) under Poisson(lam_a); criterion (a) needs
        # the MEDIAN above the bar, i.e. >=7 symbols
        def pois_tail(lam: float, k: int) -> float:
            term, s = exp(-lam), exp(-lam)
            for i in range(1, k):
                term *= lam / i
                s += term
            return max(0.0, 1.0 - s)
        p_false = min(pois_tail(lam_a, 7), pois_tail(lam_b, 8))
        print(f"\n  H={H}:")
        print(f"    (a) median mean net_OUT = {med_out:+.3f}%  "
              f"(bar +0.110%)              -> {'PASS' if a else 'FAIL'}")
        print(f"    (b) symbols net_OUT>0   = {n_pos}/13  (bar >=8)"
              f"                      -> {'PASS' if b else 'FAIL'}")
        print(f"    (c) total n_OUT         = {n_tot}  (bar >=150)"
              f"                       -> {'PASS' if c else 'FAIL'}")
        print(f"    (d) IN median sign      = {med_in:+.3f}% vs OUT "
              f"{med_out:+.3f}%           -> {'PASS' if d else 'FAIL'}")
        print(f"    P(false | zero edge) ~ {p_false:.2e}  "
              f"(x2 for two H variants; trades now non-overlapping so the "
              f"approximation is meaningful)")
        if a and b and c and d:
            confirmed_any = True
            print(f"    => H={H}: ALL CRITERIA PASS")
        else:
            print(f"    => H={H}: NOT confirmed")

    print("\n" + "=" * 96)
    if confirmed_any:
        print("  S02 VWAP-FADE: CONFIRMED under non-overlapping execution "
              "(see per-H detail above).")
    else:
        print("  S02 VWAP-FADE: NOT confirmed — the overlapping result does "
              "not survive honest one-position")
        print("  execution. The mean-revert motif goes to 'marginal, not "
              "tradable'. Closed honestly.")
    print(f"  trade logs -> {res_dir}/s02_trades_<SYMBOL>_<H>.csv   "
          f"equity -> {res_dir}/s02_equity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
