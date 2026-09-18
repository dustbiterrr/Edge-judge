#!/usr/bin/env python3
"""
appendix_c_statarb.py — APPENDIX C probe: structurally NEW strategy classes
only (Appendix B's finality for single-asset directional setups still holds).

DECLARATION (printed in the verdict header, before any result):
  Appendix C opens ONLY for classes not covered by Appendix B's finality —
    S22  market-neutral cointegration StatArb (a spread mean-revert mechanism,
         fundamentally unlike the failed momentum-ranker S20);
    S23  temporal weekend liquidity-vacuum (exploits a time window, not a
         price pattern).
  S21 (liquidation squeeze) is EXCLUDED from Appendix C as a direct twin of the
  dead S08+S19 (single-asset directional, covered by B's finality); it is run
  ONLY as a control with a pre-declared prediction: "insufficient, reproduces
  the fate of S19".

PRE-DECLARED SAMPLE-SIZE LIMIT (printed too): S22 and S23 are RARE by
construction over 6 months (cointegration yields few entries; the weekend
window is ~2/7 of time).  The likely outcome is that the n-guard marks them
INSUFFICIENT — a verdict of "NOT ENOUGH DATA", not "edge / no edge".  The
n-threshold is fixed now and is NOT relaxed afterward.

Interpretation note (pre-registered) for S22's "regression window 7-14 days":
the spec pins beta AND spread mean/std to the IN half ("only from IN, applied
to OUT") — which is exactly correct for cointegration (a cointegrated spread is
stationary, so IN estimates hold out-of-sample).  The 7/14-day parameter is
therefore applied as the holding TIME-LIMIT (force-close if |Z| has not
reverted within the window).  Rolling-window z would be a one-line change.

Methodology (unchanged): no look-ahead (signal <= close(T), enter open(T+1));
walk-forward 50/50 by time; quantiles/params from the IN half; non-overlapping
execution from the first run; verdict guarded by n_out >= class threshold;
cumulative E[false] over all ~1200 prior cells + Appendix C printed.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s02_nonoverlap_backtest import half_stats, simulate  # noqa: E402
from setup_library import s21_liq_squeeze_control, s23_weekend_vacuum  # noqa: E402

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
except Exception:                                          # pragma: no cover
    sm = None

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
           "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
           "ADAUSDT", "XRPUSDT", "DOTUSDT"]
TF_MS = {"15m": 900_000, "1h": 3_600_000}
FEE_RT = 0.11                       # % taker round trip, one leg

# pre-registered thresholds (fixed now)
S22_BAR, S22_GUARD = 0.44, 30       # net_OUT %/cycle (double RT), cycles
S23_BAR, S23_GUARD = 0.22, 30
S21_BAR, S21_GUARD = 0.22, 30       # control
ADF_P = 0.05
Z_IN, Z_OUT = 2.5, 0.5
PAIR_COST = 2 * FEE_RT              # 0.22%: both legs round-tripped

# Unique H1-2026 cells with a CSV in results/ (scripts/count_cells.py):
# 780 setup wave + 234 untouched + 156 appendix B + 4 S20 + 27 funding.
# The ~90 generic-probe cells have no CSV and are not counted.  PRIOR_EFALSE
# is the ledger the earlier probes printed at run time; the CSVs do not
# carry the per-trade sigma needed to recompute it.
PRIOR_CELLS, PRIOR_EFALSE = 1201, 1.40
DATA = Path("data/native")
RES = Path("out")                 # results/ is the immutable evidence base
EVIDENCE = Path("results")


def phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def load(sym: str, tf: str) -> pd.DataFrame | None:
    fp = DATA / sym / tf / "bars.parquet"
    return pd.read_parquet(fp) if fp.is_file() else None


def e_false_cell(net: np.ndarray, bar: float) -> float:
    """P(mean>bar | null mean 0) * P(mean_in>0 | null) via normal approx."""
    if len(net) < 3 or net.std() == 0:
        return 0.0
    se = net.std() / sqrt(len(net))
    return (1 - phi(bar / se)) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# S22 — cointegration StatArb (pairs)
# ─────────────────────────────────────────────────────────────────────────────

def eg_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Engle-Granger on IN slice: OLS a ~ const + beta*b, ADF on residuals.
    Returns (adf_p, const, beta, resid_std)."""
    X = sm.add_constant(b)
    res = sm.OLS(a, X).fit()
    const, beta = float(res.params[0]), float(res.params[1])
    resid = a - (const + beta * b)
    adf_p = float(adfuller(resid, autolag="AIC")[1])
    return adf_p, const, beta, float(np.std(resid))


def sim_pair(oa, ca, ob, cb, const, beta, std, mid, tl_bars):
    """Non-overlapping spread mean-revert sim.  Z from IN-static (const,beta,
    std).  Enter open(T+1) on |Z|>2.5, exit open(tau+1) on |Z|<0.5 or after
    tl_bars.  Returns per-cycle net % rows with the entry-bar half tag."""
    z = (ca - const - beta * cb) / std if std > 0 else np.zeros_like(ca)
    n = len(ca)
    rows, nxt = [], 0
    for T in range(n - 1):
        if T < nxt or abs(z[T]) <= Z_IN:
            continue
        dir_a = -1.0 if z[T] > 0 else 1.0            # z high -> short A / long B
        dir_b = -dir_a
        ea, eb = oa[T + 1], ob[T + 1]
        if not (ea > 0 and eb > 0):
            continue
        exit_i = None
        last = min(T + tl_bars, n - 2)
        for tau in range(T + 1, last + 1):
            if abs(z[tau]) < Z_OUT:
                exit_i = tau + 1
                break
        if exit_i is None:
            exit_i = last + 1
        xa, xb = oa[exit_i], ob[exit_i]
        if not (xa > 0 and xb > 0):
            nxt = exit_i
            continue
        gross = (dir_a * (xa / ea - 1.0) + dir_b * (xb / eb - 1.0)) * 100.0
        rows.append({"net_pnl_pct": gross - PAIR_COST,
                     "half": "IN" if T < mid else "OUT"})
        nxt = exit_i
    return pd.DataFrame(rows, columns=["net_pnl_pct", "half"])


def run_s22(rows_out: list, coint_pairs: list, tl_days: int) -> None:
    tl_bars = tl_days * 24
    frames = {s: load(s, "1h") for s in SYMBOLS}
    for a_sym, b_sym in combinations(SYMBOLS, 2):
        da, db = frames[a_sym], frames[b_sym]
        if da is None or db is None:
            continue
        m = da[["open_time", "open", "close"]].merge(
            db[["open_time", "open", "close"]], on="open_time",
            suffixes=("_a", "_b"))
        if len(m) < 400:
            continue
        mid = len(m) // 2
        ca = m["close_a"].values
        cb = m["close_b"].values
        oa = m["open_a"].values
        ob = m["open_b"].values
        adf_p, const, beta, std = eg_test(ca[:mid], cb[:mid])
        if not (adf_p < ADF_P and std > 0):
            continue
        if tl_days == 7:
            coint_pairs.append((f"{a_sym[:3]}/{b_sym[:3]}", adf_p, beta))
        tr = sim_pair(oa, ca, ob, cb, const, beta, std, mid, tl_bars)
        si = half_stats(tr, "IN")
        so = half_stats(tr, "OUT")
        status = _status(si, so, S22_BAR, S22_GUARD)
        rows_out.append({
            "setup": "S22 cointegration", "key": f"{a_sym[:3]}/{b_sym[:3]}",
            "tf": "1h", "param": f"tl={tl_days}d", "adf_p": round(adf_p, 4),
            "n_in": si["n"], "net_in": si.get("mean", np.nan),
            "n_out": so["n"], "net_out": so.get("mean", np.nan),
            "hit_out": so.get("hit", np.nan), "status": status,
            "_net_out_arr": tr[tr.half == "OUT"]["net_pnl_pct"].values,
        })


# ─────────────────────────────────────────────────────────────────────────────
# S23 — weekend vacuum (calendar exit at Monday 00:00 UTC)
# ─────────────────────────────────────────────────────────────────────────────

def sim_weekend(df: pd.DataFrame, sig: np.ndarray, mid: int,
                tf_ms: int) -> pd.DataFrame:
    ot = df["open_time"].values.astype(np.int64)
    op, cl = df["open"].values, df["close"].values
    idx = pd.DatetimeIndex(pd.to_datetime(ot, unit="ms", utc=True))
    monday00 = ((idx.dayofweek.values == 0) & (idx.hour.values == 0)
                & (idx.minute.values == 0))
    n = len(df)
    rows, nxt = [], 0
    for T in range(n - 1):
        if T < nxt or sig[T] == 0:
            continue
        # exit = first Monday-00:00 bar strictly after entry bar T+1
        ex = None
        for j in range(T + 2, n):
            if monday00[j]:
                ex = j
                break
        if ex is None:
            continue
        d = float(sig[T])
        entry, exit_ = op[T + 1], cl[ex]
        if entry <= 0:
            continue
        net = d * (exit_ / entry - 1.0) * 100.0 - FEE_RT
        rows.append({"net_pnl_pct": net, "half": "IN" if T < mid else "OUT"})
        nxt = ex
    return pd.DataFrame(rows, columns=["net_pnl_pct", "half"])


def run_s23(rows_out: list) -> None:
    for tf in ("15m", "1h"):
        for sym in SYMBOLS:
            df = load(sym, tf)
            if df is None:
                continue
            n = len(df)
            mid = n // 2
            in_mask = np.zeros(n, bool)
            in_mask[:mid] = True
            sig = np.nan_to_num(np.asarray(
                s23_weekend_vacuum(df, in_mask), dtype=np.float64))
            tr = sim_weekend(df, sig, mid, TF_MS[tf])
            si, so = half_stats(tr, "IN"), half_stats(tr, "OUT")
            rows_out.append({
                "setup": "S23 weekend-vacuum", "key": sym[:6], "tf": tf,
                "param": "MonExit", "adf_p": np.nan,
                "n_in": si["n"], "net_in": si.get("mean", np.nan),
                "n_out": so["n"], "net_out": so.get("mean", np.nan),
                "hit_out": so.get("hit", np.nan),
                "status": _status(si, so, S23_BAR, S23_GUARD),
                "_net_out_arr": tr[tr.half == "OUT"]["net_pnl_pct"].values,
            })


# ─────────────────────────────────────────────────────────────────────────────
# S21 — control (single-asset directional, s02 engine, fixed H=16)
# ─────────────────────────────────────────────────────────────────────────────

def run_s21(rows_out: list) -> None:
    for sym in SYMBOLS:
        df = load(sym, "1h")
        if df is None:
            continue
        n = len(df)
        mid = n // 2
        in_mask = np.zeros(n, bool)
        in_mask[:mid] = True
        sig = np.nan_to_num(np.asarray(
            s21_liq_squeeze_control(df, in_mask), dtype=np.float64))
        tr = simulate(df, sig, 16, mid, tf_ms=TF_MS["1h"])
        si, so = half_stats(tr, "IN"), half_stats(tr, "OUT")
        rows_out.append({
            "setup": "S21 CONTROL", "key": sym[:6], "tf": "1h", "param": "H=16",
            "adf_p": np.nan, "n_in": si["n"], "net_in": si.get("mean", np.nan),
            "n_out": so["n"], "net_out": so.get("mean", np.nan),
            "hit_out": so.get("hit", np.nan),
            "status": _status(si, so, S21_BAR, S21_GUARD, control=True),
            "_net_out_arr": tr[tr.half == "OUT"]["net_pnl_pct"].values,
        })


def _status(si, so, bar, guard, control=False):
    # pre-registered guard is n_out >= class threshold ONLY (spec); sign(IN==OUT)
    # just needs IN to be non-empty. No extra n_in floor (would tighten the bar).
    if so["n"] < guard or si["n"] < 1:
        return "insufficient"
    if so["mean"] > bar and si["mean"] > 0:
        return "PASS"
    if 0 < so["mean"] <= bar and si["mean"] > 0:
        return "marginal"
    return "dead"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    global RES
    ap = argparse.ArgumentParser(prog="appendix_c_statarb")
    ap.add_argument("--outdir", default=str(RES),
                    help="where appendix_c_full.csv is written (default: "
                         "out/). 'results' overwrites the campaign evidence "
                         "base - full reproduction only.")
    args = ap.parse_args()
    RES = Path(args.outdir)
    if sm is None:
        print("[fatal] statsmodels not available (needed for S22 ADF).")
        return 2
    if RES.resolve() == EVIDENCE.resolve():
        print("  !! --outdir results: OVERWRITING the campaign evidence base. "
              "Only meaningful with all 13 symbols fetched for the campaign "
              "window.")
    print("=" * 100)
    print("  APPENDIX C — structurally NEW classes only (Appendix B finality "
          "holds for single-asset directional)")
    print("  DECLARATION: opens ONLY for S22 cointegration StatArb (spread "
          "mean-revert, unlike failed S20 ranker)")
    print("  and S23 temporal weekend liquidity-vacuum (window, not pattern). "
          "S21 = CONTROL (S08/S19 twin),")
    print("  pre-declared 'insufficient'. PRE-DECLARED: S22/S23 are RARE by "
          "construction -> likely 'NOT ENOUGH")
    print("  DATA', not 'edge/no-edge'; n-guard fixed now, NOT relaxed after. "
          "S22 z-params IN-static (cointegration")
    print("  => stationary spread); 7/14d = holding time-limit.")
    print("=" * 100)

    rows: list = []
    coint_pairs: list = []
    print("\n[S22] cointegration scan (78 pairs x tl in {7,14}d) ...")
    for tl in (7, 14):
        run_s22(rows, coint_pairs, tl)
    print(f"  cointegrated pairs on IN (ADF p<{ADF_P}): {len(coint_pairs)}")
    print("[S23] weekend liquidity-vacuum (15m + 1h x 13 symbols) ...")
    run_s23(rows)
    print("[S21] control (13 symbols, 1h) ...")
    run_s21(rows)

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "_net_out_arr"}
                        for r in rows])
    if res.empty:
        print("\n  nothing evaluated: no bars under data/native/ for the "
              "basket. Fetch first (python scripts/fetch_binance_native.py "
              "--symbol ETHUSDT --months 6); nothing written.")
        return 2
    RES.mkdir(parents=True, exist_ok=True)
    res.to_csv(RES / "appendix_c_full.csv", index=False, float_format="%.4f")

    # cumulative false-pass ledger over EVALUABLE cells
    e_false = 0.0
    n_eval = 0
    for r in rows:
        if r["status"] in ("PASS", "marginal", "dead"):
            n_eval += 1
            bar = (S22_BAR if r["setup"].startswith("S22")
                   else S21_BAR if r["setup"].startswith("S21") else S23_BAR)
            e_false += e_false_cell(r["_net_out_arr"], bar)

    # ── cointegrated pairs table ─────────────────────────────────────────────
    print("\n  COINTEGRATED PAIRS ON IN (Engle-Granger ADF p<0.05):")
    if coint_pairs:
        for name, p, beta in sorted(coint_pairs, key=lambda x: x[1]):
            print(f"    {name:<10s} ADF p={p:.4f}  beta={beta:+.4f}")
    else:
        print("    NONE — no cointegrated pair in this window (honest verdict: "
              "no tradable spread to test).")

    # ── per-setup table ──────────────────────────────────────────────────────
    print("\n  PER-CELL RESULTS (net %/trade-or-cycle, IN | OUT):")
    print(f"  {'setup':<20s} {'key':<10s} {'tf':<4s} {'param':<8s} "
          f"{'n_in':>5s} {'net_in':>8s} {'n_out':>5s} {'net_out':>8s} "
          f"{'hit':>5s} {'status':>12s}")
    order = {"PASS": 0, "marginal": 1, "dead": 2, "insufficient": 3}
    for _, r in res.sort_values(
            by=["setup", "status", "n_out"],
            key=lambda c: c.map(order) if c.name == "status" else c).iterrows():
        ni = f"{r['net_in']:+.3f}" if pd.notna(r['net_in']) else "  -  "
        no = f"{r['net_out']:+.3f}" if pd.notna(r['net_out']) else "  -  "
        hi = f"{r['hit_out']:.0f}" if pd.notna(r['hit_out']) else " - "
        print(f"  {r['setup']:<20s} {str(r['key']):<10s} {r['tf']:<4s} "
              f"{str(r['param']):<8s} {int(r['n_in']):>5d} {ni:>8s} "
              f"{int(r['n_out']):>5d} {no:>8s} {hi:>5s} {r['status']:>12s}")

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  VERDICT — APPENDIX C")
    print("=" * 100)
    npass = int((res["status"] == "PASS").sum())
    ninsuf = int((res["status"] == "insufficient").sum())
    print(f"  cells: {len(res)} total | PASS {npass} | "
          f"marginal {int((res['status']=='marginal').sum())} | "
          f"dead {int((res['status']=='dead').sum())} | insufficient {ninsuf}")
    for setup in ("S22 cointegration", "S23 weekend-vacuum", "S21 CONTROL"):
        sub = res[res["setup"] == setup]
        if len(sub) == 0:
            continue
        ins = int((sub["status"] == "insufficient").sum())
        print(f"    {setup:<20s}: {len(sub)} cells, {ins} insufficient, "
              f"{int((sub['status']=='PASS').sum())} PASS")
    # control check
    s21 = res[res["setup"] == "S21 CONTROL"]
    if len(s21):
        ok = (s21["status"] == "insufficient").mean()
        print(f"  CONTROL S21: {ok*100:.0f}% of cells insufficient -> "
              f"pre-declared prediction {'CONFIRMED' if ok >= 0.5 else 'NOT held'}"
              f" (S19 twin reproduced).")
    print(f"\n  FALSE-PASS LEDGER: this wave {n_eval} evaluable cells, "
          f"E[false] ~ {e_false:.2f}")
    print(f"  campaign cumulative: {PRIOR_CELLS + len(res)} unique cells "
          f"with a CSV in results/ (+90 generic-probe cells, console only), "
          f"E[false] ~ {PRIOR_EFALSE + e_false:.2f} (prior waves' E "
          f"recorded at run time, not reproducible from results/)")
    print()
    if npass == 0:
        print("  => No new-class edge cleared its pre-registered bar. Most "
              "cells are INSUFFICIENT (rare by")
        print("     construction, as pre-declared) — a 'NOT ENOUGH DATA' "
              "verdict, not 'edge / no edge'. The")
        print("     structural doors are recorded as untested-at-power, not "
              "falsified; the campaign's 0 stands.")
    else:
        print(f"  => {npass} cell(s) PASSED vs E[false] ~{e_false:.2f}. "
              "Confirm on a shifted split + untouched")
        print("     symbols before ANY capital — a lone pass in a rare class "
              "is prior-suspect.")
    print(f"\n  full table -> {RES / 'appendix_c_full.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
