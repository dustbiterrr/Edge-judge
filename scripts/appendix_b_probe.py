#!/usr/bin/env python3
"""
appendix_b_probe.py — APPENDIX B: the pre-registered FINAL wave of
single-asset directional setups (S14-S19) on this dataset.

DECLARATION (printed in the verdict header, binding regardless of outcome):
no further single-asset setup waves will be run on this dataset — only
structurally new strategy classes or new data.

The S02 lesson is built in from the start: NO per-bar screening exists here.
Every cell is judged with the NON-OVERLAPPING executor (one position at a
time) imported from s02_nonoverlap_backtest: entry open(T+1), exit
close(T+H_eff), re-entry strictly after the exit bar, gap-gated, taker RT.

Pre-registered thresholds (per cell):
  PASS: mean net_OUT > +0.22%/trade AND net_IN > 0 (same sign) AND
        n_OUT >= 100  (S14 is rare by construction: n_OUT >= 50 allowed,
        such passes are marked "small-n")
  0 < net_OUT <= +0.22% (with net_IN > 0) -> marginal.

False-pass ledger: E[false] for THIS wave (normal approx per cell — honest
now, trades are non-overlapping) AND the cumulative campaign ledger
(approximate for earlier per-bar waves, labeled as such).

Cells: S14 x 13 symbols x H{2,4} on 15m  +  S15-S19 x 13 symbols x H{8,16}
on 1h = 156 cells.  15m duplicates of S15-S19 are deliberately NOT run —
fewer comparisons make the final wave stricter.
"""

from __future__ import annotations

import argparse
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s02_nonoverlap_backtest import half_stats, simulate  # noqa: E402
from setup_library import APPENDIX_B_SETUPS  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
           "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
           "ADAUSDT", "XRPUSDT", "DOTUSDT"]
PASS_NET = 0.22
FEE_RT = 0.11
RES = Path("out")                 # results/ is the immutable evidence base
EVIDENCE = Path("results")

# setup -> (tf, [(H_label, H_eff)], min_n_out)
# S14: exit close(T+1+H) puts the exit H bars past the snapshot bar -> H_eff=H+1
PLAN = {
    "S14 fund-snapshot": ("15m", [(2, 3), (4, 5)], 50),
    "S15 coiled-spring": ("1h", [(8, 8), (16, 16)], 100),
    "S16 cvd-exhaustion": ("1h", [(8, 8), (16, 16)], 100),
    "S17 toxic-flow": ("1h", [(8, 8), (16, 16)], 100),
    "S18 fvg-fill": ("1h", [(8, 8), (16, 16)], 100),
    "S19 liq-cascade": ("1h", [(8, 8), (16, 16)], 100),
}

# Ledger of earlier waves.  Cells are UNIQUE hypotheses with a CSV in
# results/ (scripts/count_cells.py is the source of truth): the 780-cell
# setup wave + the 234-cell untouched-symbol confirmation.  Two things are
# deliberately NOT in PRIOR_CELLS: the ~90 generic-probe cells (no CSV was
# ever written) and the 26 S02 non-overlap runs (re-evaluations of cells
# already counted).  PRIOR_EFALSE is what the earlier probes printed at
# run time - it needs each cell's per-trade sigma, which the CSVs do not
# carry, so it is a recorded number, not one reproducible from results/.
PRIOR_CELLS = 780 + 234              # setup wave + untouched confirmation
PRIOR_CELLS_NO_CSV = 90              # generic 15m/1h probe, console only
PRIOR_EFALSE = 1.25 + 0.10 + 0.05    # recorded at run time; not reproducible


def phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def main() -> int:
    global RES
    ap = argparse.ArgumentParser(prog="appendix_b_probe")
    ap.add_argument("--outdir", default=str(RES),
                    help="where appendix_b_full.csv is written (default: "
                         "out/). 'results' overwrites the campaign evidence "
                         "base - full reproduction only.")
    args = ap.parse_args()
    RES = Path(args.outdir)
    if RES.resolve() == EVIDENCE.resolve():
        print("  !! --outdir results: OVERWRITING the campaign evidence base. "
              "Only meaningful with all 13 symbols fetched for the campaign "
              "window.")
    print("=" * 100)
    print("  APPENDIX B — PRE-REGISTERED FINAL WAVE of single-asset setups "
          "(S14-S19), non-overlapping judge")
    print("  DECLARATION: regardless of outcome, NO further single-asset "
          "setup waves on this dataset.")
    print(f"  PASS: net_OUT > +{PASS_NET}%/trade, net_IN > 0, n_OUT >= 100 "
          f"(S14: >=50, marked small-n); taker RT {FEE_RT}%")
    print("=" * 100)

    rows = []
    e_false = 0.0
    n_eval = 0
    trades_cache: dict[tuple, pd.DataFrame] = {}

    for setup, (tf, hs, min_n) in PLAN.items():
        for sym in SYMBOLS:
            fp = Path("data/native") / sym / tf / "bars.parquet"
            if not fp.is_file():
                continue
            df = pd.read_parquet(fp)
            n = len(df)
            mid = n // 2
            half_in = np.zeros(n, bool)
            half_in[:mid] = True
            sig = np.nan_to_num(np.asarray(
                APPENDIX_B_SETUPS[setup](df, half_in), dtype=np.float64))
            tf_ms = 900_000 if tf == "15m" else 3_600_000
            for H_label, H_eff in hs:
                tr = simulate(df, sig, H_eff, mid, tf_ms=tf_ms)
                trades_cache[(setup, sym, H_label)] = tr
                si = half_stats(tr, "IN") if len(tr) else {"n": 0}
                so = half_stats(tr, "OUT") if len(tr) else {"n": 0}
                status = "insufficient"
                if si["n"] >= 30 and so["n"] >= min_n:
                    n_eval += 1
                    if so["mean"] > PASS_NET and si["mean"] > 0:
                        status = ("PASS(small-n)" if so["n"] < 100 else "PASS")
                    elif si["mean"] > 0 and 0 < so["mean"] <= PASS_NET:
                        status = "marginal"
                    else:
                        status = "dead"
                    g = tr[tr["half"] == "OUT"]["net_pnl_pct"]
                    gi = tr[tr["half"] == "IN"]["net_pnl_pct"]
                    if len(g) > 4 and g.std() > 0 and len(gi) > 4 and gi.std() > 0:
                        p_o = 1 - phi((PASS_NET + FEE_RT) / (g.std() / sqrt(len(g))))
                        p_i = 1 - phi(FEE_RT / (gi.std() / sqrt(len(gi))))
                        e_false += p_o * p_i
                rows.append({
                    "setup": setup, "tf": tf, "symbol": sym, "H": H_label,
                    "n_in": si.get("n", 0), "mean_in": si.get("mean", np.nan),
                    "hit_in": si.get("hit", np.nan),
                    "n_out": so.get("n", 0), "mean_out": so.get("mean", np.nan),
                    "med_out": so.get("med", np.nan),
                    "hit_out": so.get("hit", np.nan),
                    "dd_out": so.get("maxdd", np.nan), "status": status,
                })
        print(f"  [done] {setup} ({tf})")

    res = pd.DataFrame(rows)
    if res.empty:
        print("\n  nothing evaluated: no bars under data/native/ for the "
              "basket. Fetch first (python scripts/fetch_binance_native.py "
              "--symbol ETHUSDT --months 6); nothing written.")
        return 2
    RES.mkdir(parents=True, exist_ok=True)
    res.to_csv(RES / "appendix_b_full.csv", index=False, float_format="%.4f")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n  CELLS BY STATUS:", dict(res["status"].value_counts()))
    passed = res[res["status"].str.startswith("PASS")]
    marg = res[res["status"] == "marginal"].sort_values(
        "mean_out", ascending=False)
    if len(passed):
        print("\n  PASSED cells:")
        for _, r in passed.sort_values("mean_out", ascending=False).iterrows():
            print(f"    {r['symbol']:<9s} {r['tf']:<4s} {r['setup']:<19s} "
                  f"H={r['H']:<3d} OUT net={r['mean_out']:+.3f}% "
                  f"hit={r['hit_out']:.1f}% n={r['n_out']} [{r['status']}] | "
                  f"IN net={r['mean_in']:+.3f}% n={r['n_in']}")
    else:
        print("\n  PASSED cells: none")
    print("\n  top-5 MARGINAL:")
    for _, r in marg.head(5).iterrows():
        print(f"    {r['symbol']:<9s} {r['tf']:<4s} {r['setup']:<19s} "
              f"H={r['H']:<3d} OUT net={r['mean_out']:+.3f}% n={r['n_out']} | "
              f"IN net={r['mean_in']:+.3f}%")
    if marg.empty:
        print("    (none)")

    # per-setup transferability
    print("\n  PER-SETUP TRANSFERABILITY (median mean_out across symbols, "
          "evaluable cells only):")
    ev = res[res["status"].isin(["PASS", "PASS(small-n)", "marginal", "dead"])]
    for (setup, H), g in ev.groupby(["setup", "H"]):
        if len(g) >= 3:
            print(f"    {setup:<19s} H={H:<3d} median={g['mean_out'].median():+.3f}%  "
                  f"pos {int((g['mean_out'] > 0).sum())}/{len(g)}  "
                  f"(n_out sum={int(g['n_out'].sum())})")

    # n-diagnostic: ground every "insufficient" in actual trade counts, and
    # give an UNDERPOWERED READ (diagnostic only, NOT a judged claim) for
    # setups whose counts sit under the pre-registered guard
    print("\n  N-DIAGNOSTIC (why cells are/aren't evaluable; underpowered "
          "read is NOT a verdict):")
    print(f"    {'setup':<19s} {'H':>3s} {'med n_IN':>9s} {'med n_OUT':>10s} "
          f"{'max n_OUT':>10s} {'cells>=guard':>13s} "
          f"{'diag med mean_OUT (n>=20)':>26s}")
    for (setup, H), g in res.groupby(["setup", "H"]):
        min_n = PLAN[setup][2]
        ok_cells = int((g["n_out"] >= min_n).sum())
        g20 = g[g["n_out"] >= 20]
        diag = (f"{g20['mean_out'].median():+.3f}% ({len(g20)} cells)"
                if len(g20) >= 3 else "n<20 everywhere")
        print(f"    {setup:<19s} {H:>3d} {g['n_in'].median():>9.0f} "
              f"{g['n_out'].median():>10.0f} {g['n_out'].max():>10.0f} "
              f"{ok_cells:>10d}/13 {diag:>26s}")

    # ── false-pass ledger ────────────────────────────────────────────────────
    print("\n  FALSE-PASS LEDGER:")
    print(f"    this wave: {n_eval} evaluable cells, E[false] ~ {e_false:.2f} "
          f"(normal approx; honest — non-overlapping trades)")
    print(f"    campaign cumulative: {PRIOR_CELLS + len(res)} unique cells "
          f"with a CSV in results/ (+{PRIOR_CELLS_NO_CSV} generic-probe "
          f"cells, console only), E[false] ~ {PRIOR_EFALSE + e_false:.2f} "
          f"(prior waves' E recorded at run time, not reproducible from "
          f"results/; per-bar cells' E is an underestimate)")

    # ── verdict ──────────────────────────────────────────────────────────────
    n_pass = len(passed)
    print("\n" + "=" * 100)
    print("  VERDICT — APPENDIX B (final single-asset wave)")
    print("=" * 100)
    if n_pass == 0:
        print("  0 cells passed. The final wave found nothing the earlier "
              "waves missed.")
        print("  Single-asset directional setups on this dataset are now "
              "CLOSED by pre-registration —")
        print("  no further waves regardless of ideas; next steps require new "
              "strategy classes or new data.")
    else:
        print(f"  {n_pass} cell(s) passed vs ~{e_false:.2f} expected false "
              f"this wave (cumulative ledger above).")
        print("  Any survivor must clear the S02-style confirmation protocol "
              "(shifted splits + untouched data)")
        print("  BEFORE being called edge — and by declaration this dataset "
              "offers no further waves after that.")
    print(f"\n  full table -> {RES / 'appendix_b_full.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
