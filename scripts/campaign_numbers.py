#!/usr/bin/env python3
"""
campaign_numbers.py - every campaign number the README quotes that can be
reproduced from results/, printed by one command.  No network, no data/.

    python scripts/campaign_numbers.py

Sections
  A  cells                    delegated to count_cells.py
  B  passes per wave          every cell with status PASS, listed
  C  S02 non-overlap judge    criteria (a)-(d) per H from the trade logs
  D  S20 relative value       H1 and extended, per config x half
  E  S21-S23 structural       status per setup, the evaluable S22 cells
  F  funding / basis carry    portfolio OUT net %/yr, best symbol, $/month
  G  extended window          months covered, pooled OUT expectancy + CI
  H  false-pass proxy         see below
  I  NOT reproducible         README numbers that exist only in a probe's
                              console output

H - the false-pass proxy.  The probes printed E[false passes | zero edge]
at run time from each cell's own per-trade sigma; that sigma was never
written to a CSV, so the recorded ledgers (~1.25 for the setup wave,
~1.4 cumulative) cannot be recomputed here.  What CAN be computed is a
proxy with stated assumptions:
  * per-trade sigma per symbol is calibrated on the S02 non-overlap trade
    logs (results/s02_trades_<SYM>_{8,16}.csv) and scaled by sqrt(hours
    held), i.e. sd(trade) = k_sym * sqrt(hold_h);
  * a cell passes by luck with probability
        P(gross_OUT > 0.33%) * P(gross_IN > 0.11%)
    under a zero-edge normal null with sd = sd(trade)/sqrt(n) - the same
    form the probes used;
  * S01-S13 cells count overlapping per-bar signals, so their n overstates
    independent trades and the proxy UNDERestimates E[false] (the probes
    printed the same caveat).  The sensitivity rows divide n by 2 and by
    3.9, the campaign's measured mean stretch-episode length in bars.
The proxy reproduces the H1 ledger's order of magnitude (2.4 vs the
recorded 1.25 at nominal n).  It is a proxy, and the README says so.
"""

from __future__ import annotations

import glob
import sys
from math import erf, exp, factorial, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_cells  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

RES = Path(__file__).resolve().parent.parent / "results"
PASS_NET_OUT = 0.22          # pre-registered: net OUT > 2x taker round trip
TAKER_RT = 0.11
GROSS_OUT_BAR = PASS_NET_OUT + TAKER_RT      # 0.33
GROSS_IN_BAR = TAKER_RT                      # net IN > 0
EPISODE_BARS = 3.9           # measured mean stretch episode (README rule 3)
EVALUABLE = ("dead", "marginal", "PASS", "PASS(small-n)")


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _pois_tail(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return 1.0 - sum(exp(-lam) * lam ** i / factorial(i) for i in range(k))


def _read(name: str) -> pd.DataFrame:
    d = pd.read_csv(RES / name)
    if "H" in d.columns:
        d["H"] = pd.to_numeric(d["H"], errors="coerce")
    return d


def _directional(d: pd.DataFrame) -> pd.DataFrame:
    """Unify net/mean column names across setup_probe and appendix_b."""
    d = d.copy()
    for a, b in (("net_in", "mean_in"), ("net_out", "mean_out")):
        if a not in d.columns:
            d[a] = np.nan
        if b in d.columns:
            d[a] = d[a].fillna(d[b])
    return d


def h(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"  {title}")
    print("=" * 96)


# ── B. passes ────────────────────────────────────────────────────────────────

def passes(label: str, d: pd.DataFrame) -> None:
    d = _directional(d)
    st = d["status"].value_counts().to_dict()
    ev = int(d["status"].isin(EVALUABLE).sum())
    p = d[d["status"].str.startswith("PASS")]
    print(f"\n  {label}: {len(d)} cells, {ev} evaluable, status {st}")
    if len(p):
        print(f"    PASS cells ({len(p)}):")
        cols = ["setup", "symbol", "tf", "H", "n_in", "net_in", "n_out",
                "net_out", "hit_out"]
        for _, r in p.sort_values(["setup", "symbol", "H"]).iterrows():
            print(f"      {r['setup']:<20s} {r['symbol']:<9s} {r['tf']:>3s} "
                  f"H={int(r['H']):<3d} n_in={int(r['n_in']):5d} "
                  f"net_in={r['net_in']:+.3f}  n_out={int(r['n_out']):5d} "
                  f"net_out={r['net_out']:+.3f}  hit_out={r['hit_out']:.1f}%")


# ── H. false-pass proxy ──────────────────────────────────────────────────────

def calibrate_sigma() -> dict[str, float]:
    """k_sym: per-trade sd in % per sqrt(hour held), from the S02 logs."""
    out: dict[str, float] = {}
    syms = sorted({Path(f).stem.split("_")[2]
                   for f in glob.glob(str(RES / "s02_trades_*_*.csv"))})
    for sym in syms:
        ests = []
        for H in (8, 16):
            fp = RES / f"s02_trades_{sym}_{H}.csv"
            if fp.is_file():
                t = pd.read_csv(fp)
                if len(t) > 2:
                    ests.append(t["net_pnl_pct"].std(ddof=1) / sqrt(H))
        if ests:
            out[sym] = float(np.mean(ests))
    return out


def efalse(d: pd.DataFrame, k: dict[str, float], n_div: float = 1.0
           ) -> tuple[int, int, float]:
    """(evaluable, passes, E[false]) under the stated null."""
    d = _directional(d)
    ev = d[d["status"].isin(EVALUABLE)]
    hold_h = ev["H"].astype(float) * np.where(ev["tf"] == "15m", 0.25, 1.0)
    kk = ev["symbol"].map(k).fillna(float(np.median(list(k.values()))))
    sd_out = kk * np.sqrt(hold_h) / np.sqrt(ev["n_out"].astype(float) / n_div)
    sd_in = kk * np.sqrt(hold_h) / np.sqrt(ev["n_in"].astype(float) / n_div)
    p = [(1 - _phi(GROSS_OUT_BAR / a)) * (1 - _phi(GROSS_IN_BAR / b))
         for a, b in zip(sd_out, sd_in)]
    return len(ev), int(ev["status"].str.startswith("PASS").sum()), float(sum(p))


def main() -> int:
    h("A. CELLS  (scripts/count_cells.py)")
    count_cells.main()

    sp = _read("setup_probe_full.csv")
    su = _read("setup_probe_untouched.csv")
    ab = _read("appendix_b_full.csv")
    ew = _read("extended_window_full.csv")
    e13 = ew[ew["wave"] == "S01-S13 directional"]
    e19 = ew[ew["wave"] == "S14-S19 directional"]

    h("B. PASSES PER WAVE  (pre-registered per-cell bar: net_OUT > +0.22%, "
      "net_IN > 0, n >= 100 both halves)")
    passes("H1 S01-S13 setup library (10 symbols)", sp)
    passes("H1 S01-S13 untouched-symbol confirmation (ADA/XRP/DOT)", su)
    print("    the 3 setups that survived screening, on the untouched symbols "
          "(1h H=16, where they passed):")
    surv = su[su["setup"].str.startswith(("S01", "S02", "S08"))
              & (su["tf"] == "1h") & (su["H"] == 16)]
    for _, r in surv.sort_values(["setup", "symbol"]).iterrows():
        print(f"      {r['setup']:<20s} {r['symbol']:<9s} n_in={int(r['n_in']):4d} "
              f"net_in={r['net_in']:+.3f}  n_out={int(r['n_out']):4d} "
              f"net_out={r['net_out']:+.3f}  {r['status']}")
    passes("H1 S14-S19 appendix B", ab)
    print("    per setup (n-guard is 100 OUT trades; S14 allowed 50):")
    for setup, g in ab.groupby("setup", sort=True):
        ev = g[g["status"] != "insufficient"]
        pos = int((pd.to_numeric(ev["mean_out"], errors="coerce") > 0).sum())
        print(f"      {setup:<20s} cells={len(g):2d}  max n_OUT={int(g['n_out'].max()):3d}  "
              f"evaluable={len(ev):2d}  net-positive OUT={pos}")
    passes("EXT S01-S13 (13 symbols, 18.2 months)", e13)
    passes("EXT S14-S19 (18.2 months)", e19)
    e_pass = e13[e13["status"] == "PASS"]
    print(f"\n  EXT directional passes by setup: "
          f"{e_pass['setup'].value_counts().to_dict()}")
    print(f"  EXT directional passes by tf/H:  "
          f"{e_pass.groupby(['tf', 'H']).size().to_dict()}")

    h("C. S02 NON-OVERLAP JUDGE, H1  (from results/s02_trades_<SYM>_<H>.csv)")
    print("  confirm iff (a) median over symbols of mean net_OUT > +0.11%  "
          "(b) >= 8/13 symbols net_OUT > 0\n"
          "              (c) total n_OUT >= 150  (d) sign of median net_IN "
          "== sign of median net_OUT")
    for H in (8, 16):
        rows = []
        for f in sorted(glob.glob(str(RES / f"s02_trades_*_{H}.csv"))):
            t = pd.read_csv(f)
            i, o = t[t["half"] == "IN"]["net_pnl_pct"], t[t["half"] == "OUT"]["net_pnl_pct"]
            rows.append((i.mean(), o.mean(), len(o)))
        r = np.array(rows)
        a, b, c = np.median(r[:, 1]), int((r[:, 1] > 0).sum()), int(r[:, 2].sum())
        d_in = np.median(r[:, 0])
        print(f"  H={H:<3d} symbols={len(r):2d}  (a) median net_OUT {a:+.3f}% "
              f"-> {'PASS' if a > 0.11 else 'FAIL'}   (b) {b}/13 -> "
              f"{'PASS' if b >= 8 else 'FAIL'}   (c) n_OUT {c} -> "
              f"{'PASS' if c >= 150 else 'FAIL'}   (d) median net_IN {d_in:+.3f}% "
              f"vs OUT {a:+.3f}% -> {'PASS' if np.sign(d_in) == np.sign(a) else 'FAIL'}")

    h("D. S20 RELATIVE VALUE  (bar: mean net_OUT > +0.44%/cycle, sign IN == "
      "sign OUT, >= 50 OUT cycles)")
    for label, d in (("H1", _read("s20_summary.csv")),
                     ("EXT", ew[ew["wave"] == "S20 structural"])):
        print(f"  {label}:")
        for cfg, g in d.groupby("config", sort=False):
            i = g[g["half"] == "IN"].iloc[0]
            o = g[g["half"] == "OUT"].iloc[0]
            print(f"    {cfg:<13s} IN  n={int(i['cycles']):3d} net {i['net_mean']:+.4f}%   "
                  f"OUT n={int(o['cycles']):3d} net {o['net_mean']:+.4f}%   "
                  f"sign {'same' if np.sign(i['net_mean']) == np.sign(o['net_mean']) else 'FLIP'}")

    h("E. S21-S23 STRUCTURAL, extended window  (appendix C)")
    e23 = ew[ew["wave"] == "S21-S23 structural"]
    print("  status by setup:", e23.groupby(["setup", "status"]).size().to_dict())
    ev = e23[e23["status"] != "insufficient"]
    print("  evaluable cells:")
    for _, r in ev.iterrows():
        print(f"    {r['setup']:<18s} {r['key']:<8s} {r['param']:<6s} "
              f"n_in={int(r['n_in']):2d} net_in={r['net_in']:+.3f}  "
              f"n_out={int(r['n_out']):3d} net_out={r['net_out']:+.3f}  {r['status']}")

    h("F. FUNDING / BASIS CARRY, H1  (bar: OUT net >= +8%/yr on deployed, at taker)")
    fp = _read("funding_probe_full.csv")
    port = fp[(fp["scope"] == "PORTFOLIO") & (fp["half"] == "OUT")]
    for _, r in port.sort_values(["strategy", "fee"]).iterrows():
        usd = r["ann_dep_pct"] / 100 * 10_000 / 12
        print(f"  {r['strategy']:<18s} {r['fee']:<6s} OUT net {r['ann_dep_pct']:+7.2f}%/yr "
              f"on deployed  = {usd:+6.1f} $/month on $10k")
    sym = fp[(fp["scope"] != "PORTFOLIO") & (fp["half"] == "OUT")]
    best = sym.sort_values("ann_dep_pct", ascending=False).iloc[0]
    print(f"  best single symbol OUT: {best['strategy']} {best['scope']} "
          f"{best['fee']} {best['ann_dep_pct']:+.2f}%/yr")

    h("G. EXTENDED WINDOW")
    dt = _read("extended_window_data_table.csv")
    print(f"  data table: {len(dt)} symbol x tf series, {dt['months'].min():.1f}-"
          f"{dt['months'].max():.1f} months, {dt['start'].min()} -> {dt['end'].max()}")
    print("  pooled OUT expectancy over evaluable directional cells, 95% CI "
          "(mean +- 1.96 sd/sqrt(n)):")
    pooled = []
    for label, d in (("S01-S13", e13), ("S14-S19", e19)):
        v = _directional(d)
        v = v[v["status"].isin(EVALUABLE)]["net_out"].dropna().to_numpy()
        m, ci = v.mean(), 1.96 * v.std() / sqrt(len(v))
        pooled.append(v)
        print(f"    {label:<8s} n={len(v):4d}  {m:+.3f}%  [{m - ci:+.3f}, {m + ci:+.3f}]")
    v = np.concatenate(pooled)
    m, ci = v.mean(), 1.96 * v.std() / sqrt(len(v))
    print(f"    {'S01-S19':<8s} n={len(v):4d}  {m:+.3f}%  [{m - ci:+.3f}, {m + ci:+.3f}]")

    h("H. FALSE-PASS PROXY  (assumptions in the module docstring; NOT the "
      "ledger the probes printed)")
    k = calibrate_sigma()
    print(f"  sigma per sqrt(hour held), % (from S02 logs, {len(k)} symbols): "
          f"median {np.median(list(k.values())):.3f}, "
          f"min {min(k.values()):.3f}, max {max(k.values()):.3f}")
    print(f"\n  {'wave':<38s} {'n_eff':>7s} {'eval':>5s} {'PASS':>5s} "
          f"{'E[false]':>9s} {'P(X>=PASS)':>11s}")
    waves = (("H1 S01-S13 setup library", sp, True),
             ("H1 S01-S13 untouched", su, True),
             ("H1 S14-S19 appendix B (non-overlap)", ab, False),
             ("EXT S01-S13", e13, True),
             ("EXT S14-S19 (non-overlap)", e19, False))
    for label, d, overlapping in waves:
        divs = (1.0, 2.0, EPISODE_BARS) if overlapping else (1.0,)
        for div in divs:
            n_ev, n_pass, e = efalse(d, k, div)
            tag = "n" if div == 1.0 else f"n/{div:g}"
            print(f"  {label:<38s} {tag:>7s} {n_ev:5d} {n_pass:5d} {e:9.2f} "
                  f"{_pois_tail(n_pass, e):11.3f}")
    print("\n  Reading: at nominal n the S01-S13 passes look like an excess; "
          "at the campaign's own\n  measured episode length they sit below "
          "the noise floor.  A per-bar screen cannot resolve\n  this - only "
          "the non-overlap judge can, and it was not run on the extended "
          "window.\n  Recorded run-time ledgers (not reproducible here): "
          "setup wave ~1.25, generic ~0.10, S02 ~0.05,\n  H1 cumulative ~1.4.")

    h("I. README NUMBERS THAT ARE NOT REPRODUCIBLE FROM results/  (console "
      "output only)")
    for line in (
        "RL diagnostics (1m): entry win-rate 7-19pp below the RW-neutral benchmark",
        "offline 1m price probe: expectancy ~ -0.11%/trade on 284,550 entries",
        "generic 15m/1h probe: ~90 cells, 0 past +0.22% (native_tf_edge_probe.py prints only)",
        "run-time false-pass ledgers: ~1.25 (setup wave), ~1.4 (H1 cumulative)",
        "per-bar best cell +0.63% before the non-overlap judge",
        "stretch-episode length 3.9 bars (printed by s02_nonoverlap_backtest.py from data/)",
        "funding: 39-56% of payments negative; basket average funding ~ 0",
        "extended split regime: IN drift -11% / +0.35bp -> OUT drift -71% / -0.19bp",
    ):
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
