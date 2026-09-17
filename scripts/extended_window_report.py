#!/usr/bin/env python3
"""
extended_window_report.py — EXTENDED-WINDOW (15-18mo) campaign re-run.

Reuses the existing engines unchanged (setup_probe, appendix_b_probe,
s20_relative_value, appendix_c_statarb) — this script only (1) prints the
heterogeneous data table, (2) characterizes the IN/OUT regime split, (3)
orchestrates the probes on the now-longer window, (4) merges their result
CSVs into results/extended_window_full.csv, and (5) prints a class verdict
against the SAME pre-registered thresholds as the 6-month campaign.

PRE-REGISTRATION (printed in the verdict header):
  * directional (S01-S19): death CONFIRMS harder (tighter CI around -fee),
    does NOT flip. A flip = hunt for look-ahead / a bug, not an edge.
  * structural (S20/S22/S23): FIRST TIME they get a sample for a verdict —
    only here is a real PASS/FAIL possible instead of insufficient.
  * thresholds are UNCHANGED; not relaxed for the new sample.
  * cumulative E[false] over the whole campaign + this run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "native"
RES = ROOT / "results"
SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
           "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
           "ADAUSDT", "XRPUSDT", "DOTUSDT"]
TF_MS = {"15m": 900_000, "1h": 3_600_000}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — data table (heterogeneous window, no silent truncation)
# ─────────────────────────────────────────────────────────────────────────────

def data_table() -> pd.DataFrame:
    print("=" * 96)
    print("  STAGE 1 — DATA (extended window; symbols listed later are SHORTER "
          "— flagged, not truncated)")
    print("=" * 96)
    print(f"  {'symbol':<9s} {'tf':<4s} {'start':<11s} {'end':<11s} "
          f"{'n_bars':>7s} {'months':>7s} {'gaps%':>6s} {'status':>10s}")
    rows = []
    for sym in SYMBOLS:
        for tf in ("15m", "1h"):
            fp = DATA / sym / tf / "bars.parquet"
            if not fp.is_file():
                print(f"  {sym:<9s} {tf:<4s} {'MISSING':<11s}")
                rows.append({"symbol": sym, "tf": tf, "status": "missing"})
                continue
            df = pd.read_parquet(fp)
            t = df["open_time"].values.astype(np.int64)
            start = pd.to_datetime(t[0], unit="ms", utc=True)
            end = pd.to_datetime(t[-1], unit="ms", utc=True)
            months = (t[-1] - t[0]) / (30.4 * 86400e3)
            expected = int((t[-1] - t[0]) // TF_MS[tf]) + 1
            gaps = 100.0 * (expected - len(df)) / max(expected, 1)
            status = "ok" if gaps <= 5.0 else "EXCLUDE(gaps)"
            print(f"  {sym:<9s} {tf:<4s} {start:%Y-%m-%d}  {end:%Y-%m-%d}  "
                  f"{len(df):>7d} {months:>6.1f}m {gaps:>5.1f}% {status:>10s}")
            rows.append({"symbol": sym, "tf": tf,
                         "start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}",
                         "n_bars": len(df), "months": round(months, 1),
                         "gaps_pct": round(gaps, 2), "status": status})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(RES / "extended_window_data_table.csv", index=False)
    spans = tbl[tbl.status == "ok"]["months"]
    if len(spans):
        print(f"\n  window span: {spans.min():.1f}-{spans.max():.1f} months "
              f"(heterogeneous — later listings shorter, kept at true length)")
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b — regime characterization of the 50/50 IN vs OUT split
# ─────────────────────────────────────────────────────────────────────────────

def regime_split() -> None:
    print("\n" + "=" * 96)
    print("  STAGE 1b — REGIME of the 50/50 walk-forward split (proves IN/OUT "
          "cross different regimes)")
    print("=" * 96)
    print(f"  {'symbol':<9s} | {'IN drift%':>9s} {'IN fund(bp/8h)':>14s} "
          f"{'IN vol%':>8s} | {'OUT drift%':>10s} {'OUT fund':>9s} "
          f"{'OUT vol%':>9s} | split_date")
    agg = []
    for sym in SYMBOLS:
        fp = DATA / sym / "1h" / "bars.parquet"
        if not fp.is_file():
            continue
        df = pd.read_parquet(fp)
        n = len(df)
        mid = n // 2
        c = df["close"].values
        f = df["funding"].values
        t = df["open_time"].values.astype(np.int64)
        ev = ((t // TF_MS["1h"]) % 8) == 0
        lr = np.diff(np.log(np.clip(c, 1e-9, None)))
        split = pd.to_datetime(t[mid], unit="ms", utc=True)

        def half(lo, hi):
            drift = 100.0 * (c[hi - 1] / c[lo] - 1.0)
            fund = np.nanmean(f[lo:hi][ev[lo:hi]]) * 1e4
            vol = np.nanstd(lr[lo:hi - 1]) * np.sqrt(24 * 365) * 100
            return drift, fund, vol
        di, fi, vi = half(0, mid)
        do, fo, vo = half(mid, n)
        agg.append((sym, di, fi, vi, do, fo, vo))
        print(f"  {sym:<9s} | {di:>+8.1f} {fi:>+13.2f} {vi:>7.0f} | "
              f"{do:>+9.1f} {fo:>+8.2f} {vo:>8.0f} | {split:%Y-%m-%d}")
    if agg:
        a = np.array([r[1:] for r in agg], float)
        m = np.nanmean(a, axis=0)
        print(f"  {'MEAN':<9s} | {m[0]:>+8.1f} {m[1]:>+13.2f} {m[2]:>7.0f} | "
              f"{m[3]:>+9.1f} {m[4]:>+8.2f} {m[5]:>8.0f} |")
        print(f"  -> IN mean funding {m[1]:+.2f} bp/8h vs OUT {m[4]:+.2f}; "
              f"IN drift {m[0]:+.0f}% vs OUT {m[3]:+.0f}% — "
              f"{'DIFFERENT regimes' if abs(m[0]-m[3])>10 or abs(m[1]-m[4])>0.3 else 'similar'}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — run the existing engines on the new window
# ─────────────────────────────────────────────────────────────────────────────

def run_probe(name: str, args: list[str]) -> str:
    print(f"\n  >>> running {name} ...")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / name)] + args,
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=ROOT, env={**__import__("os").environ,
                                      "PYTHONIOENCODING": "utf-8"})
    tail = "\n".join((r.stdout or "").splitlines()[-6:])
    print(tail)
    if r.returncode not in (0, 1):
        print(f"  [warn] {name} exit {r.returncode}\n{(r.stderr or '')[-500:]}")
    return r.stdout or ""


def merge_full() -> pd.DataFrame:
    """Merge each engine's per-cell CSV into one tagged table."""
    parts = []
    src = {
        "setup_probe_full.csv": ("S01-S13 directional", ["setup", "symbol",
            "tf", "H", "n_out", "net_out", "hit_out", "status"]),
        "appendix_b_full.csv": ("S14-S19 directional", ["setup", "symbol",
            "tf", "H", "n_out", "mean_out", "hit_out", "status"]),
        "appendix_c_full.csv": ("S21-S23 structural", ["setup", "key", "tf",
            "param", "n_out", "net_out", "hit_out", "status"]),
        "s20_summary.csv": ("S20 structural", ["config", "half", "cycles",
            "net_mean", "hit", "ann_dep"]),
    }
    for fn, (wave, _) in src.items():
        fp = RES / fn
        if fp.is_file():
            d = pd.read_csv(fp)
            d.insert(0, "wave", wave)
            d.insert(1, "source", fn)
            parts.append(d)
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    full.to_csv(RES / "extended_window_full.csv", index=False)
    return full


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — verdict
# ─────────────────────────────────────────────────────────────────────────────

def directional_ci() -> None:
    """Pooled OUT expectancy + 95% CI for directional cells, new vs 6-mo."""
    print("\n  DIRECTIONAL death check (pooled OUT net expectancy ± 95% CI):")
    for label, fn, col in (("S01-S13", "setup_probe_full.csv", "net_out"),
                           ("S14-S19", "appendix_b_full.csv", "mean_out")):
        fp = RES / fn
        if not fp.is_file():
            continue
        d = pd.read_csv(fp)
        d = d[d["status"].isin(["dead", "marginal", "PASS", "PASS(small-n)"])]
        v = pd.to_numeric(d[col], errors="coerce").dropna().values
        if len(v) < 5:
            print(f"    {label}: n={len(v)} cells (too few to pool)")
            continue
        mean = v.mean()
        ci = 1.96 * v.std() / np.sqrt(len(v))
        print(f"    {label}: mean OUT net {mean:+.3f}%/trade  95%CI "
              f"[{mean-ci:+.3f}, {mean+ci:+.3f}]  over {len(v)} evaluable cells "
              f"(bar: >0; ≈-fee = dead)")


def structural_verdict(full: pd.DataFrame) -> None:
    print("\n  STRUCTURAL sample check (did the long window finally reach the "
          "n-guard?):")
    # S22/S23/S21 from appendix_c_full.csv, S20 from s20_summary
    ac = RES / "appendix_c_full.csv"
    if ac.is_file():
        d = pd.read_csv(ac)
        for setup in sorted(d["setup"].unique()):
            s = d[d["setup"] == setup]
            evaluable = int((s["status"] != "insufficient").sum())
            npass = int((s["status"] == "PASS").sum())
            maxn = int(pd.to_numeric(s["n_out"], errors="coerce").max())
            print(f"    {setup:<22s}: {len(s)} cells, max n_out={maxn}, "
                  f"evaluable={evaluable}, PASS={npass}, "
                  f"insufficient={int((s['status']=='insufficient').sum())}")
    s20 = RES / "s20_summary.csv"
    if s20.is_file():
        d = pd.read_csv(s20)
        o = d[d["half"] == "OUT"]
        print(f"    {'S20 relative-value':<22s}: OUT cycles/config="
              f"{list(pd.to_numeric(o['cycles'],errors='coerce').astype(int))}, "
              f"net/cyc={list(o['net_mean'].round(3))}")


def main() -> int:
    RES.mkdir(exist_ok=True)
    print("#" * 96)
    print("#  EXTENDED-WINDOW RE-RUN — PRE-REGISTRATION")
    print("#  directional S01-S19: death CONFIRMS harder (tighter CI @ -fee), "
          "does NOT flip; a flip => bug hunt.")
    print("#  structural S20/S22/S23: FIRST real sample -> real PASS/FAIL "
          "possible (was insufficient).")
    print("#  thresholds UNCHANGED. cumulative E[false] over whole campaign + "
          "this run. PASS on OUT = CANDIDATE,")
    print("#  not edge — needs confirmation on untouched data.")
    print("#" * 96 + "\n")

    data_table()
    regime_split()

    print("\n" + "=" * 96)
    print("  STAGE 2 — ENGINES on the extended window (reused unchanged)")
    print("=" * 96)
    run_probe("setup_probe.py", [])
    run_probe("appendix_b_probe.py", [])
    run_probe("s20_relative_value.py", [])
    run_probe("appendix_c_statarb.py", [])

    full = merge_full()

    print("\n" + "=" * 96)
    print("  STAGE 3 — VERDICT (extended window; thresholds unchanged)")
    print("=" * 96)
    directional_ci()
    structural_verdict(full)
    print(f"\n  merged rows -> {RES / 'extended_window_full.csv'} "
          f"({len(full)} rows; one row per cell x half for S20, per cell "
          f"otherwise - cells are counted by scripts/count_cells.py)")
    print("  Any PASS on OUT above is a CANDIDATE, not an edge — confirm on "
          "untouched symbols/period first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
