#!/usr/bin/env python3
"""
count_cells.py - reproducible cell count for the campaign, from results/ only.

    python scripts/count_cells.py          # no arguments, deterministic

WHAT IS A CELL
  One hypothesis judged once against its pre-registered threshold on one
  instrument / timeframe / horizon.  Concretely, the natural key per class:
    directional  S01-S19 : (symbol, tf, setup, H)
    relative value  S20  : (config)              ranker x k, IN/OUT are halves
    statarb   S21-S23    : (setup, key, tf, param)   key = pair or symbol
    funding carry  S-A/B/C: (strategy, scope)  scope = symbol; S-C exists only
                            at PORTFOLIO level and is counted as one cell
  A CSV row is NOT a cell: s20_summary and funding_probe_full carry one row
  per (cell, half[, fee]); s02_trades_*, s20_cycles_*, *_equity.csv carry
  trades / cycles / equity points, not cells.

WHAT IS DEDUPLICATED
  The same key evaluated again on another cut of the data is the same cell:
    setup_probe_split40 / split60   = the 780 setup_probe_full cells re-split
    s02_trades_<sym>_<H>            = 26 S02 cells re-executed one-position-
                                      at-a-time (all 26 keys already exist)
    extended_window_full S01-S13 / S14-S19 / S20 = the H1-2026 cells re-run
                                      on the 18-month window (0 new keys)
  Such re-runs are reported as EVALUATIONS, never as new cells.

WHAT IS EXCLUDED
  - PORTFOLIO rows of S-A / S-B in funding_probe_full (aggregates of the
    per-symbol cells)
  - tests/fixtures/synthetic_KNOWN_FAKE_do_not_cite.csv (a fabricated trade
    log used to test the judge; it never lived in a campaign wave)
  - extended_window_data_table.csv (data coverage, not hypotheses)
  - the generic 15m/1h signal-family probe (native_tf_edge_probe.py) - it
    prints its cells and writes no CSV, so its ~90 cells are NOT reproducible
    from results/ and are listed separately as "not artifact-backed"
  - appendix D-G waves: not published in this repository

THREE NUMBERS PRINTED
  A  unique cells        distinct keys across every file (the honest "how
                         many hypotheses were screened")
  B  evaluations         every judgement of a cell, re-splits and re-runs
                         included (what the CSV rows add up to)
  C  H1-2026 unique      A restricted to the six-month campaign (no extended
                         window)
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

RES = Path(__file__).resolve().parent.parent / "results"
DIR_KEY = ["symbol", "tf", "setup", "H"]


def _read(name: str) -> pd.DataFrame | None:
    fp = RES / name
    if not fp.is_file():
        return None
    d = pd.read_csv(fp)
    if "H" in d.columns:
        d["H"] = pd.to_numeric(d["H"], errors="coerce").astype("Int64")
    return d


def _dir_keys(d: pd.DataFrame) -> set[tuple]:
    return {("dir", *map(str, r)) for r in d[DIR_KEY].itertuples(index=False)}


def waves() -> list[dict]:
    """Ordered list of (wave, source, rows, keys); order defines 'new'."""
    out: list[dict] = []

    def add(wave, source, rows, keys, note=""):
        out.append(dict(wave=wave, source=source, rows=rows, keys=keys,
                        note=note))

    # ── H1-2026 campaign ────────────────────────────────────────────────
    d = _read("setup_probe_full.csv")
    if d is not None:
        add("H1 S01-S13 setup library", "setup_probe_full.csv", len(d),
            _dir_keys(d))
    d = _read("setup_probe_untouched.csv")
    if d is not None:
        add("H1 S01-S13 confirmation (untouched symbols)",
            "setup_probe_untouched.csv", len(d), _dir_keys(d))
    for f in ("setup_probe_split40.csv", "setup_probe_split60.csv"):
        d = _read(f)
        if d is not None:
            add("H1 S01-S13 re-split", f, len(d), _dir_keys(d),
                "same cells, different IN/OUT cut")
    files = sorted(glob.glob(str(RES / "s02_trades_*.csv")))
    if files:
        keys = set()
        for f in files:
            sym, h = os.path.basename(f)[len("s02_trades_"):-4].rsplit("_", 1)
            keys.add(("dir", sym, "1h", "S02 vwap-fade", h))
        add("H1 S02 non-overlap judge", "s02_trades_*.csv (26 files)",
            len(files), keys, "one file = one cell; rows are trades")
    d = _read("appendix_b_full.csv")
    if d is not None:
        add("H1 S14-S19 appendix B", "appendix_b_full.csv", len(d),
            _dir_keys(d))
    d = _read("s20_summary.csv")
    if d is not None:
        add("H1 S20 relative value", "s20_summary.csv", len(d),
            {("s20", c) for c in d["config"]}, "rows = config x half")
    d = _read("funding_probe_full.csv")
    if d is not None:
        sym = d[d["scope"] != "PORTFOLIO"]
        keys = {("fund", s, sc) for s, sc in
                sym[["strategy", "scope"]].itertuples(index=False)}
        port_only = set(d.loc[d["scope"] == "PORTFOLIO", "strategy"]) - set(
            sym["strategy"])
        keys |= {("fund", s, "PORTFOLIO") for s in port_only}
        add("H1 funding / basis carry", "funding_probe_full.csv", len(d),
            keys, "rows = cell x half x fee; S-A/S-B PORTFOLIO rows excluded")

    # ── extended window (18 months) ─────────────────────────────────────
    d = _read("extended_window_full.csv")
    if d is not None:
        for wave, grp in d.groupby("wave", sort=False):
            src = f"extended_window_full.csv [{wave}]"
            if wave.startswith("S20"):
                keys = {("s20", c) for c in grp["config"]}
            elif wave.startswith("S21"):
                keys = {("statarb", *map(str, r)) for r in
                        grp[["setup", "key", "tf", "param"]]
                        .itertuples(index=False)}
            else:
                keys = _dir_keys(grp)
            add(f"EXT {wave}", src, len(grp), keys)
    return out


def main() -> int:
    seen: set[tuple] = set()
    rows = []
    for w in waves():
        new = w["keys"] - seen
        seen |= w["keys"]
        evaluations = len(w["keys"])
        rows.append((w["wave"], w["source"], w["rows"], evaluations,
                     len(new), w["note"]))

    print("CAMPAIGN CELL COUNT  (source: results/)")
    print()
    hdr = f"{'wave':<48s}{'rows':>6s}{'evals':>7s}{'new':>6s}  source / note"
    print(hdr)
    print("-" * len(hdr))
    for wave, src, n, ev, new, note in rows:
        line = f"{wave:<48s}{n:>6d}{ev:>7d}{new:>6d}  {src}"
        print(line)
        if note:
            print(" " * 69 + note)
    tot_eval = sum(r[3] for r in rows)
    tot_new = sum(r[4] for r in rows)
    h1_new = sum(r[4] for r in rows if r[0].startswith("H1"))
    print("-" * len(hdr))
    print(f"{'A  unique cells (all files, deduplicated)':<48s}{'':>6s}{'':>7s}"
          f"{tot_new:>6d}")
    print(f"{'B  evaluations (re-splits and re-runs included)':<48s}{'':>6s}"
          f"{tot_eval:>7d}")
    print(f"{'C  H1-2026 unique cells (no extended window)':<48s}{'':>6s}"
          f"{'':>7s}{h1_new:>6d}")
    print()
    print("not artifact-backed (no CSV in results/): generic 15m/1h "
          "signal-family probe, ~90 cells (native_tf_edge_probe.py prints "
          "only); appendix D-G: not published here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
