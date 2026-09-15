"""
overlap.py — one-position-at-a-time resolution (the S02 lesson, in code).

Per symbol, trades are taken in entry_time order; any trade whose entry falls
while a previous trade on the same symbol is still open is DROPPED.  All five
checks run on the resolved set — dense overlapping logs are exactly how
per-trade statistics get inflated, so the judge never looks at the raw set.
"""

from __future__ import annotations

import pandas as pd


def resolve_overlaps(trades: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Returns (non-overlapping trades, n_dropped).  Stable sort on
    entry_time only: on tied entries the FIRST row of the input wins,
    exactly as documented — not the shortest trade."""
    keep_idx: list[int] = []
    for _, g in trades.groupby("symbol", sort=False):
        g = g.sort_values("entry_time", kind="stable")
        open_until = None
        for idx, row in g.iterrows():
            if open_until is not None and row["entry_time"] < open_until:
                continue
            keep_idx.append(idx)
            open_until = row["exit_time"]
    kept = trades.loc[sorted(keep_idx)].reset_index(drop=True)
    return kept, len(trades) - len(kept)
