"""
checks.py — the pre-registered criteria, v1.1.0.

C1–C5 are FROZEN since v1.0: the values come from the falsification
campaign (2026-07) and are not configurable from the UI or CLI.  C6 was
added in v1.1.0.  The only external input is the fee profile (a property
of the user's exchange).  All checks run on the NON-OVERLAPPING trade set
(one position per symbol at a time) — the S02 lesson.

  C1 SAMPLE     >= 100 non-overlapping trades
  C2 FEES       mean net PnL per trade > 0 after the fee round-trip
  C3 RW-NEUTRAL total net PnL above the 97.5th percentile of 1000
                direction-randomized bootstraps (does choosing the side
                beat a coin flip, fees included)
  C4 STABILITY  50/50 time split: mean net same sign in both halves AND
                second half > 0 (regime artifacts fail here)
  C5 REGIME     of market regimes (UP/DOWN/FLAT by 24h drift at entry,
                +-1.5% threshold) holding >= 10 trades, >= 60% must be
                net-positive, and at least one regime must qualify
  C6 LOOK-AHEAD every entry is booked strictly after its signal_time AND
                outside the signal's own 1h bar (an entry at the open of a
                later bar is clean).  Needs a signal_time column.  Without
                one the check is UNVERIFIABLE — reported as such in the
                verdict line, the summary and every report; never as a pass.

A check has three states: PASS, FAIL, UNVERIFIABLE.  The verdict is FAIL if
any check FAILs; UNVERIFIABLE never turns into a pass by silence — refusal
beats a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from judge.ingest import IngestError
from judge.marketdata import TF_MS, drift_24h_at, ensure_klines, next_bar_open
from judge.overlap import resolve_overlaps

CRITERIA_VERSION = "1.1.0"
MIN_TRADES = 100
BOOT_N = 1000
BOOT_PCTL = 97.5
REGIME_DRIFT = 0.015
REGIME_MIN_TRADES = 10
REGIME_POS_FRAC = 0.60
RNG_SEED = 20260706          # fixed: identical verdict on identical input
LOOKAHEAD_BAR_MS = TF_MS     # "same bar" for C6 is the judge's 1h bar
MAX_ROWS_NAMED = 5           # how many offending CSV lines a detail names

PASS, FAIL, UNVERIFIABLE = "PASS", "FAIL", "UNVERIFIABLE"


@dataclass
class CheckResult:
    code: str
    name: str
    passed: bool
    key_number: str
    detail: str
    status: str = ""            # PASS | FAIL | UNVERIFIABLE

    def __post_init__(self):
        if not self.status:
            self.status = PASS if self.passed else FAIL
        if self.status == UNVERIFIABLE:
            self.passed = False     # never counts as a pass


@dataclass
class AuditResult:
    verdict: str                       # "PASS" | "FAIL"
    checks: list[CheckResult]
    n_trades: int
    n_dropped_overlap: int
    reconstructed: bool
    fee_rt: float
    fee_label: str = ""                # "binance-taker" | "custom 3 bps/side"
    criteria_version: str = CRITERIA_VERSION
    warnings: list[str] = field(default_factory=list)
    # chart artifacts
    trades: pd.DataFrame | None = None          # resolved set with net_pct
    bootstrap_totals: np.ndarray | None = None
    client_total: float = 0.0
    client_pctl: float = 0.0
    half_means: tuple[float, float] = (0.0, 0.0)
    regime_table: pd.DataFrame | None = None

    @property
    def unverifiable(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == UNVERIFIABLE]

    @property
    def evaluated(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status != UNVERIFIABLE]

    @property
    def cost_note(self) -> str:
        """Printed next to C2 everywhere: the fee is an input, not a finding."""
        return (f"costs re-computed at {self.fee_label or 'custom'} "
                f"{self.fee_rt:.3f}% round trip on every trade; this does not "
                "validate the author's own cost assumptions - the same log "
                "can flip C2 under a different profile, and slippage or "
                "funding are not modelled.")


def _rows(tr: pd.DataFrame, mask) -> list[int]:
    """Source CSV line numbers for the rows selected by mask."""
    if "row" in tr.columns:
        return [int(r) for r in tr.loc[mask, "row"]]
    return [int(i) + 1 for i in tr.index[mask]]


def _name_rows(rows: list[int]) -> str:
    shown = ", ".join(str(r) for r in rows[:MAX_ROWS_NAMED])
    more = (f", +{len(rows) - MAX_ROWS_NAMED} more"
            if len(rows) > MAX_ROWS_NAMED else "")
    return f"CSV line{'s' if len(rows) != 1 else ''} {shown}{more}"


def _pct(x: float, fmt: str = "+.3f") -> str:
    """A key number is a number or the word for its absence - never nan."""
    return f"{x:{fmt}}%" if np.isfinite(x) else "n/a"


def check_lookahead(tr: pd.DataFrame) -> CheckResult:
    """C6 — signal_time vs entry_time on the log's own timestamps.

    FAIL   if any signal_time >= entry_time (the decision is logged as made
           when or after the position was open), or if any entry falls
           inside the signal's own 1h bar (it books the move that made the
           signal).  Both kinds are counted and named by CSV line.
    PASS   otherwise.  This verifies declared timestamps only — it cannot
           see the strategy's data feed.
    UNVERIFIABLE if the log has no signal_time (or no evaluable trades)."""
    n = len(tr)
    has_col = "signal_time" in tr.columns and tr["signal_time"].notna().any()
    if n == 0 or not has_col:
        why = "no evaluable trades" if n == 0 else "no signal_time column"
        return CheckResult(
            "C6", "Look-ahead", False, why,
            "The log carries no signal_time (decision time), so the judge "
            "cannot tell whether an entry was booked before its signal could "
            "be known. A look-ahead-biased log passes C1-C5 untouched. Add a "
            "signal_time column (aliases: decision_time, signal_ts, sig_time) "
            "and re-run to make this check evaluable.",
            status=UNVERIFIABLE)
    sig = tr["signal_time"]
    ent = tr["entry_time"]
    missing = sig.isna()
    if missing.any():
        return CheckResult(
            "C6", "Look-ahead", False,
            f"{int(missing.sum())}/{n} trades lack signal_time",
            f"{int(missing.sum())} of {n} trades have no signal_time "
            f"({_name_rows(_rows(tr, missing))}). The check needs a decision "
            "time on every trade; fill them in and re-run.",
            status=UNVERIFIABLE)

    not_before = sig >= ent
    same_bar = (~not_before) & (sig.dt.floor("h") == ent.dt.floor("h"))
    n_nb, n_sb = int(not_before.sum()), int(same_bar.sum())
    if n_nb == 0 and n_sb == 0:
        return CheckResult(
            "C6", "Look-ahead", True, f"0/{n} trades",
            f"All {n} entries are booked after their signal and outside the "
            "signal's 1h bar (an entry at the open of a later bar is clean). "
            "Verified on the log's own timestamps only - the judge cannot see "
            "the strategy's data feed.")
    parts = []
    if n_nb:
        lag = (sig - ent)[not_before]
        w = lag.idxmax()
        mins = lag.loc[w].total_seconds() / 60.0
        w_row = _rows(tr, tr.index == w)[0]
        parts.append(
            f"{n_nb} of {n} trades have signal_time at or after entry_time - "
            f"the decision is logged as made when the position was already "
            f"open (worst: CSV line {w_row}, signal {sig.loc[w]:%Y-%m-%d %H:%M} "
            f"is {mins:+.0f} min relative to entry {ent.loc[w]:%Y-%m-%d %H:%M}; "
            f"{_name_rows(_rows(tr, not_before))})")
    if n_sb:
        parts.append(
            f"{n_sb} of {n} trades enter inside the signal's own 1h bar "
            f"({_name_rows(_rows(tr, same_bar))}) - the entry captures the "
            "bar that produced the signal; a clean entry is at the open of a "
            "later bar")
    return CheckResult(
        "C6", "Look-ahead", False, f"{n_nb + n_sb}/{n} trades",
        "; ".join(parts) + ". Every other number in this report inherits "
        "this bias.")


def audit(trades: pd.DataFrame, fee_rt: float,
          progress=None, fee_label: str = "") -> AuditResult:
    """fee_rt = round-trip fee in % of notional (e.g. 0.11); fee_label names
    the profile it came from so every report says which rate was used."""
    def step(msg):
        if progress:
            progress(msg)

    warnings: list[str] = []

    step("resolving overlaps (one position per symbol at a time)")
    tr, dropped = resolve_overlaps(trades)
    if dropped:
        warnings.append(f"{dropped} overlapping trades dropped "
                        f"(one-position-at-a-time rule).")

    step("loading market data")
    klines: dict[str, pd.DataFrame] = {}
    for sym in tr["symbol"].unique():
        g = tr[tr["symbol"] == sym]
        t0 = int(g["entry_time"].min().timestamp() * 1000)
        t1 = int(g["exit_time"].max().timestamp() * 1000)
        klines[sym] = ensure_klines(sym, t0, t1, progress=progress)

    step("pricing trades")
    reconstructed = False
    n_partial = 0
    n_gapped = 0
    ep = tr["entry_price"].to_numpy(dtype=float, copy=True)
    xp = tr["exit_price"].to_numpy(dtype=float, copy=True)
    for i, row in tr.iterrows():
        e_nan, x_nan = np.isnan(ep[i]), np.isnan(xp[i])
        if not (e_nan or x_nan):
            continue
        # reconstruct ONLY the missing side; a provided price is user data
        # and is never silently overwritten
        kl = klines[row["symbol"]]
        if e_nan:
            ep[i], g = next_bar_open(kl, int(row["entry_time"].timestamp()
                                             * 1000))
            n_gapped += int(g)
        if x_nan:
            xp[i], g = next_bar_open(kl, int(row["exit_time"].timestamp()
                                             * 1000))
            n_gapped += int(g)
        reconstructed = True
        n_partial += int(e_nan != x_nan)
    ok = ~(np.isnan(ep) | np.isnan(xp)) & (ep > 0)
    if (~ok).any():
        warnings.append(f"{int((~ok).sum())} trades could not be priced "
                        f"(outside available market data) and were dropped.")
        tr = tr[ok].reset_index(drop=True)
        ep, xp = ep[ok], xp[ok]
    if reconstructed:
        warnings.append("prices reconstructed from market data "
                        "(next 1h bar open after each timestamp).")
    if n_partial:
        warnings.append(f"{n_partial} rows had a price for one side only — "
                        f"the provided price was kept, only the missing side "
                        f"was reconstructed.")
    if n_gapped:
        warnings.append(f"{n_gapped} reconstructed prices fall more than one "
                        f"bar after their timestamp (hole in market data) — "
                        f"treat those trades' pricing as approximate.")

    side = tr["side"].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        gross = side * (xp / ep - 1.0) * 100.0
    # Refusal beats a guess: a return that is not a finite number (the
    # exit/entry ratio overflowed) would turn every statistic below into
    # nan or inf and still print as a verdict.  Name the rows and stop.
    bad = ~np.isfinite(gross)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise IngestError(
            f"{int(bad.sum())} trade{'s' if bad.sum() != 1 else ''} "
            f"({_name_rows(_rows(tr, bad))}) "
            f"{'have' if bad.sum() != 1 else 'has'} a return that is not a "
            f"finite number - the exit/entry price ratio overflows (e.g. "
            f"entry {ep[i]:g}, exit {xp[i]:g}). No verdict is computed on "
            "such a log; check the price columns of those rows.")
    net = gross - fee_rt
    tr = tr.assign(gross_pct=gross, net_pct=net,
                   entry_px_used=ep, exit_px_used=xp)

    checks: list[CheckResult] = []
    n = len(tr)

    step("check 1/6: sample size")
    c1 = n >= MIN_TRADES
    checks.append(CheckResult(
        "C1", "Sample size", c1, f"{n} trades",
        f"{n} non-overlapping trades (bar: >= {MIN_TRADES}). "
        + ("" if c1 else "Too few trades for any of the following numbers "
                         "to be trusted.")))

    step("check 2/6: fee survival")
    mean_net = float(net.mean()) if n else float("nan")
    c2 = n > 0 and mean_net > 0
    checks.append(CheckResult(
        "C2", "Fee survival", c2,
        f"{_pct(mean_net)}/trade" if n else "no trades",
        f"Mean net PnL {_pct(mean_net)}/trade after a {fee_rt:.3f}% "
        f"round-trip fee [{fee_label or 'custom rate'}] (bar: > 0). The fee "
        "is re-applied to every trade at this rate; any PnL column in the "
        "log is ignored."))

    step("check 3/6: random-walk neutrality (1000 bootstraps)")
    rng = np.random.default_rng(RNG_SEED)
    signs = rng.choice([-1.0, 1.0], size=(BOOT_N, n))
    boot_totals = (signs * gross[None, :] - fee_rt).sum(axis=1)
    client_total = float(net.sum())
    pctl = float((boot_totals < client_total).mean() * 100.0)
    c3 = pctl >= BOOT_PCTL
    checks.append(CheckResult(
        "C3", "Beats coin-flip", c3, f"{pctl:.1f}th pctl",
        f"Total net {client_total:+.2f}% sits at the {pctl:.1f}th percentile "
        f"of {BOOT_N} direction-randomized logs (bar: >= {BOOT_PCTL}). "
        + ("" if c3 else "Flipping a coin for direction does as well.")))

    step("check 4/6: stability across halves")
    mid_time = tr["entry_time"].min() + (tr["entry_time"].max()
                                         - tr["entry_time"].min()) / 2
    first = tr[tr["entry_time"] <= mid_time]["net_pct"]
    second = tr[tr["entry_time"] > mid_time]["net_pct"]
    m1 = float(first.mean()) if len(first) else float("nan")
    m2 = float(second.mean()) if len(second) else float("nan")
    c4 = (len(first) > 0 and len(second) > 0 and np.isfinite(m1)
          and np.isfinite(m2) and np.sign(m1) == np.sign(m2) and m2 > 0)
    checks.append(CheckResult(
        "C4", "Stability (half vs half)", c4, f"{_pct(m1)} / {_pct(m2)}",
        f"Mean net first half {_pct(m1)}, second half {_pct(m2)} "
        f"(bar: same sign, second half > 0). "
        + ("" if c4 else "A sign flip between halves is the signature of a "
                         "regime artifact, not an edge.")))

    step("check 5/6: regime robustness")
    regimes = []
    for _, row in tr.iterrows():
        d = drift_24h_at(klines[row["symbol"]],
                         int(row["entry_time"].timestamp() * 1000))
        regimes.append("UP" if d > REGIME_DRIFT
                       else "DOWN" if d < -REGIME_DRIFT
                       else "FLAT" if np.isfinite(d) else "NA")
    tr = tr.assign(regime=regimes)
    rt = (tr[tr["regime"] != "NA"].groupby("regime")["net_pct"]
          .agg(["count", "mean", "sum"]).reset_index())
    qual = rt[rt["count"] >= REGIME_MIN_TRADES]
    if len(qual) == 0:
        c5 = False
        key5 = "no qualifying regime"
        det5 = (f"No market regime holds >= {REGIME_MIN_TRADES} trades — "
                "cannot demonstrate robustness.")
    else:
        pos = int((qual["sum"] > 0).sum())
        c5 = pos / len(qual) >= REGIME_POS_FRAC
        key5 = f"{pos}/{len(qual)} regimes +"
        det5 = (f"{pos} of {len(qual)} qualifying regimes are net-positive "
                f"(bar: >= {REGIME_POS_FRAC:.0%}). Regimes = UP/DOWN/FLAT by "
                f"24h market drift at entry (+-{REGIME_DRIFT:.1%}).")
    checks.append(CheckResult("C5", "Regime robustness", c5, key5, det5))

    step("check 6/6: look-ahead (signal vs entry timing)")
    c6 = check_lookahead(tr)
    checks.append(c6)
    if c6.status == UNVERIFIABLE:
        warnings.append(f"C6 look-ahead UNVERIFIABLE ({c6.key_number}): "
                        "the judge could not check whether entries used "
                        "information from the bar they were booked on.")

    verdict = FAIL if any(c.status == FAIL for c in checks) else PASS
    return AuditResult(
        verdict=verdict, checks=checks, n_trades=n,
        n_dropped_overlap=dropped, reconstructed=reconstructed,
        fee_rt=fee_rt, fee_label=fee_label, warnings=warnings, trades=tr,
        bootstrap_totals=boot_totals, client_total=client_total,
        client_pctl=pctl, half_means=(m1, m2), regime_table=rt)
