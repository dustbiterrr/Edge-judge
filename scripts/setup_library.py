#!/usr/bin/env python3
"""
setup_library.py — 13 canonical discretionary-scalper setups codified as
strict, vectorized rules.

Contract (every setup):
    fn(df: pd.DataFrame, in_mask: np.ndarray[bool]) -> np.ndarray in {-1,0,+1}
  * signal at bar T uses ONLY data closed by end of T (rolling = backward);
    execution happens at open(T+1) — the judge handles that shift;
  * every threshold is data-driven: a quantile computed on the IN half only
    (in_mask), so the OUT half sees train-fitted constants (walk-forward
    clean).  The quantile LEVELS themselves are fixed in the spec, not tuned;
  * long/short conditions co-firing on the same bar -> 0 (no signal), never
    "last assignment wins";
  * NaN in any input -> condition False -> no signal.

df columns: open_time, open, high, low, close, volume, cvd_bar, n_trades,
            taker_buy_vol, oi, funding, premium
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _q(x: np.ndarray, p: float, in_mask: np.ndarray) -> float:
    """Quantile of x over the IN half only (NaNs dropped)."""
    v = np.asarray(x, dtype=np.float64)[in_mask]
    v = v[~np.isnan(v)]
    return float(np.quantile(v, p)) if v.size else float("nan")


def _roll(x: np.ndarray, n: int, fn: str, shift: int = 0) -> np.ndarray:
    s = pd.Series(x).rolling(n, min_periods=n)
    v = getattr(s, fn)()
    if shift:
        v = v.shift(shift)
    return v.values


def _shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    return pd.Series(x).shift(k).values


def _shift_bool(x: np.ndarray, k: int = 1) -> np.ndarray:
    return pd.Series(x).shift(k).fillna(False).astype(bool).values


def _resolve(long_m: np.ndarray, short_m: np.ndarray) -> np.ndarray:
    """Combine masks into {-1,0,+1}; conflicts (both true) -> 0."""
    sig = np.zeros(len(long_m))
    sig[long_m & ~short_m] = 1.0
    sig[short_m & ~long_m] = -1.0
    return sig


# ── setups ───────────────────────────────────────────────────────────────────

def s01_sweep_reversal(df, in_mask):
    """Liquidity-sweep reversal (classic stop-run fade, SMC/orderflow desks):
    a bar wicks through the prior 20-bar low but closes back above it — the
    sweep consumed resting stops and failed to hold, trapping breakout sellers.
    Long; mirrored at prior highs for shorts."""
    lo_prev = _roll(df["low"].values, 20, "min", shift=1)
    hi_prev = _roll(df["high"].values, 20, "max", shift=1)
    low, high, close = df["low"].values, df["high"].values, df["close"].values
    lm = (low < lo_prev) & (close > lo_prev)
    sm = (high > hi_prev) & (close < hi_prev)
    return _resolve(lm, sm)


def s02_vwap_fade(df, in_mask):
    """VWAP stretch fade (pit/floor-trader mean reversion): when price is
    stretched beyond its typical (IN-q80) distance from rolling VWAP-24,
    fade back toward VWAP — the crowd's average entry acts as a magnet."""
    c, v = df["close"].values, df["volume"].values
    pv = _roll(c * v, 24, "sum")
    vv = _roll(v, 24, "sum")
    vwap = np.where(vv > 0, pv / np.where(vv > 0, vv, 1.0), np.nan)
    stretch = np.abs(c - vwap) / vwap
    thr = _q(stretch, 0.80, in_mask)
    m = stretch > thr
    return _resolve(m & (c < vwap), m & (c > vwap))


def s03_absorption(df, in_mask):
    """Absorption / effort-vs-result (orderflow scalpers, footprint traders):
    an outsized taker delta (|cvd| > IN-q80) that FAILED to move the candle
    its way — passive liquidity absorbed it.  Per spec: enter WITH the
    aggressor flow's sign, betting the pressure eventually breaks through."""
    cvd = df["cvd_bar"].values
    body = df["close"].values - df["open"].values
    thr = _q(np.abs(cvd), 0.80, in_mask)
    m = (np.abs(cvd) > thr) & (np.sign(cvd) * np.sign(body) < 0)
    return np.where(m, np.sign(cvd), 0.0)


def s04_liquidation_bounce(df, in_mask):
    """Capitulation-wick bounce (liquidation-cascade scalpers): an extreme
    range bar (IN-q95) closing in its bottom third with extreme negative
    delta = forced sellers exhausted; buy the flush.  Mirrored for squeezes."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    cvd = df["cvd_bar"].values
    rng = (h - l) / o
    q95 = _q(rng, 0.95, in_mask)
    span = np.where(h > l, h - l, np.nan)
    pos = (c - l) / span
    cq20 = _q(cvd, 0.20, in_mask)
    cq80 = _q(cvd, 0.80, in_mask)
    lm = (rng > q95) & (pos < 1 / 3) & (cvd < cq20)
    sm = (rng > q95) & (pos > 2 / 3) & (cvd > cq80)
    return _resolve(lm, sm)


def s05_range_fade(df, in_mask):
    """Range-edge fade (consolidation scalpers): inside an unusually narrow
    20-bar channel (width < IN-q40), sell the top decile of the channel and
    buy the bottom decile — no breakout energy, edges revert."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    hh = _roll(h, 20, "max")
    ll = _roll(l, 20, "min")
    width = (hh - ll) / c
    q40 = _q(width, 0.40, in_mask)
    span = np.where(hh > ll, hh - ll, np.nan)
    pos = (c - ll) / span
    narrow = width < q40
    return _resolve(narrow & (pos < 0.10), narrow & (pos > 0.90))


def s06_breakout_continuation(df, in_mask):
    """Breakout WITH flow confirmation (momentum scalpers): close beyond the
    prior 96-bar extreme backed by strong same-side taker delta (IN-q70) —
    the filter generic breakout screens lacked."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    cvd = df["cvd_bar"].values
    hi96 = _roll(h, 96, "max", shift=1)
    lo96 = _roll(l, 96, "min", shift=1)
    c70 = _q(cvd, 0.70, in_mask)
    c30 = _q(cvd, 0.30, in_mask)
    return _resolve((c > hi96) & (cvd > c70), (c < lo96) & (cvd < c30))


def s07_failed_breakout(df, in_mask):
    """Failed-breakout fade (the contrarian twin of S06, favored by range
    traders): yesterday's close broke the 96-bar level, today's close fell
    back through it — trapped breakout traders must unwind; fade them."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    hi96 = _roll(h, 96, "max", shift=1)
    lo96 = _roll(l, 96, "min", shift=1)
    bo_up_prev = _shift_bool(c > hi96)
    bo_dn_prev = _shift_bool(c < lo96)
    hi_prev = _shift(hi96)
    lo_prev = _shift(lo96)
    sm = bo_up_prev & (c < hi_prev)
    lm = bo_dn_prev & (c > lo_prev)
    return _resolve(lm, sm)


def s08_oi_purge_bounce(df, in_mask):
    """Open-interest purge (liquidation traders): a sharp 4-bar OI drop
    (IN-q10) = positions force-closed.  Purge on falling price -> longs were
    flushed, buy; purge on rising price -> shorts squeezed out, sell."""
    oi, c = df["oi"].values, df["close"].values
    doi = oi - _shift(oi, 4)
    q10 = _q(doi, 0.10, in_mask)
    ret4 = c / _shift(c, 4) - 1.0
    m = doi < q10
    return _resolve(m & (ret4 < 0), m & (ret4 > 0))


def s09_premium_extreme_fade(df, in_mask):
    """Perp-premium extreme fade (funding arbitrageurs' entry signal): premium
    z-score beyond +-2 (window 96) = crowded perp positioning; fade it.  The
    +-2 thresholds are pre-registered from the previous generic probe (the
    57-59%% hit cells) — now through the full walk-forward judge."""
    pm = pd.Series(df["premium"].values)
    mu = pm.rolling(96, min_periods=96).mean()
    sd = pm.rolling(96, min_periods=96).std()
    z = ((pm - mu) / sd.replace(0.0, np.nan)).values
    zc = np.nan_to_num(z)
    return np.where(np.abs(zc) > 2.0, -np.sign(zc), 0.0)


def s10_delta_divergence_trend(df, in_mask):
    """Delta divergence at fresh extremes (footprint/CVD traders): price
    prints a new 20-bar closing high while cumulative 20-bar delta does NOT —
    the push lacks real buying; short the weakness.  Mirrored at lows."""
    c, cvd = df["close"].values, df["cvd_bar"].values
    cmax_prev = _roll(c, 20, "max", shift=1)
    cmin_prev = _roll(c, 20, "min", shift=1)
    cvd20 = _roll(cvd, 20, "sum")
    dmax_prev = _roll(cvd20, 20, "max", shift=1)
    dmin_prev = _roll(cvd20, 20, "min", shift=1)
    sm = (c > cmax_prev) & (cvd20 < dmax_prev)
    lm = (c < cmin_prev) & (cvd20 > dmin_prev)
    return _resolve(lm, sm)


def s11_stop_hunt_reentry(df, in_mask):
    """Stop-hunt with confirmation (patient SMC entry): S01's sweep bar, but
    wait one bar — enter only after the NEXT bar closes back above the sweep
    bar's open, confirming the reversal instead of catching the knife."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    lo_prev = _roll(l, 20, "min", shift=1)
    hi_prev = _roll(h, 20, "max", shift=1)
    sweep_lo = (l < lo_prev) & (c > lo_prev)
    sweep_hi = (h > hi_prev) & (c < hi_prev)
    o_prev = _shift(o)
    lm = _shift_bool(sweep_lo) & (c > o_prev)
    sm = _shift_bool(sweep_hi) & (c < o_prev)
    return _resolve(lm, sm)


def s12_compression_expansion(df, in_mask):
    """Volatility compression -> expansion (classic squeeze play): after >=3
    consecutive ultra-low-ATR bars (IN-q20), the first bar with real range
    (IN-q60) sets the direction — ride the released energy."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    c_prev = _shift(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    atrn = _roll(tr, 14, "mean") / c
    q20 = _q(atrn, 0.20, in_mask)
    comp = np.nan_to_num(atrn, nan=np.inf) < q20
    comp3_prev = (_shift_bool(comp, 1) & _shift_bool(comp, 2)
                  & _shift_bool(comp, 3))
    rngn = (h - l) / o
    q60 = _q(rngn, 0.60, in_mask)
    m = comp3_prev & (rngn > q60)
    return np.where(m, np.sign(c - o), 0.0)


def s13_funding_flip_momentum(df, in_mask):
    """Funding sign flip (positioning traders): the 8h funding rate crossing
    zero marks a fresh positioning imbalance; trade in the direction of the
    NEW funding sign while the crowd is still repositioning."""
    f = df["funding"].values
    fp = _shift(f)
    flip = (f != fp) & (np.sign(f) * np.sign(fp) < 0)
    flip = np.where(np.isnan(f) | np.isnan(fp), False, flip)
    return np.where(flip, np.sign(f), 0.0)


SETUPS: dict[str, callable] = {
    "S01 sweep-reversal": s01_sweep_reversal,
    "S02 vwap-fade": s02_vwap_fade,
    "S03 absorption": s03_absorption,
    "S04 liq-bounce": s04_liquidation_bounce,
    "S05 range-fade": s05_range_fade,
    "S06 breakout-cont": s06_breakout_continuation,
    "S07 failed-breakout": s07_failed_breakout,
    "S08 oi-purge-bounce": s08_oi_purge_bounce,
    "S09 premium-fade": s09_premium_extreme_fade,
    "S10 delta-diverge": s10_delta_divergence_trend,
    "S11 stophunt-reentry": s11_stop_hunt_reentry,
    "S12 compress-expand": s12_compression_expansion,
    "S13 funding-flip": s13_funding_flip_momentum,
}


# ═════════════════════════════════════════════════════════════════════════════
# APPENDIX B — pre-registered FINAL wave of single-asset setups (S14-S19).
# Declared before the run: regardless of outcome there will be NO further
# single-asset setup waves on this dataset.  All are judged non-overlapping
# from the first run (the S02 lesson).
# ═════════════════════════════════════════════════════════════════════════════

_MS8H = 8 * 3_600_000
_MS15M = 15 * 60_000


def s14_funding_snapshot_fade(df, in_mask):
    """Funding-snapshot fade (15m): on the bar opening
    30min before the 00/08/16 UTC snapshot, an extreme premium (IN q90/q10
    of pre-snapshot bars) marks crowded positioning about to pay/receive —
    fade it into and just past the payment."""
    t = df["open_time"].values.astype(np.int64)
    prem = df["premium"].values
    pre = (t % _MS8H) == (_MS8H - 2 * _MS15M)      # opens t_s - 30min
    pv = prem[pre & in_mask]
    pv = pv[~np.isnan(pv)]
    if pv.size < 20:
        return np.zeros(len(df))
    q90, q10 = np.quantile(pv, 0.90), np.quantile(pv, 0.10)
    sig = np.zeros(len(df))
    sig[pre & (prem > q90)] = -1.0
    sig[pre & (prem < q10)] = 1.0
    return sig


def s15_coiled_spring_oi(df, in_mask):
    """Coiled spring with OI build (breakout traders): a 20-bar channel
    narrower than q20(IN) in ATR14 units WHILE 20-bar OI grew above q85(IN)
    = positions loading inside compression; trade the first close that
    escapes the channel."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    oi = df["oi"].values
    c_prev = _shift(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    atr = _roll(tr, 14, "mean")
    hh = _roll(h, 20, "max", shift=1)
    ll = _roll(l, 20, "min", shift=1)
    atr_prev = _shift(atr)
    width_atr = (hh - ll) / np.where(atr_prev > 0, atr_prev, np.nan)
    doi20 = oi - _shift(oi, 20)
    q20 = _q(width_atr, 0.20, in_mask)
    q85 = _q(_shift(doi20), 0.85, in_mask)
    armed = (width_atr < q20) & (_shift(doi20) > q85)
    return _resolve(armed & (c > hh), armed & (c < ll))


def s16_cvd_accel_exhaustion(df, in_mask):
    """CVD deceleration at fresh extremes (orderflow): new 20-bar closing
    high on strong (q70 IN) but DECELERATING taker delta = the last buyers;
    fade.  Mirrored at lows with decaying sell delta."""
    c, cvd = df["close"].values, df["cvd_bar"].values
    cvd_prev = _shift(cvd)
    hi20 = _roll(c, 20, "max")
    lo20 = _roll(c, 20, "min")
    q70 = _q(cvd, 0.70, in_mask)
    q30 = _q(cvd, 0.30, in_mask)
    sm = (c >= hi20) & (cvd > q70) & (cvd < cvd_prev)
    lm = (c <= lo20) & (cvd < q30) & (cvd > cvd_prev)
    return _resolve(lm, sm)


def s17_toxic_flow_absorption(df, in_mask):
    """Toxic flow absorbed (tape readers): >=80% of bar volume was taker
    BUYING yet the bar closed in its bottom 30% — buyers dumped into passive
    sellers who won; short.  Mirrored for absorbed selling."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    v = df["volume"].values
    tb = df["taker_buy_vol"].values
    ratio = np.where(v > 0, tb / np.where(v > 0, v, 1.0), np.nan)
    span = np.where(h > l, h - l, np.nan)
    pos = (c - l) / span
    sm = (ratio > 0.80) & (pos < 0.30)
    lm = (ratio < 0.20) & (pos > 0.70)
    return _resolve(lm, sm)


def s18_strict_fvg_fill(df, in_mask):
    """Strict Fair-Value-Gap fill (SMC): a q80(IN)-range displacement bar j
    leaving a 3-bar gap (low(j) > high(j-2)); when price returns INTO the
    zone by the bar's LOW (no intrabar guessing) within 48 bars, with
    positive delta, buy the fill.  One entry per FVG; a sweep through the
    zone bottom invalidates it.  Mirrored bearish."""
    h, l, c, o = (df[k].values for k in ("high", "low", "close", "open"))
    cvd = df["cvd_bar"].values
    n = len(df)
    rng = (h - l) / np.where(o > 0, o, np.nan)
    q80 = _q(rng, 0.80, in_mask)
    sig = np.zeros(n)
    bulls: list[tuple[int, float, float]] = []   # (birth, top=low(j), bot=high(j-2))
    bears: list[tuple[int, float, float]] = []   # (birth, bot=high(j), top=low(j-2))
    # v1.0.1 hit-logic fix (post-dates the shipped CSVs; every S18 cell was
    # "insufficient", status unchanged): bull/bear fills are collected
    # separately — two SAME-side fills on one bar are still one signal;
    # only a bull+bear conflict cancels to 0.  The old in-place expression
    # zeroed a second bear fill against the first.
    for T in range(2, n):
        bulls = [g for g in bulls if T - g[0] <= 48]
        bears = [g for g in bears if T - g[0] <= 48]
        bull_hit = False
        bear_hit = False
        for i, (b, top, bot) in enumerate(bulls):
            if T == b:
                continue
            if l[T] < bot:                       # swept through -> invalidated
                bulls[i] = None
            elif l[T] <= top and cvd[T] > 0:
                bull_hit = True
                bulls[i] = None
        bulls = [g for g in bulls if g is not None]
        for i, (b, bot, top) in enumerate(bears):
            if T == b:
                continue
            if h[T] > top:
                bears[i] = None
            elif h[T] >= bot and cvd[T] < 0:
                bear_hit = True
                bears[i] = None
        bears = [g for g in bears if g is not None]
        sig[T] = 0.0 if bull_hit == bear_hit else (1.0 if bull_hit else -1.0)
        if not np.isnan(rng[T]) and rng[T] > q80:
            if l[T] > h[T - 2]:
                bulls.append((T, l[T], h[T - 2]))
            if h[T] < l[T - 2]:
                bears.append((T, h[T], l[T - 2]))
    return sig


def s19_liquidity_cascade(df, in_mask):
    """Liquidation cascade with flow confirmation (cascade snipers): a close
    below the prior 48-bar low WITH a q05(IN) one-bar OI dump = forced
    unwind; if the NEXT bar's delta turns positive, the cascade is spent —
    long at open(T+2).  Mirrored for short squeezes.  (Signal is placed on
    the confirmation bar T+1, so the judge's open(T+1) entry lands at
    open(T+2) of the cascade bar — the author's look-ahead is fixed.)"""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    oi, cvd = df["oi"].values, df["cvd_bar"].values
    lo48 = _roll(l, 48, "min", shift=1)
    hi48 = _roll(h, 48, "max", shift=1)
    doi1 = oi - _shift(oi)
    q05 = _q(doi1, 0.05, in_mask)
    casc_dn = (c < lo48) & (doi1 < q05)
    casc_up = (c > hi48) & (doi1 < q05)
    lm = _shift_bool(casc_dn) & (cvd > 0)
    sm = _shift_bool(casc_up) & (cvd < 0)
    return _resolve(lm, sm)


APPENDIX_B_SETUPS: dict[str, callable] = {
    "S14 fund-snapshot": s14_funding_snapshot_fade,       # 15m only, H {2,4}
    "S15 coiled-spring": s15_coiled_spring_oi,
    "S16 cvd-exhaustion": s16_cvd_accel_exhaustion,
    "S17 toxic-flow": s17_toxic_flow_absorption,
    "S18 fvg-fill": s18_strict_fvg_fill,
    "S19 liq-cascade": s19_liquidity_cascade,
}


# ═════════════════════════════════════════════════════════════════════════════
# APPENDIX C — structurally NEW classes only (Appendix B finality still holds
# for single-asset directional).  S22 (cointegration StatArb) lives in
# scripts/appendix_c_statarb.py (pairs, not bars->{-1,0,+1}).  Here:
#   S23 weekend liquidity-vacuum (TEMPORAL — exploits a window, not a pattern)
#   S21 liquidation-squeeze CONTROL (single-asset directional; pre-declared
#       "insufficient", the S08+S19 twin — run only to reproduce that fate)
# ═════════════════════════════════════════════════════════════════════════════

def s21_liq_squeeze_control(df, in_mask):
    """CONTROL (not a new class): OI building for 24h above IN-q80 while funding
    is deeply negative (IN-q15), then a micro-break of the 12h high on FALLING
    1-bar OI = a squeeze ignition.  A direct twin of the dead S08/S19; run only
    to confirm the pre-declared 'insufficient' prediction.  Long only (+1)."""
    oi = df["oi"].values
    c, h = df["close"].values, df["high"].values
    fund = df["funding"].values
    doi24 = oi - _shift(oi, 24)                       # 1h TF: 24 bars = 24h
    q80 = _q(doi24, 0.80, in_mask)
    fq15 = _q(fund, 0.15, in_mask)
    hi12_prev = _roll(h, 12, "max", shift=1)
    doi1 = oi - _shift(oi, 1)
    m = ((doi24 > 0) & (doi24 > q80) & (fund < fq15)
         & (c > hi12_prev) & (doi1 < 0))
    return np.where(np.nan_to_num(m.astype(float)) > 0, 1.0, 0.0)


def s23_weekend_vacuum(df, in_mask):
    """TEMPORAL — weekend liquidity vacuum (Sat 00:00 -> Sun 12:00 UTC): thin
    books let a lone sweep break the prior 48h extreme; if taker delta on the
    break bar leans AGAINST it (absorption, S03-style), fade the sweep back.
    Break high -> short (-1); break low -> long (+1).  Exit is calendar-based
    (Monday 00:00 UTC) and handled by the probe, not here."""
    ot = df["open_time"].values.astype(np.int64)
    dt = int(np.median(np.diff(ot))) if len(ot) > 1 else 3_600_000
    b48 = max(2, round(48 * 3_600_000 / dt))
    idx = pd.DatetimeIndex(pd.to_datetime(ot, unit="ms", utc=True))
    wd = idx.dayofweek.values
    hr = idx.hour.values
    inwin = (wd == 5) | ((wd == 6) & (hr < 12))
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    cvd = df["cvd_bar"].values
    hi_prev = _roll(h, b48, "max", shift=1)
    lo_prev = _roll(l, b48, "min", shift=1)
    up_break = inwin & (c > hi_prev) & (cvd < 0)      # sweep up, selling absorbs
    dn_break = inwin & (c < lo_prev) & (cvd > 0)      # sweep down, buying absorbs
    return _resolve(np.nan_to_num(dn_break).astype(bool),
                    np.nan_to_num(up_break).astype(bool))


APPENDIX_C_SETUPS = {
    "S21 liq-squeeze-CTRL": s21_liq_squeeze_control,   # 1h, control
    "S23 weekend-vacuum": s23_weekend_vacuum,          # 15m + 1h, calendar exit
}


# ═════════════════════════════════════════════════════════════════════════════
# APPENDIX D — personal CLOSURE run (NOT part of the public campaign).
# S24 is single-asset directional, COVERED by Appendix B's finality; run by
# request as a closure test. Different signature: takes explicit `tf` kwarg and
# returns a pd.Series (+ stashes a funnel dict on the function object).
# ═════════════════════════════════════════════════════════════════════════════

def _s24_core(bars: pd.DataFrame, in_mask, *, tf: str,
              adaptive: bool) -> tuple[pd.Series, dict]:
    """
    S24 KINETIC CHOKE — Effort vs Result + OI Trap + Funding Gravity.

    Hypothesis: aggressive flow slams a limit wall (high EvR), new leverage
    stacks into it (OI up), and it overpays funding.  A break of the choke
    bar's low ignites the trapped.

    Signal at bar T -> execution at open(T+1)  [campaign convention].
    adaptive=False (S24a): funding/EvR thresholds = IN-half quantiles (static),
                           oi_delta strict. Comparable to the campaign.
    adaptive=True  (S24b): funding thresholds = rolling(14d) quantiles,
                           oi_delta = rolling(3).max(). NOT isolated to IN.
    tf: '15m' | '1h' passed explicitly (index is RangeIndex; do not derive it).
    """
    signal = pd.Series(0, index=bars.index, dtype=int)

    effort = bars['cvd_bar'].abs()
    result = (bars['close'] - bars['open']).abs() / bars['open']
    evr = effort / result.clip(lower=1e-4)
    evr_smooth = evr.rolling(window=3, min_periods=3).median()

    q95_evr = evr_smooth.loc[in_mask].quantile(0.95)

    if adaptive:
        roll_w = 1344 if tf == '15m' else 336          # ~14 days
        q75_f = bars['funding'].rolling(roll_w, min_periods=96).quantile(0.75)
        q25_f = bars['funding'].rolling(roll_w, min_periods=96).quantile(0.25)
    else:
        q75_f = bars.loc[in_mask, 'funding'].quantile(0.75)
        q25_f = bars.loc[in_mask, 'funding'].quantile(0.25)

    oi_delta_raw = bars['oi'].diff()
    oi_up = (oi_delta_raw.rolling(window=3, min_periods=1).max() > 0
             if adaptive else oi_delta_raw > 0)

    longs_trapped = (
        (bars['close'] > bars['open']) &
        (evr_smooth > q95_evr) &
        oi_up &
        (bars['funding'] > q75_f)
    )
    shorts_trapped = (
        (bars['close'] < bars['open']) &
        (evr_smooth > q95_evr) &
        oi_up &
        (bars['funding'] < q25_f)
    )

    lt_prev = longs_trapped.shift(1).fillna(False)
    st_prev = shorts_trapped.shift(1).fillna(False)
    trigger_short = lt_prev & (bars['close'] < bars['low'].shift(1))
    trigger_long = st_prev & (bars['close'] > bars['high'].shift(1))

    signal.loc[trigger_short] = -1
    signal.loc[trigger_long] = 1
    signal = signal.where(signal != signal.shift(1), 0)

    evr_hit = evr_smooth > q95_evr
    funnel = {
        'variant': 'S24b-adaptive' if adaptive else 'S24a-static',
        'tf': tf,
        'bars_total': len(bars),
        'evr_extreme': int(evr_hit.sum()),
        'plus_bullish': int((evr_hit & (bars['close'] > bars['open'])).sum()),
        'plus_oi_up': int((evr_hit & (bars['close'] > bars['open']) & oi_up).sum()),
        'plus_funding': int(longs_trapped.sum()),
        'trigger_short': int(trigger_short.sum()),
        'trigger_long': int(trigger_long.sum()),
        'signals_total': int((signal != 0).sum()),
        'oi_delta_zero_pct': round(float((oi_delta_raw == 0).mean()) * 100, 1),
        'funding_unique': int(bars['funding'].nunique()),
    }
    return signal, funnel


def setup_24a_kinetic_choke_static(bars, in_mask, *, tf: str) -> pd.Series:
    sig, funnel = _s24_core(bars, in_mask, tf=tf, adaptive=False)
    setup_24a_kinetic_choke_static.last_funnel = funnel
    return sig


def setup_24b_kinetic_choke_adaptive(bars, in_mask, *, tf: str) -> pd.Series:
    sig, funnel = _s24_core(bars, in_mask, tf=tf, adaptive=True)
    setup_24b_kinetic_choke_adaptive.last_funnel = funnel
    return sig
