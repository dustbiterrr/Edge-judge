#!/usr/bin/env python3
"""
funding_carry_probe.py — FINAL probe of the campaign: structural,
direction-neutral funding/basis carry (short perp + long spot).

The question is not "does the premium exist" (it does, structurally) but
"does it BEAT COSTS, and how many DOLLARS is that on $1k / $10k deployed".

Mechanics:
  * short perp RECEIVES funding when rate > 0, PAYS when rate < 0;
  * funding accrues at PAYMENT EVENTS only: bars whose open_time sits on the
    8h UTC boundary (00/08/16) — the rate on that bar is the rate just paid
    (value-change detection is printed as a cross-check; it undercounts when
    consecutive rates are identical, e.g. the 0.01% clamp);
  * basis MTM per cycle: pnl ~= premium_entry - premium_exit (profit when the
    premium narrows against our short perp);
  * decisions at close(T) execute at open(T+1); funding events are collected
    strictly AFTER the entry bar and BEFORE the exit bar;
  * costs per completed cycle = perp_RT + spot_RT (both legs, entry+exit),
    charged at cycle close; taker and maker variants both reported;
  * capital: leg notional N, perp margin 3x -> deployed = N + N/3 = 1.333N;
    ALL percentages and annualized figures are ON DEPLOYED CAPITAL;
  * annualization: linear, net% x 365/period_days (no compounding).

Walk-forward: 50/50 by time, sim runs separately per half (forced close at
half boundaries, costs charged).  S-B threshold (q60 of positive IN funding
events) comes from IN only.  "Best strategy" is selected on IN (taker
annualized) and judged on its OUT — selection never touches the verdict half.

PRE-REGISTERED VERDICT (declared before the run):
  (a) best strategy's OUT net annualized >= 8%/yr on deployed, at TAKER
      (maker printed as upside; maker-only pass is labeled as such);
  (b) sign of IN net matches OUT;
  (c) the verdict MUST print $/month at $1,000 and $10,000 deployed.
  Below 8%/yr -> "structurally alive but under the operational bar — no trade".
Guard: symbol with < 100 funding events over the full period -> insufficient.

Usage:
    python scripts/funding_carry_probe.py [--spot-rt 0.20]
"""

from __future__ import annotations

import argparse
import sys
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
PERP_RT = {"taker": 0.11, "maker": 0.04}     # % round trip, perp leg
DEPLOY = 4.0 / 3.0                           # deployed capital per notional N
MIN_EVENTS = 100
BAR_YR = 8.0                                 # pre-registered: 8%/yr on deployed
RES = Path("out")                 # results/ is the immutable evidence base
EVIDENCE = Path("results")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_matrix(data_dir: Path) -> dict:
    """Unified bar grid across symbols: funding(ffill), event mask, premium."""
    frames = {}
    for sym in SYMBOLS:
        fp = data_dir / sym / "1h" / "bars.parquet"
        if not fp.is_file():
            print(f"  [skip] {sym}: missing")
            continue
        df = pd.read_parquet(fp)[["open_time", "funding", "premium"]]
        frames[sym] = df.set_index("open_time")
    if not frames:
        return None
    grid = sorted(set().union(*[set(f.index) for f in frames.values()]))
    t = np.array(grid, dtype=np.int64)
    hour = (t // TF_MS) % 24
    event = (hour % 8) == 0                    # 00/08/16 UTC payment bars
    fund = pd.DataFrame(index=grid)
    prem = pd.DataFrame(index=grid)
    for sym, f in frames.items():
        fund[sym] = f["funding"].reindex(grid)
        prem[sym] = f["premium"].reindex(grid)
    return {"t": t, "event": event, "fund": fund, "prem": prem}


# ─────────────────────────────────────────────────────────────────────────────
# Cycle accounting
# ─────────────────────────────────────────────────────────────────────────────

class Ledger:
    """Per-bar pnl components in % of leg notional N (converted to deployed
    at report time).  Costs applied per fee mode at cycle-close bars."""

    def __init__(self, n: int):
        self.fund = np.zeros(n)        # funding pnl at event bars
        self.basis = np.zeros(n)       # basis pnl realized at cycle closes
        self.closes = np.zeros(n)      # number of cycle closes per bar
        self.events: list[float] = []  # collected funding rates (for streaks)

    def equity(self, cost_rt: float) -> np.ndarray:
        return np.cumsum(self.fund + self.basis - self.closes * cost_rt)

    def stats(self, cost_rt: float) -> dict:
        gross = float(self.fund.sum())
        basis = float(self.basis.sum())
        cycles = float(self.closes.sum())      # fractional for portfolio/slots
        cost = cycles * cost_rt
        net = gross + basis - cost
        eq = self.equity(cost_rt)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
        # max consecutive negative-funding streak among collected events
        streak, worst, s_cost, w_cost = 0, 0, 0.0, 0.0
        for r in self.events:
            if r < 0:
                streak += 1
                s_cost += r * 100
                if streak > worst:
                    worst, w_cost = streak, s_cost
            else:
                streak, s_cost = 0, 0.0
        return {"n_events": len(self.events), "gross": gross, "basis": basis,
                "cycles": cycles, "cost": cost, "net": net, "dd": dd,
                "neg_streak": worst, "streak_cost": w_cost}


def collect(led: Ledger, fund: np.ndarray, event: np.ndarray,
            entry_bar: int, exit_bar: int) -> None:
    """Funding events strictly after entry bar, strictly before exit bar."""
    for e in range(entry_bar + 1, exit_bar):
        if event[e] and not np.isnan(fund[e]):
            led.fund[e] += fund[e] * 100.0
            led.events.append(float(fund[e]))


# ─────────────────────────────────────────────────────────────────────────────
# Strategies (sim on one half; decisions close(T) -> execution open(T+1))
# ─────────────────────────────────────────────────────────────────────────────

def sim_always_on(fund: np.ndarray, prem: np.ndarray,
                  event: np.ndarray) -> Ledger:
    """S-A: one entry at the start of the half, one exit at its end."""
    n = len(fund)
    led = Ledger(n)
    entry, exit_ = 1, n - 1
    collect(led, fund, event, entry, exit_)
    p_in = prem[0] if not np.isnan(prem[0]) else 0.0
    p_out = prem[exit_ - 1] if not np.isnan(prem[exit_ - 1]) else 0.0
    led.basis[exit_] += (p_in - p_out) * 100.0
    led.closes[exit_] += 1
    return led


def sim_conditional(fund: np.ndarray, prem: np.ndarray, event: np.ndarray,
                    thr: float) -> Ledger:
    """S-B: enter when ffill funding > thr(IN q60 of positive events);
    exit after two consecutive negative funding EVENTS."""
    n = len(fund)
    led = Ledger(n)
    pos, entry_bar, p_in, neg2 = False, -1, 0.0, 0
    for tt in range(n - 1):
        if pos and tt > entry_bar and event[tt] and not np.isnan(fund[tt]):
            led.fund[tt] += fund[tt] * 100.0
            led.events.append(float(fund[tt]))
            neg2 = neg2 + 1 if fund[tt] < 0 else 0
        f = fund[tt]
        if not pos:
            if not np.isnan(f) and f > thr:
                pos, entry_bar, neg2 = True, tt + 1, 0
                p_in = prem[tt] if not np.isnan(prem[tt]) else 0.0
        elif neg2 >= 2:
            p_out = prem[tt] if not np.isnan(prem[tt]) else 0.0
            led.basis[tt + 1] += (p_in - p_out) * 100.0
            led.closes[tt + 1] += 1
            pos = False
    if pos:                                    # force close at half end
        p_out = prem[n - 2] if not np.isnan(prem[n - 2]) else 0.0
        led.basis[n - 1] += (p_in - p_out) * 100.0
        led.closes[n - 1] += 1
    return led


def sim_rotation(fund: pd.DataFrame, prem: pd.DataFrame, event: np.ndarray,
                 syms: list[str]) -> Ledger:
    """S-C: every 24h rank symbols by current ffill rate, hold top-3 equally
    (1/3 notional each); swap only changed slots; min-hold enforced by the
    daily cadence itself."""
    n = len(fund)
    led = Ledger(n)
    F = {s: fund[s].values for s in syms}
    P = {s: prem[s].values for s in syms}
    held: dict[str, tuple[int, float]] = {}    # sym -> (entry_bar, prem_in)
    for tt in range(n - 1):
        for s, (eb, _) in held.items():
            if tt > eb and event[tt] and not np.isnan(F[s][tt]):
                led.fund[tt] += F[s][tt] * 100.0 / 3.0
                led.events.append(float(F[s][tt]) / 3.0)
        if tt % 24 == 0:
            ranks = {s: F[s][tt] for s in syms if not np.isnan(F[s][tt])}
            target = set(sorted(ranks, key=ranks.get, reverse=True)[:3])
            for s in [x for x in held if x not in target]:
                eb, p_in = held.pop(s)
                p_out = P[s][tt] if not np.isnan(P[s][tt]) else 0.0
                led.basis[tt + 1] += (p_in - p_out) * 100.0 / 3.0
                led.closes[tt + 1] += 1.0 / 3.0    # cost scales with slot size
            for s in target - set(held):
                held[s] = (tt + 1,
                           P[s][tt] if not np.isnan(P[s][tt]) else 0.0)
    for s, (eb, p_in) in held.items():             # force close at half end
        p_out = P[s][n - 2] if not np.isnan(P[s][n - 2]) else 0.0
        led.basis[n - 1] += (p_in - p_out) * 100.0 / 3.0
        led.closes[n - 1] += 1.0 / 3.0
    return led


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/native")
    ap.add_argument("--spot-rt", type=float, default=0.20,
                    help="spot leg round-trip cost %% (default 0.20)")
    ap.add_argument("--outdir", default=str(RES),
                    help="where funding_probe_full.csv is written (default: "
                         "out/). 'results' overwrites the campaign evidence "
                         "base - full reproduction only.")
    args = ap.parse_args()
    RES = Path(args.outdir)
    if RES.resolve() == EVIDENCE.resolve():
        print("  !! --outdir results: OVERWRITING the campaign evidence base. "
              "Only meaningful with all 13 symbols fetched for the campaign "
              "window.")

    print("=" * 98)
    print("  FUNDING/BASIS CARRY PROBE — short perp + long spot, "
          "delta-neutral (the campaign's final door)")
    print(f"  costs/cycle: perp RT taker {PERP_RT['taker']}% / maker "
          f"{PERP_RT['maker']}%  +  spot RT {args.spot_rt}%   |   "
          f"deployed = 1.333 x N, all %% on deployed")
    print(f"  verdict (pre-registered): OUT annualized >= {BAR_YR}%/yr on "
          f"deployed at TAKER; strategy selected on IN")
    print("=" * 98)

    M = load_matrix(Path(args.data_dir))
    if M is None:
        print(f"\n  nothing evaluated: no bars under {args.data_dir}/ for the "
              "basket. Fetch first (python scripts/fetch_binance_native.py "
              "--symbol ETHUSDT --months 6); nothing written.")
        return 2
    t, event, fund, prem = M["t"], M["event"], M["fund"], M["prem"]
    n = len(t)
    mid = n // 2
    halves = {"IN": slice(0, mid), "OUT": slice(mid, n)}
    days = {h: (halves[h].stop - halves[h].start) / 24.0 for h in halves}
    syms = [s for s in SYMBOLS if s in fund.columns]

    # guard + event-count cross-check
    print("\n  funding-event audit (8h-boundary events vs value-change count, "
          "full period):")
    valid_syms = []
    for s in syms:
        col = fund[s].values
        ev_n = int(np.sum(event & ~np.isnan(col)))
        chg = int(np.sum(pd.Series(col).diff().fillna(0) != 0))
        ok = ev_n >= MIN_EVENTS
        print(f"    {s:<9s} events={ev_n:>4d}  value-changes={chg:>4d}  "
              f"{'ok' if ok else 'INSUFFICIENT'}")
        if ok:
            valid_syms.append(s)

    # S-B thresholds from IN events only
    thr = {}
    for s in valid_syms:
        col = fund[s].values[halves["IN"]]
        ev = col[event[halves["IN"]] & ~np.isnan(col)]
        pos = ev[ev > 0]
        thr[s] = float(np.quantile(pos, 0.60)) if len(pos) >= 5 else np.nan

    rows = []
    port_paths: dict[tuple, dict[str, np.ndarray]] = {}

    def add_row(strat, scope, half, led: Ledger):
        for mode, prt in PERP_RT.items():
            cost_rt = prt + args.spot_rt
            st = led.stats(cost_rt)
            net_dep = st["net"] / DEPLOY
            ann = net_dep * 365.0 / days[half]
            rows.append({"strategy": strat, "scope": scope, "half": half,
                         "fee": mode, "n_events": st["n_events"],
                         "gross_pct": st["gross"], "basis_pct": st["basis"],
                         "cycles": round(st["cycles"], 2),
                         "cost_pct": st["cost"],
                         "net_dep_pct": net_dep, "ann_dep_pct": ann,
                         "neg_streak": st["neg_streak"],
                         "streak_cost_pct": st["streak_cost"],
                         "maxdd_dep_pct": st["dd"] / DEPLOY})

    for half, sl in halves.items():
        ev_h = event[sl]
        for s in valid_syms:
            f = fund[s].values[sl]
            p = prem[s].values[sl]
            led_a = sim_always_on(f, p, ev_h)
            add_row("S-A always-on", s, half, led_a)
            led_b = sim_conditional(f, p, ev_h, thr[s])
            add_row("S-B conditional", s, half, led_b)
            port_paths[("A", half, s)] = led_a
            port_paths[("B", half, s)] = led_b
        led_c = sim_rotation(fund.iloc[sl].reset_index(drop=True),
                             prem.iloc[sl].reset_index(drop=True),
                             ev_h, valid_syms)
        add_row("S-C top3-rotation", "PORTFOLIO", half, led_c)
        print(f"  [done] half {half}: {len(valid_syms)} symbols x S-A/S-B + S-C")

    # equal-weight portfolio rows for A and B (mean of per-symbol ledgers)
    for tag, strat in (("A", "S-A always-on"), ("B", "S-B conditional")):
        for half in halves:
            nh = halves[half].stop - halves[half].start
            agg = Ledger(nh)
            k = len(valid_syms)
            for s in valid_syms:
                led = port_paths[(tag, half, s)]
                agg.fund += led.fund / k
                agg.basis += led.basis / k
                agg.closes += led.closes / k
                agg.events += led.events        # streak: pooled events
            add_row(strat, "PORTFOLIO", half, agg)

    res = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    res.to_csv(RES / "funding_probe_full.csv", index=False,
               float_format="%.4f")

    # ── portfolio summary ────────────────────────────────────────────────────
    print("\n  PORTFOLIO SUMMARY (equal weight, %% on deployed capital)")
    print(f"  {'strategy':<18s} {'fee':<6s} | {'IN ann%':>8s} {'OUT ann%':>9s} "
          f"{'OUT net%':>9s} {'OUT gross%':>10s} {'OUT cost%':>9s} "
          f"{'cycles':>7s} {'OUT maxDD%':>10s}")
    P = res[res["scope"] == "PORTFOLIO"]
    for strat in P["strategy"].unique():
        for mode in ("taker", "maker"):
            ri = P[(P.strategy == strat) & (P.half == "IN") & (P.fee == mode)].iloc[0]
            ro = P[(P.strategy == strat) & (P.half == "OUT") & (P.fee == mode)].iloc[0]
            print(f"  {strat:<18s} {mode:<6s} | {ri['ann_dep_pct']:>+8.2f} "
                  f"{ro['ann_dep_pct']:>+9.2f} {ro['net_dep_pct']:>+9.3f} "
                  f"{ro['gross_pct']/DEPLOY:>+10.3f} {ro['cost_pct']/DEPLOY:>9.3f} "
                  f"{ro['cycles']:>7.1f} {ro['maxdd_dep_pct']:>10.2f}")

    # ── verdict: select best on IN (taker), judge its OUT ────────────────────
    pin = P[(P.half == "IN") & (P.fee == "taker")]
    best_strat = pin.sort_values("ann_dep_pct", ascending=False).iloc[0]["strategy"]
    bo_t = P[(P.strategy == best_strat) & (P.half == "OUT") & (P.fee == "taker")].iloc[0]
    bo_m = P[(P.strategy == best_strat) & (P.half == "OUT") & (P.fee == "maker")].iloc[0]
    bi_t = P[(P.strategy == best_strat) & (P.half == "IN") & (P.fee == "taker")].iloc[0]

    print("\n" + "=" * 98)
    print(f"  VERDICT — best-on-IN strategy: {best_strat}  "
          f"(IN taker {bi_t['ann_dep_pct']:+.2f}%/yr)")
    print("=" * 98)
    a = bo_t["ann_dep_pct"] >= BAR_YR
    a_m = bo_m["ann_dep_pct"] >= BAR_YR
    b = np.sign(bi_t["net_dep_pct"]) == np.sign(bo_t["net_dep_pct"]) \
        and bo_t["net_dep_pct"] != 0
    print(f"  (a) OUT annualized (taker) = {bo_t['ann_dep_pct']:+.2f}%/yr  "
          f"(bar {BAR_YR}%)      -> {'PASS' if a else 'FAIL'}"
          f"{'' if a or not a_m else '  [maker-only: ' + format(bo_m['ann_dep_pct'], '+.2f') + '%/yr]'}")
    print(f"      OUT annualized (maker) = {bo_m['ann_dep_pct']:+.2f}%/yr")
    print(f"  (b) IN sign vs OUT sign: {bi_t['net_dep_pct']:+.3f}% vs "
          f"{bo_t['net_dep_pct']:+.3f}%              -> {'PASS' if b else 'FAIL'}")

    print(f"\n  (c) DOLLAR BLOCK — {best_strat}, OUT half annualized, "
          f"per month on deployed capital:")
    for cap in (1_000, 10_000):
        mt = cap * bo_t["ann_dep_pct"] / 100 / 12
        mm = cap * bo_m["ann_dep_pct"] / 100 / 12
        print(f"      ${cap:>6,d} deployed:  taker {mt:+7.2f} $/mo   "
              f"maker {mm:+7.2f} $/mo")

    print()
    if a and b:
        print("  => PASSED the pre-registered bar. Structural carry beats "
              "costs on the OUT half.")
    elif a_m and b:
        print("  => MAKER-ONLY pass: clears the bar with maker perp entries, "
              "fails at taker. Operationally")
        print("     viable only with reliable passive fills — flag, not a "
              "full pass.")
    else:
        print("  => DID NOT PASS. Structurally alive but under the "
              "operational bar — no trade.")
    mt1 = 1_000 * bo_t["ann_dep_pct"] / 100 / 12
    mm1 = 1_000 * bo_m["ann_dep_pct"] / 100 / 12
    mt10 = 10_000 * bo_t["ann_dep_pct"] / 100 / 12
    mm10 = 10_000 * bo_m["ann_dep_pct"] / 100 / 12
    lo, hi = sorted([min(mt1, mm1), max(mt1, mm1)])
    lo10, hi10 = sorted([min(mt10, mm10), max(mt10, mm10)])
    print(f"  monthly result on deployed capital (maker/taker range): "
          f"{lo:+.1f} to {hi:+.1f} $/mo per $1k, "
          f"{lo10:+.0f} to {hi10:+.0f} $/mo per $10k")
    print(f"\n  full table -> {RES / 'funding_probe_full.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
