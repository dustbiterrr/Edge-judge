# edge-judge

**A falsification pipeline for trading strategies. Most backtests lie to their authors; this one makes lying hard.**

This repository contains the full tooling and results of a six-month campaign that set out to find directional edge on Binance perpetual futures with an RL trading system - and instead built something more useful: a judging protocol that killed every false positive it met, including the ones we badly wanted to believe.

The pipeline is strategy-agnostic. The case study is ours.

---

## The idea in one paragraph

Retail backtests fail their authors in repeatable ways: thresholds tuned on the same data that scores them, dense signals counted once per bar instead of once per trade, dozens of variants screened with no accounting for how many should pass by luck, and "edges" that are one market regime wearing a costume. Each failure mode has a cheap, mechanical countermeasure. This repo is those countermeasures, wired into one pipeline: **pre-registered thresholds, walk-forward splits, a random-walk-neutral benchmark, binomial false-pass accounting, a non-overlapping execution judge, and confirmation on untouched symbols.** A strategy is not an edge until it survives all of them.

## What the judge caught (case study)

We ran the full protocol against one system's worth of hypotheses. Every number below is from the CSVs in `results/`.

| Stage | Hypothesis | Verdict |
|---|---|---|
| RL diagnostics (1m) | Entries carry signal | **Adverse** - entry win-rate 7–19pp *below* the random-walk-neutral benchmark |
| Offline price probe (1m) | Price momentum predicts | Coin-flip; expectancy ≈ −0.11%/trade = exactly the round-trip fee |
| Native 15m/1h probe | 5 generic signal families (price, flow, divergence, MTF, OI/funding) | 0 cells past the pre-registered +0.22% OUT threshold |
| Setup library | 13 codified setups × 10 evaluable symbols × 2 TF × 3 horizons = 780 cells (11-symbol basket, INJUSDT excluded by data validation; ADA/XRP/DOT held out for confirmation) | 5 passes vs ~1.25 expected false - first real excess of the campaign |
| Confirmation (untouched symbols) | 3 surviving setups | 2 killed (S01, S08); S02 vwap-fade survives 2/3 |
| **Non-overlap judge** | S02 vwap-fade, one-position-at-a-time | (a),(b),(c) PASS - but **(d) FAIL: IN-half −0.33%/trade vs OUT +0.24%.** Sign flip between halves = regime artifact, not edge. Not tradable. |
| Funding/basis carry | Structural premium beats costs | **No premium in window** - basket avg funding ≈ 0 across H1-2026 (39–56% of payments negative); best symbol +3.1%/yr on deployed vs the pre-registered 8% bar; all three harvest strategies negative after costs (−0.8% to −56%/yr). On $10k deployed: **−$5 to −$7/month.** Regime-dependent: monitored, not traded. |
| Appendix B (pre-declared final wave) | 6 advanced setups (funding-frontrun, OI-spring, CVD-exhaustion, toxic-flow, FVG, liquidation-cascade) + cross-sectional relative value (S20) | Only **S14 reached tradable sample** → **dead, 0/7 symbols net-positive** (best −0.04%). The other five never cleared the n-guard: S15/S18/S19 too rare by construction, S17 near-zero signals at these bar sizes, **S16 guard-blocked on the same IN/OUT sign-flip that killed S02 - caught *before* judgment.** **S20: both rankers sign-flipped between halves** (momentum IN −0.30% → OUT +0.43%; flow the exact mirror). 0 confirmations. |

The last row is the whole point. A per-bar backtest showed +0.63% on the best cell. Honest one-position execution plus one pre-registered criterion - *sign agreement between halves* - revealed the "edge" was April–July weather. That single line of protocol is the difference between a research note and a drawdown.

**Final state: on 1m–1h bar data for these 13 instruments in H1-2026, nothing beats the commission.** Closing arithmetic: **~1,200 cells screened across all waves, ~1.4 false passes expected by chance, confirmed edges: 0** - a result statistically indistinguishable from an honest search of a dead space, which is exactly what validates the instrument. Directional prediction is dead everywhere we looked; the structural premium was absent in this window (regime-dependent - we monitor the funding rate, we don't trade it). What survives is the judge itself - and the map of where not to dig.

**Extended-window confirmation (18.2 months, 2025-01 → 2026-07, ~2,500 cumulative cells).** Re-run across a hard bull→bear walk-forward split (IN drift −11% / funding +0.35bp → OUT drift −71% / funding −0.19bp - the most adversarial regime cut available):

- **Directional (S01–S19): death confirmed, tighter.** Pooled OUT expectancy **−0.091%/trade, 95% CI [−0.101, −0.082]** - the interval now excludes zero. Doubling the sample narrowed the confidence bound instead of revealing an edge, exactly as pre-registered.
- **Structural classes, tested at power for the first time:** relative-value (S20) upgraded from *insufficient* to **falsified** (274 cycles, best +0.035% ≪ +0.44% bar); cointegration (S22) - 4 pairs reached sample, all **FAIL**; weekend-vacuum (S23) still under-powered (needs ~5-year window - the one honestly-undertested door).
- **One mechanism, every level.** S20 and all four evaluable S22 cells died by the *same* IN+ → OUT− sign flip that killed S02: mean-reversion and spread strategies that worked in the bullish first half broke in the bearish second. The campaign didn't just find zero - it measured *why*: patterns that work in one regime don't survive the regime change.

Zero edges across ~2,500 cumulative cells, through two market regimes, at statistical power. That is not fatigue - it is a result.

## Repository layout

`results/` is the campaign's evidence base and is treated as immutable: nothing in Quick start writes there - the judge and the S02 script write to `out/` (gitignored), so a clean clone stays clean.
The research probes that regenerate campaign CSVs (`setup_probe`, `appendix_*`, `s20_*`, `funding_*`, `extended_window_report`) do write into `results/` by design - run them only for a full reproduction.

```
edge-judge/
├── judge/                          # the judge as a library - criteria v1.1.0 (C1-C5 frozen since v1.0)
│   ├── ingest.py                   #   trade-log CSV -> validated trades, human errors
│   ├── overlap.py                  #   one-position-at-a-time resolution
│   ├── marketdata.py               #   1h klines, cache-first, archive dumps only
│   ├── checks.py                   #   the six pre-registered checks; C6 is UNVERIFIABLE without signal_time
│   ├── report.py                   #   verdict line, md / self-contained html, charts
│   └── __main__.py                 #   CLI entry: python -m judge audit ...
├── app.py                          # Streamlit web app (upload / demo verdict)
├── scripts/
│   ├── fetch_binance_native.py     # klines + flow (taker_buy) + OI + funding, validated
│   ├── setup_library.py            # codified setups: S01-S13, Appendix B S14-S19, appendix C S21/S23 (+S24 closure)
│   ├── native_tf_edge_probe.py     # generic signal families, walk-forward
│   ├── setup_probe.py              # setup × symbol × TF × horizon grid + false-pass accounting
│   ├── s02_nonoverlap_backtest.py  # one-position-at-a-time execution judge
│   ├── appendix_b_probe.py         # pre-declared final single-asset wave
│   ├── appendix_c_statarb.py       # structural classes: S22 cointegration, S23 weekend, S21 control
│   ├── s20_relative_value.py       # cross-sectional dollar-neutral protocol
│   ├── funding_carry_probe.py      # structural carry, cost-honest, USD verdict
│   └── extended_window_report.py   # 18.2mo re-run orchestrator (regime split + merged verdict)
├── results/                        # full CSVs from the campaign (immutable evidence)
├── tests/                          # offline unit tests (pytest, network disabled)
├── out/                            # gitignored: everything Quick start writes
└── LICENSE                         # MIT
```

## The rules (non-negotiable)

1. **Pre-register or it didn't happen.** Thresholds, sample guards, and confirmation criteria are written down *before* the run. Ours: OUT expectancy > 2× round-trip fee, IN/OUT sign agreement, n ≥ 100.
2. **No look-ahead, mechanically.** Signal at bar T uses data ≤ close(T); execution at open(T+1); rolling windows backward-only; all quantile thresholds fitted on the IN half only.
3. **Count what a trader gets, not what a bar shows.** Dense signals are judged one-position-at-a-time. Per-bar expectancy on an overlapping signal measures episode shape, not tradable edge (our stretch episodes averaged 3.9 bars - per-bar counting scored each ~4×, biased toward the deepest bars).
4. **Account for luck.** Screening N cells means some pass by chance. Print expected false passes next to actual passes, always.
5. **Confirm on untouched data.** Symbols/periods never seen during screening. Survivors of the grid died here - that is the system working.
6. **Agent reports are text, artifacts are evidence.** Every run must leave CSVs and trade logs. We caught our own coding agent embellishing twice; the files didn't lie.

## Quick start

Requires Python 3.10+ (PEP 604 unions). Network: first runs download public
Binance archive dumps from data.binance.vision into `data/` (gitignored) -
the CLI self-test below pulls ~0.25 MB, the one-month fetch ~2 MB.

Every command below is acceptance-tested on this repo - if it is printed
here, it ran. Run together on a clean clone they leave `git status` empty.

```bash
pip install -r requirements.txt

# market data (cache-first: only missing months are downloaded)
python scripts/fetch_binance_native.py --symbol ETHUSDT --months 1

# audit a trade log with the frozen criteria
python -m judge audit --log results/s02_trades_LINKUSDT_16.csv --symbol LINKUSDT --out out/report.md

# the campaign-form one-position-at-a-time judge (writes to out/)
python scripts/s02_nonoverlap_backtest.py

# offline unit tests (network disabled at the socket level)
pytest -q tests

# the web app
streamlit run app.py
```

**Warning:** `python scripts/s02_nonoverlap_backtest.py --outdir results`
regenerates the campaign trade logs *in place* and overwrites the evidence
base. Use it only for a full reproduction after fetching all 13 symbols for
the campaign window - on partial data it replaces real logs with partial ones.

To judge **your** strategy: export its trades to CSV (`symbol, side,
entry_time, exit_time, [signal_time, entry_price, exit_price, qty]`) and feed
it to the CLI or the web app. Include `signal_time` - the moment each
decision was made - or the look-ahead check (C6) is reported UNVERIFIABLE:
a log that entered on the bar that produced its signal passes the five
statistical checks untouched, and the judge will say so rather than PASS
it quietly. To put a *rule* through the research grid instead:
implement it as a `bars -> {-1,0,+1}` function in `scripts/setup_library.py`,
add it to the registry, and let `scripts/setup_probe.py` run. If it survives,
you have something. Ours didn't - and knowing that cost us a GPU budget
instead of a deposit.

## CLI

```bash
python -m judge audit --log trades.csv --out out/report.md \
    [--fees binance-taker|binance-maker|<bps-per-side>] [--symbol ETHUSDT] \
    [--date-format DMY|MDY]
```

Reads a trade-log CSV (never your code), runs the six pre-registered checks
(criteria v1.1.0 - not configurable from any flag; the only inputs besides
the log are the fee profile, a property of your exchange, and the day/month
order for slash-style dates), and writes a markdown or self-contained HTML
report. Exit code 0 = PASS, 1 = FAIL. Every check has three states: PASS,
FAIL, UNVERIFIABLE - and an unverifiable check is named in the verdict
line (`PASS - 5 of 5 evaluated checks passed ...; C6 look-ahead
UNVERIFIABLE (no signal_time column)`), never folded into a pass.

Slash-style dates (`03/02/2026`) are read under one day/month rule for the
whole file, inferred only when a value forces it; an all-ambiguous file is
refused unless `--date-format` states the rule.

v1.1.0 adds C6 look-ahead: `signal_time` must be strictly before
`entry_time` and outside the signal's own 1h bar (entering at the open of a
later bar is clean). v1.0.1 fixed a regime-lookup lookahead affecting
mid-bar entries; C1-C5 are unchanged since v1.0.

Self-test: the judge reproduces the campaign's own FAIL on its own logs -
`results/s02_trades_LINKUSDT_16.csv` yields `FAIL - 4 of 6 checks failed`,
with the same numbers the campaign recorded (mean −0.140%/trade, halves
−0.421% / +0.217%, n=118).

## Web app

```bash
streamlit run app.py
```

One screen: drag a CSV, pick a fee profile, get the verdict - criteria
table, three charts (coin-flip bootstrap, non-overlap equity with IN/OUT
halves, PnL by market regime), and downloadable md/html reports. The
**"Try the demo verdict"** button audits our own S02 trade log, so a visitor
sees a real FAIL on a real strategy in ~30 seconds without uploading
anything. Release acceptance (a 15-check suite, including byte-identical
numbers between the demo button and the CLI on the same input) runs in the
development repo before each release; this repository ships the offline
unit tests in `tests/` (`pytest -q tests`).

## What this is not

- Not a profitable strategy. The honest result of the campaign is negative, and we publish it as such.
- Not financial advice. It is a methodology for not fooling yourself.
- Not exhaustive. Real-L2 market-making and cross-venue structure were out of scope (data requirements).
- Not a view into your code or data feed. The judge sees a trade log, nothing else: C6 checks the decision times you declare, and without them it says UNVERIFIABLE rather than guessing. It cannot certify pre-registration, walk-forward provenance, or that your prices came from data available at the time.

## License

MIT - see [LICENSE](LICENSE).

---

*Built while hunting an edge that did not exist. The hunt produced the judge; the judge is the product.*
