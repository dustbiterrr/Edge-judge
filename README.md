# edge-judge

**A trade-log judge for trading strategies. Most backtests lie to their authors; this one makes the common lies visible - and says out loud which ones it cannot see.**

This repository contains the full tooling and results of a six-month campaign that set out to find directional edge on Binance perpetual futures with an RL trading system - and instead built something more useful: a judging protocol that killed every false positive it met, including the ones we badly wanted to believe.

The pipeline is strategy-agnostic. The case study is ours.

---

## The idea in one paragraph

Retail backtests fail their authors in repeatable ways: thresholds tuned on the same data that scores them, dense signals counted once per bar instead of once per trade, dozens of variants screened with no accounting for how many should pass by luck, and "edges" that are one market regime wearing a costume. Each failure mode has a cheap, mechanical countermeasure. This repo is those countermeasures, wired into one pipeline: **pre-registered thresholds, walk-forward splits, a random-walk-neutral benchmark, binomial false-pass accounting, a non-overlapping execution judge, and confirmation on untouched symbols.** A strategy is not an edge until it survives all of them.

## What the judge caught (case study)

We ran the full protocol against one system's worth of hypotheses. Every number
below is one of two kinds. **✓** reproduces from `results/` with one command,
`python scripts/campaign_numbers.py` (cells: `python scripts/count_cells.py`).
**†** was printed by a probe at run time and never written to a CSV: it is
quoted from the run and is **not reproducible from this repository**; the
script lists every one of them.

| Stage | Hypothesis | Verdict |
|---|---|---|
| RL diagnostics (1m) † | Entries carry signal | **Adverse** - entry win-rate 7–19pp *below* the random-walk-neutral benchmark. From the RL repository's diagnostics; no artifact in this one. |
| Offline price probe (1m) † | Price momentum predicts | Coin-flip; expectancy ≈ −0.11%/trade = exactly the round-trip fee. Console output of a probe not shipped here. |
| Native 15m/1h probe † | 5 generic signal families (price, flow, divergence, MTF, OI/funding), ~90 cells | 0 cells past the pre-registered +0.22% OUT threshold. `native_tf_edge_probe.py` prints and writes no CSV; the ~90 cells are not in `results/`. |
| Setup library ✓ | 13 codified setups × 10 evaluable symbols × 2 TF × 3 horizons = 780 cells (11-symbol basket, INJUSDT excluded by data validation; ADA/XRP/DOT held out for confirmation) | **5 passes** (S01 ×2, S02 ×2, S08 ×1, all 1h H=16) vs ~1.25 expected false as printed at run time† - the proxy in `campaign_numbers.py` gives ~2.4 at nominal n and ~12 once per-bar overlap is discounted. First excess of the campaign, of uncertain size. |
| Confirmation (untouched symbols) ✓ | the 3 surviving setups on ADA/XRP/DOT | S01 and S08: no cell passes. S02 vwap-fade: OUT positive on 2 of 3 symbols (XRP marginal +0.05%, DOT +0.96% with IN −0.13%, a sign flip), no cell passes the bar. **Also on the untouched symbols: S04 liq-bounce passes on DOTUSDT at 15m H=8 and H=16** - a setup that did not survive screening, passing on confirmation data; by protocol that is screening on the confirmation set, not a confirmation, and it was not pursued. Earlier versions of this README did not mention it. |
| **Non-overlap judge** ✓ | S02 vwap-fade, one-position-at-a-time | (a),(b),(c) PASS at both horizons - but **(d) FAIL: median IN −0.33%/trade vs OUT +0.33% (H=8); −0.32% vs +0.22% (H=16).** Sign flip between halves = regime artifact, not edge. Not tradable. |
| Funding/basis carry ✓ | Structural premium beats costs | **No premium in window.** All three harvest strategies negative after costs on the OUT half at taker (S-A −0.8%, S-B −8.7%, S-C −56%/yr on deployed); the best single symbol is LINKUSDT at +2.3%/yr maker, +2.1% taker, against the pre-registered 8% bar. S-A on $10k deployed: **−$5 (maker) to −$7 (taker) per month.** Basket average funding ≈ 0 and 39–56% of payments negative are run-time prints†. Regime-dependent: monitored, not traded. |
| Appendix B (pre-declared final wave) ✓ | 6 advanced setups (funding-frontrun, OI-spring, CVD-exhaustion, toxic-flow, FVG, liquidation-cascade) + cross-sectional relative value (S20) | Only **S14 reached tradable sample** → **dead, 0 of 14 evaluable cells (7 symbols × 2 H) net-positive**. The other five never reached the n-guard of 100 OUT trades: S17 produced zero signals at these bar sizes, S15/S19 at most 8/13, S18 at most 26, S16 at most 53 - the run-time diagnostic that called S16 "guard-blocked on a sign flip" is not in the CSV†. **S20: all four configs sign-flipped between halves** (momentum_k1 IN −0.30% → OUT +0.43%, flow_z_k1 IN +0.04% → OUT −0.30%). 0 confirmations. |

The non-overlap row is the whole point. The per-bar screen showed +0.63% on the best S02 cell†. Honest one-position execution plus one pre-registered criterion - *sign agreement between halves* - revealed the "edge" was April–July weather. That single line of protocol is the difference between a research note and a drawdown.

**Final state: on 1m–1h bar data for these 13 instruments in H1-2026, nothing beats the commission.** Closing arithmetic for H1-2026: **1,201 unique cells with a CSV in `results/`** - 780 setup-library + 234 untouched-symbol confirmation + 156 appendix B + 4 relative-value configs + 27 funding-carry strategy×symbol cells (`python scripts/count_cells.py`) - plus ~90 generic-probe cells that were printed and never written to CSV†. The false-pass ledger the probes printed at run time summed to ~1.4†; the CSVs do not carry the per-trade sigma it needs, so it cannot be recomputed here. **Confirmed edges: 0** - a result consistent with an honest search of a dead space, which is what validates the instrument. Directional prediction is dead everywhere we looked; the structural premium was absent in this window (regime-dependent - we monitor the funding rate, we don't trade it). What survives is the judge itself - and the map of where not to dig.

**Extended-window re-run (18.2 months, 2025-01 → 2026-07-06).** 1,174 of the H1 cells were re-tested on the long window - all 1,014 S01–S13 cells including the untouched symbols, all 156 S14–S19 cells, the 4 S20 configs - and 69 structural cells (S21–S23) were added: **1,270 unique cells campaign-wide** (`python scripts/count_cells.py`; the earlier "~2,500 cumulative" counted the re-tests as new cells). The split is a hard bull→bear cut (IN drift −11% / funding +0.35bp → OUT drift −71% / funding −0.19bp†, the most adversarial regime cut available):

- **Directional (S01–S19): the population is dead; 13 individual cells pass and none was confirmed.** Pooled OUT expectancy over the 1,012 evaluable cells is **−0.091%/trade, 95% CI [−0.101, −0.082]** ✓ - the interval excludes zero, and tripling the sample tightened it instead of revealing an edge. But 13 cells clear the pre-registered per-cell bar ✓: S06 breakout-cont ×6 (OPUSDT at H=4, 8 *and* 16 - one signal counted at three horizons - SUIUSDT ×2, NEARUSDT), S02 vwap-fade ×4 (AVAX, DOGE, LINK, SUI, all 1h H=16), S16 cvd-exhaustion ×2 (APT, DOT), S05 range-fade ×1 (ADA). An earlier version of this README said "death confirmed" and did not mention them. **How many should pass by luck is not settled by the data in this repository.** The probes' run-time false-pass ledger was never saved†; a proxy calibrated on the S02 trade logs (`campaign_numbers.py`, section H, assumptions stated there) gives **~5.3 expected at nominal trade counts - 13 would be a 2.4× excess (Poisson p ≈ 0.005 for the 11 S01–S13 cells)** - but S01–S13 counts are overlapping per-bar signals, and at the campaign's own measured episode length of 3.9 bars the expectation rises to **~23, with the 11 observed sitting below the noise floor**. The per-bar screen cannot tell those two readings apart; only the non-overlap judge can, and it was not run on the extended window. S02 vwap-fade is the setup that failed exactly that judge in H1. The S14–S19 cells are non-overlapping by construction: 2 passes against ~1.1 expected, p ≈ 0.30, noise. **Status of the 13: unconfirmed candidates.** By the campaign's own rule 5 they are not edges until they survive one-position execution and untouched data, and neither was done.
- **Structural classes, at power for the first time ✓:** relative value (S20): 274 OUT cycles per config, best OUT **+0.035%/cycle against the +0.44% bar - falsified**; the two momentum rankers flipped IN− → OUT+, flow_z_k1 flipped IN+ → OUT−, flow_z_k2 was negative in both halves. Cointegration (S22): 30 pair-cells, 4 reached the n-guard - **3 dead, 1 marginal** (OPU/DOT +0.036%/cycle, far under the bar), and their IN halves hold 1–4 cycles, too few to call a sign flip either way. Weekend-vacuum (S23): 26 cells, all insufficient - it needs a ~5-year window and remains the one honestly-untested door. S21 control: 13 cells, all insufficient, as pre-declared.
- **The mechanism that killed S02 recurs, but not everywhere.** Sign disagreement between halves - a pattern that works in one regime and breaks in the next - shows up in all four H1 S20 configs and three of four extended ones. It is *not* what the 13 directional candidates show: they pass with the same sign in both halves, which is exactly why they need the non-overlap judge rather than a paragraph.

Zero confirmed edges across 1,270 unique cells, 1,174 of them tested through two market regimes; 13 per-bar candidates the screen cannot resolve are listed above, unconfirmed. That is the result, with its remainder stated.

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
2. **No look-ahead, mechanically.** Signal at bar T uses data ≤ close(T); execution at open(T+1); rolling windows backward-only; all quantile thresholds fitted on the IN half only. This is how *our* probes were built. On *your* log the judge can verify only the decision times you declare (C6), and says UNVERIFIABLE when you declare none - see "What this is not".
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
anything. What this repository can prove about itself is in `tests/`
(`pytest -q tests`, offline, network disabled at the socket level): the fee arithmetic, date and
timestamp ingest, overlap resolution, C3/C4/C5 boundary behaviour, all
three C6 states, and a regression test that pins the self-test numbers
quoted above to the code and to this README.

## What this is not

The judge reads a CSV of closed trades. Everything it cannot see from that
CSV is listed here, because a PASS is only as strong as this list.

- **Not a profitable strategy.** The honest result of the campaign is negative, and we publish it as such.
- **Not financial advice.** It is a methodology for not fooling yourself.
- **Not exhaustive.** Real-L2 market-making and cross-venue structure were out of scope (data requirements).

- **It cannot certify pre-registration.** A trade log carries no proof that the hypothesis, the thresholds or the IN/OUT split were fixed *before* the results were seen. C4 splits whatever log it is handed in half; an author who tuned thresholds on the whole sample and submits every trade can pass C3 and C4 on pure overfit, and the judge has no way to tell. This is the campaign's blocker B1, found while dogfooding the judge on our own strategy, and it is closed by this sentence, not by a check: the only proof of pre-registration is an external, dated artifact - a commit hash, a signed document - never a statistic computed on the log. If you want a verdict that means "out-of-sample", pre-register outside the judge and submit only the trades after that date.

- **It checks look-ahead only as far as you let it, and says so.** Since v1.1.0:
  - *What is checked (C6), and on what data.* When every trade carries a `signal_time` (aliases: `decision_time`, `signal_ts`, `sig_time`), the judge verifies that each entry is strictly after its signal and outside the signal's own 1h bar; an entry at the open of a later bar is clean. It counts the offending trades, names their CSV lines and the worst case. A log whose entries were booked on the bar that produced the signal - which passed all five statistical checks at the 100th percentile before v1.1.0 - now fails C6.
  - *What is never checked.* Whether the prices or indicators behind the signal were available at `signal_time` (vendor backfill, revised data, indicators computed on the whole series); anything finer than the 1h bar; and whether the `signal_time` column is truthful. C6 verifies declared timestamps; a fabricated column passes it. Sub-hour strategies that decide and enter inside one 1h bar fail C6 by design - the judge prefers a wrong refusal to a wrong pass.
  - *What UNVERIFIABLE means, and why it is not PASS.* If the log has no `signal_time`, C6 does not run. The verdict line says so - `PASS - 5 of 5 evaluated checks passed ...; C6 look-ahead UNVERIFIABLE (no signal_time column)` - and so do the summary, the Notes and the web app. Five statistical passes with C6 unverifiable are exactly what a look-ahead-biased log produces; the judge measured that. Treat such a verdict as "the numbers hold up *if* the timing was honest, and the judge could not check the timing".

- **It does not validate your costs.** C2 re-applies the fee profile *you* pick to every trade and ignores any PnL column in your log. The same log can flip between FAIL and PASS by switching taker to maker; slippage and funding are not modelled. The profile and rate are printed next to C2 in every report so this is never hidden.
- **It cannot see selection.** How many variants you tried before this one, how many symbols you dropped, how many parameters you tuned - none of it is in a trade log. The campaign's answer to that was to count its own cells and print expected false passes next to actual ones (`scripts/campaign_numbers.py`). The judge cannot do that for you.
- **It cannot see your code or your data feed.** Bugs, look-ahead in feature construction, survivorship in the instrument list, a backfilled column - all invisible from closed trades. In our own dogfood run 16 of 19 known defects in a strategy were structurally out of the judge's reach for this reason.

## License

MIT - see [LICENSE](LICENSE).

---

*Built while hunting an edge that did not exist. The hunt produced the judge; the judge is the product.*
