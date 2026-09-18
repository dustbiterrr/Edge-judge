# Changelog

Behavioural changes to the judge and to what the repository claims about
itself. Criteria C1–C5 are frozen since v1.0; a version bump means the
judge answers differently on some input, or the documentation stops
saying something it could not back.

## 1.1.0 — 2026-09

**Principle: refusal beats a guess.** Where the judge cannot check
something it now says so, in the verdict line, the summary, the Notes and
the web app - never silently as a pass.

### The judge

- **C6 Look-ahead (new).** With a `signal_time` column (aliases
  `decision_time`, `signal_ts`, `signal_timestamp`, `decision_ts`,
  `sig_time`, `signal_at`, `decided_at`) the judge verifies that every
  entry is strictly after its signal and outside the signal's own 1h bar;
  an entry at the open of a later bar is clean. FAIL counts the offending
  trades, names their CSV lines and the worst case. A log that booked its
  entries on the bar that produced the signal passed all five statistical
  checks at the 100th percentile in 1.0.1; it fails C6 in 1.1.0.
- **UNVERIFIABLE (new state).** Without `signal_time`, C6 does not run
  and is reported UNVERIFIABLE - a third state next to PASS and FAIL. The
  verdict line names it (`PASS — 5 of 5 evaluated checks passed …; C6
  look-ahead UNVERIFIABLE (no signal_time column)`), the "What this
  means" text says look-ahead could not be checked, the Notes carry it,
  the CLI prints the reason under the check, the web app shows an amber
  warning and a third badge. A PASS with C6 unverifiable no longer says
  the pattern is one "a coin flip does not explain".
- **Verdict line counts evaluated checks.** The campaign self-test now
  reads `FAIL — 4 of 6 checks failed` (its log carries `signal_time`, so
  C6 is evaluated and passes); a log without `signal_time` reads
  `… of 5 checks …; C6 look-ahead UNVERIFIABLE`.
- **What a PASS means** is printed under every PASS in the CLI, the
  reports and the web app: what it establishes, what it does not, and
  that it is read with the scope list and the state of C6.
- **C2 names its inputs.** The fee profile and rate are printed next to
  C2 everywhere, with the line "costs re-computed at <profile> <rate>
  round trip on every trade; this does not validate the author's own
  cost assumptions". Behaviour unchanged: the same log still flips
  between FAIL and PASS when the profile changes, and now says so.

### Ingest

- **One day/month rule per file.** Slash-style dates (`03/02/2026`) were
  parsed per element: `03/02` became 2 March and `13/02` became 13
  February in the same column, silently. Now the rule is inferred once
  for the whole file and only when a value forces it (a field > 12),
  applied to every row and verified row by row; an all-ambiguous file is
  refused unless `--date-format DMY|MDY` (CLI) or the date selector (web
  app) states the rule; a file that forces both rules is refused. The
  rule never touches ISO or epoch values.
- `signal_time` is a recognised optional column; a partial column is
  refused (fill every row or drop the column).
- Every trade keeps its CSV line number so checks can name lines.
- The download template carries `signal_time`.
- The warning for a blank or non-numeric price cell says which side, how
  many rows, which lines, and that provided prices are kept; it no
  longer says "No prices in the log" when one side was provided.

### Repository and documentation

- `results/synthetic_profitable.csv` - a fabricated log with constant
  prices that passes C1–C5 - moved to
  `tests/fixtures/synthetic_KNOWN_FAKE_do_not_cite.csv` with a README
  and a test; `results/` holds only campaign artifacts.
- `scripts/count_cells.py` defines a cell and counts unique cells from
  `results/` (1,270 campaign-wide; 1,201 in H1-2026; 1,174 re-tested on
  the extended window; 69 added). "~2,500 cumulative cells" is gone: it
  counted re-tests as new cells.
- `scripts/campaign_numbers.py` reproduces every campaign number the
  README quotes from `results/`, lists the numbers that exist only in a
  probe's console output, and computes a false-pass proxy with stated
  assumptions in place of the run-time ledger that was never saved.
- README states what earlier versions rounded past: 13 extended-window
  cells pass the per-cell bar (unconfirmed candidates; the per-bar
  screen cannot resolve whether they exceed the noise floor); S22 has 3
  dead and 1 marginal, not "all FAIL"; S04 liq-bounce passed twice on
  the confirmation symbols; the best funding symbol is +2.3%/yr, not
  +3.1%; S16 was under the n-guard, not "guard-blocked on a sign flip".
  Reproducible numbers are marked ✓, console-only ones †.
- README "What this is not" now says the judge cannot certify
  pre-registration, states the look-ahead boundary in three parts
  (what is checked and on what data, what is never checked, what
  UNVERIFIABLE means), and that costs, selection, code and data feed are
  invisible. The tagline no longer says "makes lying hard".
- README leads with the tool (install, first verdict, the six checks, a
  legend for the output, "Does PASS mean I have an edge?") and puts the
  campaign below a rule. The 15-check acceptance suite is no longer
  cited; `tests/` is what the repository can prove about itself.
- `s02_nonoverlap_backtest.py` labels a run on fewer than 13 symbols a
  partial run, not a verdict.
- Every research probe writes to `out/` by default; `results/` is written
  only with an explicit `--outdir results` (`--out results/...` for
  `setup_probe.py`), and the script names the overwrite when it happens.
  Before: `setup_probe.py` run as the README described - one symbol
  fetched - replaced the 780-row campaign table with a 78-row partial run
  and printed a verdict on it. It now labels such a run PARTIAL RUN and
  prints no verdict; with no data it refuses and writes nothing.
  `tests/test_scripts_outdir.py` pins this for all seven scripts.
- `requirements.txt` pins the versions the release was verified against
  (numpy 2.5.3, pandas 3.0.5, pyarrow 25.0.1, matplotlib 3.11.2,
  streamlit 1.64.0, pytest 9.1.1).
- Tests: 87, offline. New: dates (16), C6 (16), C3/C4/C5 boundaries
  (25), the self-test regression pinned to README, the labelled fake.

## 1.0.1 — 2026-07

- C5 regime lookup anchored on the last bar *closed* at or before the
  entry; the previous form took the last bar *opened* before it, whose
  close could lie after a mid-bar entry - a look-ahead in the judge's
  own regime tagging. Verdicts on bar-aligned logs unchanged.
- Ingest hardening: epoch timestamps by magnitude, future exits and
  non-positive prices refused; market-data cache continuity and
  listing-day handling; no tracebacks reach the web app.

## 1.0 — 2026-07

- Criteria C1–C5 frozen from the H1-2026 campaign; CLI, markdown/HTML
  reports, Streamlit app.
