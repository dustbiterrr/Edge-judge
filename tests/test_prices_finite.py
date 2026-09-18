"""A price is a finite positive number; a verdict never contains nan or inf.

Before: entry_price = 1e-320 passed the "<= 0" check, exit/entry overflowed
to inf, gross became nan, and the screen read "FAIL - 4 of 6 checks failed
(C2 fee survival: +nan%/trade)"; with --out report.html the CLI died in
matplotlib on "autodetected range of [nan, nan]".  Refusal beats a guess:
ingest refuses the cell by CSV line, checks refuse a non-finite return by
CSV line, and no chart is drawn on non-finite data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import TF_MS, make_klines, trade_log
from judge.checks import AuditResult, CheckResult, audit
from judge.ingest import IngestError, load_trades
from judge.report import (chart_bootstrap, chart_equity, chart_regimes,
                          to_html, to_markdown, verdict_line)

NAN_OR_INF = re.compile(r"\b(nan|inf)\b", re.IGNORECASE)


def log_with(entry, exit_="100.5", row: int = 3, n: int = 120) -> bytes:
    """trade_log() with one cell replaced on CSV line `row` (header = 1)."""
    lines = trade_log(n=n).decode().splitlines()
    cells = lines[row - 1].split(",")
    cells[4], cells[5] = str(entry), str(exit_)
    lines[row - 1] = ",".join(cells)
    return ("\n".join(lines) + "\n").encode()


# ── ingest: refused by CSV line ──────────────────────────────────────────────

@pytest.mark.parametrize("value, shown", [
    ("1e-320", "1e-320"), ("5e-324", "5e-324"), ("inf", "inf"),
    ("-inf", "-inf"), ("1e999", "inf"), ("0", "0"), ("-100", "-100")])
def test_unusable_price_is_refused_with_its_line(value, shown):
    with pytest.raises(IngestError) as e:
        load_trades(log_with(value, row=7))
    msg = str(e.value)
    assert "1 entry_price value cannot be a price" in msg
    assert "CSV line 7" in msg and shown in msg
    assert "finite positive number" in msg


def test_unusable_exit_price_is_refused_too():
    with pytest.raises(IngestError, match="1 exit_price value .*CSV line 5"):
        load_trades(log_with("100", exit_="-inf", row=5))


def test_many_bad_cells_are_counted_and_truncated():
    lines = trade_log(n=20).decode().splitlines()
    for r in range(2, 12):                                   # 10 bad rows
        c = lines[r - 1].split(","); c[4] = "0"; lines[r - 1] = ",".join(c)
    with pytest.raises(IngestError, match=r"10 entry_price values .*\+5 more"):
        load_trades(("\n".join(lines) + "\n").encode())


def test_nan_cell_is_blank_not_a_verdict_input(offline_klines):
    """A literal `nan` is a missing value: reconstructed from market data
    with a warning naming the line, and the verdict stays finite."""
    tr, warns = load_trades(log_with("nan", row=4))
    assert any("CSV line 4" in w and "reconstructed" in w for w in warns)
    res = audit(tr, 0.11)
    text = verdict_line(res) + " ".join(c.key_number for c in res.checks)
    assert not NAN_OR_INF.search(text)


def test_tiny_but_real_prices_are_accepted():
    """1e-8 is a real price (SHIB-class tokens); only denormals are refused."""
    tr, _ = load_trades(log_with("1e-8", exit_="1.05e-8"))
    assert float(tr.loc[tr["row"] == 3, "entry_price"].iloc[0]) == 1e-8


# ── checks: a non-finite return is a refusal, not a verdict ──────────────────

def test_overflowing_ratio_is_refused_by_checks(offline_klines):
    """1e-300 is a normal float (passes ingest) but 1e10 / 1e-300 overflows
    to inf: the return is not finite, and checks must say so by CSV line."""
    tr, _ = load_trades(log_with("1e-300", exit_="1e10", row=9))
    with pytest.raises(IngestError) as e:
        audit(tr, 0.11)
    msg = str(e.value)
    assert "CSV line 9" in msg and "not a finite number" in msg
    assert "entry 1e-300" in msg and "exit 1e+10" in msg


# ── the zero- and one-trade corners: "n/a", never nan ────────────────────────

@pytest.fixture
def klines_ending_before_the_log(monkeypatch):
    def fake(symbol, t0_ms, t1_ms, progress=None):
        return make_klines(t0_ms - 48 * TF_MS, 10)
    monkeypatch.setattr("judge.checks.ensure_klines", fake)


def test_zero_trades_prints_no_trades_not_nan(klines_ending_before_the_log):
    tr, _ = load_trades(trade_log(n=30, with_prices=False))
    res = audit(tr, 0.11)
    assert res.n_trades == 0
    by = {c.code: c for c in res.checks}
    assert by["C2"].key_number == "no trades"
    assert by["C4"].key_number == "n/a / n/a"
    text = verdict_line(res) + to_markdown(res) + to_html(res)
    assert not NAN_OR_INF.search(text)


def test_one_trade_prints_na_for_the_empty_half(offline_klines):
    tr, _ = load_trades(trade_log(n=1))
    res = audit(tr, 0.11)
    by = {c.code: c for c in res.checks}
    assert by["C4"].key_number.endswith("/ n/a")
    assert not NAN_OR_INF.search(verdict_line(res) + to_markdown(res))


# ── report: no chart on non-finite data, no traceback ────────────────────────

def _res_with(values: np.ndarray) -> AuditResult:
    t = pd.date_range("2026-01-05", periods=len(values), freq="2h", tz="UTC")
    tr = pd.DataFrame({"entry_time": t, "exit_time": t + pd.Timedelta(hours=1),
                       "net_pct": values, "regime": "FLAT"})
    rt = pd.DataFrame({"regime": ["FLAT"], "count": [len(values)],
                       "mean": [float(np.mean(values))],
                       "sum": [float(np.sum(values))]})
    return AuditResult(
        verdict="FAIL", checks=[CheckResult("C1", "Sample size", False,
                                            f"{len(values)} trades", "")],
        n_trades=len(values), n_dropped_overlap=0, reconstructed=False,
        fee_rt=0.11, trades=tr, bootstrap_totals=values,
        client_total=float(np.sum(values)),
        client_pctl=float("nan") if not np.isfinite(values).all() else 50.0,
        half_means=(float(np.mean(values)), float(np.mean(values))),
        regime_table=rt)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_charts_refuse_non_finite_data_instead_of_raising(bad):
    values = np.full(120, 0.4)
    values[7] = bad
    res = _res_with(values)
    for chart in (chart_bootstrap, chart_equity, chart_regimes):
        png = chart(res)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"          # a real image, drawn
    html = to_html(res)
    assert html.count("data:image/png;base64,") == 3


def test_charts_still_draw_finite_data():
    res = _res_with(np.full(120, 0.4))
    sizes = [len(chart(res)) for chart in (chart_bootstrap, chart_equity,
                                           chart_regimes)]
    placeholder = len(chart_bootstrap(_res_with(np.full(120, np.nan))))
    assert all(s != placeholder for s in sizes)


# ── CLI: the html path that used to die in matplotlib ────────────────────────

def test_cli_html_on_denormal_price_is_one_error_line(tmp_path, monkeypatch,
                                                      capsys):
    from judge.__main__ import EXIT_ERROR, main
    monkeypatch.chdir(tmp_path)
    Path("t.csv").write_bytes(log_with("1e-320"))
    rc = main(["audit", "--log", "t.csv", "--out", "report.html"])
    text = capsys.readouterr().out
    assert rc == EXIT_ERROR
    assert text.startswith("ERROR: 1 entry_price value cannot be a price")
    assert "Traceback" not in text and not NAN_OR_INF.search(text)
    assert not Path("report.html").exists()
