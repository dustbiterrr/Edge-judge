"""An empty evaluable set (no trade can be priced) must end in INSUFFICIENT -
a verdict and a report, never a traceback."""

from __future__ import annotations

import sys

import pytest

from conftest import TF_MS, make_klines, trade_log
from judge.checks import audit
from judge.ingest import load_trades


@pytest.fixture
def klines_ending_before_the_log(monkeypatch):
    """Market data stops long before the first trade: every reconstructed
    price is NaN and every trade is dropped as unpriceable."""
    def fake(symbol, t0_ms, t1_ms, progress=None):
        return make_klines(t0_ms - 48 * TF_MS, 10)
    monkeypatch.setattr("judge.checks.ensure_klines", fake)


def test_audit_returns_zero_trades_without_raising(klines_ending_before_the_log):
    tr, _ = load_trades(trade_log(n=30, with_prices=False))
    res = audit(tr, 0.11)
    assert res.n_trades == 0
    assert res.verdict == "FAIL"
    assert [c.code for c in res.checks] == ["C1", "C2", "C3", "C4", "C5"]
    assert not any(c.passed for c in res.checks)
    assert any("could not be priced" in w for w in res.warnings)


def test_cli_prints_insufficient_exit_1_and_creates_out_dir(
        klines_ending_before_the_log, tmp_path, monkeypatch, capsys):
    from judge.__main__ import main
    log = tmp_path / "log.csv"
    log.write_bytes(trade_log(n=30, with_prices=False))
    out = tmp_path / "out" / "nested" / "report.md"      # parent must be made
    monkeypatch.setattr(sys, "argv", ["judge", "audit", "--log", str(log),
                                      "--out", str(out)])
    rc = main()
    text = capsys.readouterr().out
    assert rc == 1
    assert "INSUFFICIENT" in text
    assert "Traceback" not in text
    assert out.is_file()
    assert "EdgeJudge report" in out.read_text(encoding="utf-8")


def test_cli_pass_path_exit_0_and_html_report(offline_klines, tmp_path,
                                              monkeypatch, capsys):
    from judge.__main__ import main
    log = tmp_path / "log.csv"
    log.write_bytes(trade_log(entry=100.0, exit_=100.5))
    out = tmp_path / "out" / "report.html"
    monkeypatch.setattr(sys, "argv", ["judge", "audit", "--log", str(log),
                                      "--out", str(out), "--fees", "5.5"])
    rc = main()
    text = capsys.readouterr().out
    assert rc == 0 and "PASS" in text
    assert out.read_text(encoding="utf-8").lstrip().startswith("<")
