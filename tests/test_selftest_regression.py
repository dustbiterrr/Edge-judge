"""The README self-test, pinned.

README promises that results/s02_trades_LINKUSDT_16.csv yields exactly
`FAIL - 4 of 6 checks failed`, mean -0.140%/trade, halves -0.421% / +0.217%,
n=118.  This file ties those numbers to the code (offline, on a frozen
klines fixture) AND to the README text: change either and this breaks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from judge import FEE_PROFILES
from judge.checks import audit
from judge.ingest import load_trades
from judge.report import verdict_line

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "s02_trades_LINKUSDT_16.csv"
KLINES = Path(__file__).parent / "fixtures" / "LINKUSDT_1h_klines_2026-01_2026-07.csv"
README = ROOT / "README.md"

# the campaign's own numbers - the contract the README quotes
EXPECTED = {
    "verdict": "FAIL",
    "n_trades": 118,
    "line": "FAIL — 4 of 6 checks failed (C2 fee survival: -0.140%/trade).",
    "C1": "118 trades",
    "C2": "-0.140%/trade",
    "C3": "44.9th pctl",
    "C4": "-0.421% / +0.217%",
    "C5": "1/3 regimes +",
    "C6": "0/118 trades",
}


@pytest.fixture
def campaign_klines(monkeypatch):
    """The LINKUSDT 1h klines the campaign audit saw, frozen to a CSV."""
    kl = pd.read_csv(KLINES)

    def fake(symbol, t0_ms, t1_ms, progress=None):
        assert symbol == "LINKUSDT"
        return kl
    monkeypatch.setattr("judge.checks.ensure_klines", fake)


def test_selftest_reproduces_the_campaign_numbers(campaign_klines):
    tr, _ = load_trades(LOG.read_bytes(), filename=LOG.name, symbol="LINKUSDT")
    res = audit(tr, FEE_PROFILES["binance-taker"], fee_label="binance-taker")
    got = {c.code: c.key_number for c in res.checks}
    assert res.verdict == EXPECTED["verdict"]
    assert res.n_trades == EXPECTED["n_trades"]
    assert verdict_line(res) == EXPECTED["line"]
    for code in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert got[code] == EXPECTED[code], code
    assert [c.status for c in res.checks] == \
        ["PASS", "FAIL", "FAIL", "FAIL", "FAIL", "PASS"]


def test_readme_quotes_exactly_these_numbers():
    text = README.read_text(encoding="utf-8").replace("−", "-")
    assert "yields `FAIL - 4 of 6 checks failed`" in text
    assert "mean -0.140%/trade" in text
    assert "halves\n-0.421% / +0.217%" in text or "-0.421% / +0.217%" in text
    assert "n=118" in text


def test_readme_and_code_agree_on_the_criteria_version():
    from judge.checks import CRITERIA_VERSION
    text = README.read_text(encoding="utf-8")
    assert f"criteria v{CRITERIA_VERSION}" in text
    assert "criteria v1.0.1 " not in text and "criteria v1.0," not in text
