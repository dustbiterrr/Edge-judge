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


def test_readme_expected_output_block_is_the_cli_output(campaign_klines,
                                                        tmp_path, monkeypatch,
                                                        capsys):
    """The README block under "Output of step 1" is compared line by line
    with what `python -m judge audit ...` prints.  Offline the seven
    `fetching` lines do not appear (the months come from the fixture, not
    the archive), so they are the one thing dropped from both sides; the
    README is still checked to list exactly the seven months a first run
    downloads.  Change the output or the block and this breaks."""
    from judge.__main__ import main
    text = README.read_text(encoding="utf-8")
    head, _, rest = text.partition("Output of step 1, byte for byte")
    block = rest.split("```\n", 2)[1].rstrip("\n").split("\n")
    fetch = [l for l in block if l.startswith("  .. fetching ")]
    assert fetch == [f"  .. fetching LINKUSDT 2026-{m:02d} ({m}/7)"
                     for m in range(1, 8)]
    expected = [l for l in block if not l.startswith("  .. fetching ")]

    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    log = tmp_path / "results" / LOG.name
    log.write_bytes(LOG.read_bytes())
    rc = main(["audit", "--log", f"results/{LOG.name}", "--symbol",
               "LINKUSDT", "--out", "out/report.md"])
    got = capsys.readouterr().out.rstrip("\n").split("\n")
    assert rc == 1
    assert got == expected
