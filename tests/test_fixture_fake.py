"""The known-fake fixture documents a boundary, not a result: a fabricated
log with constant wins passes C1-C5 by construction, and the judge's only
honest answer is C6 UNVERIFIABLE (the file carries no signal_time)."""

from __future__ import annotations

from pathlib import Path

from judge.checks import UNVERIFIABLE, audit
from judge.ingest import load_trades
from judge.report import verdict_line

FAKE = Path(__file__).parent / "fixtures" / "synthetic_KNOWN_FAKE_do_not_cite.csv"


def test_fake_log_passes_c1_to_c5_and_c6_is_unverifiable(offline_klines):
    tr, _ = load_trades(FAKE.read_bytes(), filename=FAKE.name)
    res = audit(tr, 0.11)
    assert res.n_trades == 240
    assert all(c.passed for c in res.checks[:5])
    assert res.checks[5].status == UNVERIFIABLE
    assert "C6 look-ahead UNVERIFIABLE" in verdict_line(res)


def test_fake_log_is_not_in_the_evidence_base():
    assert not (Path(__file__).parent.parent / "results"
                / "synthetic_profitable.csv").exists()
    assert FAKE.exists()
