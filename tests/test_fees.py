"""Fee mathematics: parse_fees (CLI) and C2 fee survival (checks).
Round-trip fee in % = bps per side * 2 / 100; net = gross - fee_rt."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import trade_log
from judge import FEE_PROFILES
from judge.__main__ import parse_fees
from judge.checks import audit
from judge.ingest import load_trades


def test_named_profiles_match_the_frozen_table():
    assert parse_fees("binance-taker") == FEE_PROFILES["binance-taker"] == 0.11
    assert parse_fees("binance-maker") == FEE_PROFILES["binance-maker"] == 0.04


def test_bps_per_side_converts_to_round_trip_percent():
    assert parse_fees("5.5") == pytest.approx(0.11)     # taker, spelled out
    assert parse_fees("2") == pytest.approx(0.04)       # maker, spelled out
    assert parse_fees("10") == pytest.approx(0.20)


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "nan", "inf", ""])
def test_invalid_fee_values_exit_with_a_message(bad):
    with pytest.raises(SystemExit):
        parse_fees(bad)


def _c2(res):
    c = res.checks[1]
    assert c.code == "C2"
    return c


def test_c2_net_equals_gross_minus_round_trip_fee(offline_klines):
    tr, _ = load_trades(trade_log(entry=100.0, exit_=100.5))    # +0.5% gross
    res = audit(tr, 0.11)
    assert np.allclose(res.trades["gross_pct"], 0.5)
    assert np.allclose(res.trades["net_pct"], 0.5 - 0.11)
    c2 = _c2(res)
    assert c2.passed and c2.key_number == "+0.390%/trade"


def test_c2_fails_when_the_fee_eats_the_edge(offline_klines):
    tr, _ = load_trades(trade_log(entry=100.0, exit_=100.1))    # +0.1% gross
    res = audit(tr, 0.11)
    c2 = _c2(res)
    assert not c2.passed and c2.key_number == "-0.010%/trade"
    assert res.verdict == "FAIL"


def test_c2_uses_the_fee_it_is_given(offline_klines):
    tr, _ = load_trades(trade_log(entry=100.0, exit_=100.1))
    assert _c2(audit(tr, 0.04)).passed          # maker: 0.1 - 0.04 > 0
    assert not _c2(audit(tr, 0.11)).passed      # taker: 0.1 - 0.11 < 0


def test_short_side_sign_is_applied_before_the_fee(offline_klines):
    tr, _ = load_trades(trade_log(entry=100.0, exit_=99.0, side="short"))
    res = audit(tr, 0.11)
    assert np.allclose(res.trades["gross_pct"], 1.0)
    assert np.allclose(res.trades["net_pct"], 0.89)
