"""C3 (coin-flip bootstrap), C4 (half-vs-half stability), C5 (regime
robustness): the three checks that carry the verdict's statistical weight.
Boundary cases, degenerate input, small samples.  All offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import TF_MS, make_klines
from judge.checks import (BOOT_PCTL, REGIME_MIN_TRADES, RNG_SEED, audit)
from judge.ingest import load_trades

T0 = pd.Timestamp("2026-01-05 00:00", tz="UTC")
H = pd.Timedelta(hours=1)
FEE = 0.11


def log(gross: list[float], start: pd.Timestamp = T0,
        spacing_h: int = 2) -> bytes:
    """One ETHUSDT long per gross % value, 1h long, spacing_h apart."""
    rows = ["symbol,side,entry_time,exit_time,entry_price,exit_price"]
    for i, g in enumerate(gross):
        e = start + i * spacing_h * H
        x = e + H
        rows.append(f"ETHUSDT,long,{e:%Y-%m-%d %H:%M},{x:%Y-%m-%d %H:%M},"
                    f"100.0,{100.0 * (1 + g / 100):.6f}")
    return ("\n".join(rows) + "\n").encode()


def run(raw: bytes):
    tr, _ = load_trades(raw)
    return audit(tr, FEE)


def check(res, code):
    c = next(c for c in res.checks if c.code == code)
    return c


# ═════════════════════════════════════════════════════════════════════════════
# C3 — beats coin-flip
# ═════════════════════════════════════════════════════════════════════════════

def test_c3_consistent_wins_sit_at_the_top_percentile(offline_klines):
    res = run(log([0.5] * 200))
    c = check(res, "C3")
    assert c.passed and c.key_number == "100.0th pctl"
    assert res.client_pctl == 100.0
    assert res.bootstrap_totals.shape == (1000,)


def test_c3_symmetric_wins_and_losses_are_a_coin_flip(offline_klines):
    """Alternating +/-0.5%: direction carries no information."""
    res = run(log([0.5, -0.5] * 100))
    c = check(res, "C3")
    assert not c.passed
    assert 20.0 < res.client_pctl < 80.0
    assert "coin" in c.detail.lower()


def test_c3_all_zero_gross_is_degenerate_but_does_not_crash(offline_klines):
    """Every bootstrap equals the client total; strictly-below fraction = 0."""
    res = run(log([0.0] * 120))
    c = check(res, "C3")
    assert not c.passed and c.key_number == "0.0th pctl"
    assert np.allclose(res.bootstrap_totals, res.client_total)


def test_c3_is_deterministic_for_identical_input(offline_klines):
    rng = np.random.default_rng(7)
    g = list(rng.normal(0.2, 0.8, 150))
    a, b = run(log(g)), run(log(g))
    assert a.client_pctl == b.client_pctl
    assert np.array_equal(a.bootstrap_totals, b.bootstrap_totals)
    assert RNG_SEED == 20260706          # the seed is part of the contract


def test_c3_threshold_is_the_frozen_97_5(offline_klines):
    assert BOOT_PCTL == 97.5
    res = run(log([0.5] * 200))
    assert check(res, "C3").passed is (res.client_pctl >= 97.5)


def test_c3_on_a_single_trade_cannot_pass(offline_klines):
    """n=1: half of the sign-flips beat it, half do not -> ~50th pctl."""
    res = run(log([2.0]))
    c = check(res, "C3")
    assert not c.passed
    assert 30.0 < res.client_pctl < 70.0


def test_c3_fee_is_inside_the_bootstrap(offline_klines):
    """Coin-flip logs pay the same fee: with zero gross the client cannot
    beat them, whatever the fee."""
    tr, _ = load_trades(log([0.0] * 120))
    for fee in (0.0, 0.11, 0.5):
        res = audit(tr, fee)
        assert not check(res, "C3").passed


# ═════════════════════════════════════════════════════════════════════════════
# C4 — stability across halves (split by entry_time midpoint)
# ═════════════════════════════════════════════════════════════════════════════

def test_c4_passes_when_both_halves_are_positive(offline_klines):
    res = run(log([0.5] * 60 + [0.3] * 60))
    c = check(res, "C4")
    assert c.passed and c.key_number == "+0.390% / +0.190%"
    assert res.half_means == pytest.approx((0.39, 0.19))


def test_c4_sign_flip_fails_and_names_it(offline_klines):
    res = run(log([-0.5] * 60 + [0.8] * 60))
    c = check(res, "C4")
    assert not c.passed
    assert "sign flip" in c.detail


def test_c4_positive_then_negative_fails(offline_klines):
    res = run(log([0.8] * 60 + [-0.5] * 60))
    assert not check(res, "C4").passed


def test_c4_both_negative_fails_second_half_must_be_positive(offline_klines):
    res = run(log([-0.5] * 60 + [-0.2] * 60))
    c = check(res, "C4")
    assert not c.passed and c.key_number == "-0.610% / -0.310%"


def test_c4_splits_by_time_not_by_count(offline_klines):
    """10 trades in one day, 10 trades a month later: the midpoint in time
    puts the whole first cluster in half one."""
    raw = log([0.5] * 10) + b"".join(
        line + b"\n" for line in
        log([-0.4] * 10, start=T0 + pd.Timedelta(days=30)).splitlines()[1:])
    res = run(raw)
    assert res.half_means == pytest.approx((0.39, -0.51))
    assert not check(res, "C4").passed


def test_c4_single_trade_has_an_empty_second_half_and_fails(offline_klines):
    res = run(log([2.0]))
    c = check(res, "C4")
    assert not c.passed
    assert np.isnan(res.half_means[1])


def test_c4_two_trades_split_one_each(offline_klines):
    res = run(log([0.5, 0.7]))
    assert res.half_means == pytest.approx((0.39, 0.59))
    assert check(res, "C4").passed


# ═════════════════════════════════════════════════════════════════════════════
# C5 — regime robustness (UP / DOWN / FLAT by 24h drift at entry)
# ═════════════════════════════════════════════════════════════════════════════

DAY = 24


@pytest.fixture
def regime_klines(monkeypatch):
    """Price path: days 0-6 flat (FLAT), days 7-12 +0.1%/bar (24h drift
    +2.4% -> UP), days 13-18 -0.1%/bar (DOWN).  Starts 48h before T0."""
    n = 20 * DAY
    step = np.ones(n)
    step[7 * DAY:13 * DAY] = 1.001
    step[13 * DAY:] = 0.999
    close = 100.0 * np.cumprod(step)
    t0 = int((T0 - 48 * H).timestamp() * 1000)
    kl = make_klines(t0, n)
    for col in ("open", "high", "low", "close"):
        kl[col] = close
    monkeypatch.setattr("judge.checks.ensure_klines",
                        lambda symbol, a, b, progress=None: kl)
    return kl


def _at_day(day: int) -> pd.Timestamp:
    return T0 - 48 * H + day * DAY * H


def test_c5_all_flat_single_regime_positive_passes(offline_klines):
    res = run(log([0.5] * 100))
    c = check(res, "C5")
    assert c.passed and c.key_number == "1/1 regimes +"
    assert set(res.trades["regime"]) == {"FLAT"}


def test_c5_fewer_than_ten_trades_means_no_qualifying_regime(offline_klines):
    res = run(log([2.0] * (REGIME_MIN_TRADES - 1)))
    c = check(res, "C5")
    assert not c.passed and c.key_number == "no qualifying regime"


def test_c5_exactly_ten_trades_qualify(offline_klines):
    res = run(log([2.0] * REGIME_MIN_TRADES))
    assert check(res, "C5").key_number == "1/1 regimes +"


def test_c5_tags_up_flat_and_down_from_the_price_path(regime_klines):
    raw = (log([0.5] * 12, start=_at_day(3))
           + log([0.5] * 12, start=_at_day(9))[len("symbol,side,entry_time,exit_time,entry_price,exit_price\n"):]
           + log([0.5] * 12, start=_at_day(15))[len("symbol,side,entry_time,exit_time,entry_price,exit_price\n"):])
    res = run(raw)
    tags = res.trades.groupby("regime").size().to_dict()
    assert tags == {"FLAT": 12, "UP": 12, "DOWN": 12}


def _three_regime_log(flat: float, up: float, down: float) -> bytes:
    head = "symbol,side,entry_time,exit_time,entry_price,exit_price\n"
    return (log([flat] * 12, start=_at_day(3))
            + log([up] * 12, start=_at_day(9))[len(head):]
            + log([down] * 12, start=_at_day(15))[len(head):])


def test_c5_two_of_three_regimes_positive_passes_the_60_percent_bar(regime_klines):
    res = run(_three_regime_log(flat=0.5, up=0.5, down=-1.0))
    c = check(res, "C5")
    assert c.passed and c.key_number == "2/3 regimes +"


def test_c5_one_of_three_regimes_positive_fails(regime_klines):
    res = run(_three_regime_log(flat=0.5, up=-1.0, down=-1.0))
    c = check(res, "C5")
    assert not c.passed and c.key_number == "1/3 regimes +"


def test_c5_a_regime_below_ten_trades_is_ignored_not_counted(regime_klines):
    """12 FLAT winners + 5 DOWN losers: DOWN never qualifies, so 1/1."""
    head = "symbol,side,entry_time,exit_time,entry_price,exit_price\n"
    raw = (log([0.5] * 12, start=_at_day(3))
           + log([-2.0] * 5, start=_at_day(15))[len(head):])
    res = run(raw)
    assert check(res, "C5").key_number == "1/1 regimes +"


def test_c5_trades_without_24h_of_history_are_tagged_na_and_excluded(monkeypatch):
    """Klines start only 3h before the first trade: drift is NaN -> 'NA',
    and NA never forms a regime."""
    t0 = int((T0 - 3 * H).timestamp() * 1000)
    kl = make_klines(t0, 400)
    monkeypatch.setattr("judge.checks.ensure_klines",
                        lambda s, a, b, progress=None: kl)
    res = run(log([0.5] * 12))
    assert (res.trades["regime"] == "NA").sum() > 0
    assert "NA" not in set(res.regime_table["regime"])
