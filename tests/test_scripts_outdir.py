"""results/ is the campaign's evidence base and is treated as immutable.

Every research probe under scripts/ used to write there by default, so the
README's own "put your rule through the grid" instruction - fetch one symbol,
run setup_probe.py - replaced the 780-row campaign table with a 78-row
partial run.  These tests pin the fix: with default arguments no script
touches results/; everything lands under out/; the evidence base is written
only when asked for by name, and the script says so."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# (script, extra args) - default arguments, exactly as a reader would run them
PROBES = [
    "setup_probe.py",
    "appendix_b_probe.py",
    "s20_relative_value.py",
    "funding_carry_probe.py",
    "appendix_c_statarb.py",
    "s02_nonoverlap_backtest.py",
    "extended_window_report.py",
]


def _run(script: str, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600)


def _snapshot(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes()
            for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A working directory with a results/ that must survive untouched and no
    market data - the state of a fresh clone before any fetch."""
    res = tmp_path / "results"
    res.mkdir()
    (res / "setup_probe_full.csv").write_text("sentinel\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("script", PROBES)
def test_default_run_never_touches_results(sandbox: Path, script: str):
    if script == "appendix_c_statarb.py" and \
            importlib.util.find_spec("statsmodels") is None:
        pytest.skip("appendix_c needs statsmodels (research-only dependency)")
    before = _snapshot(sandbox / "results")
    r = _run(script, sandbox)
    assert "Traceback" not in r.stderr, r.stderr[-800:]
    assert _snapshot(sandbox / "results") == before, \
        f"{script} wrote into results/ with default arguments"
    written = [p for p in sandbox.rglob("*")
               if p.is_file() and "results" not in p.parts]
    assert all("out" in p.parts for p in written), \
        f"{script} wrote outside out/: {written}"


def test_setup_probe_without_data_refuses_and_writes_nothing(sandbox: Path):
    r = _run("setup_probe.py", sandbox)
    assert r.returncode == 2
    assert "nothing evaluated" in r.stdout and "nothing written" in r.stdout
    assert not (sandbox / "out").exists()


def test_setup_probe_out_into_results_is_named_as_an_overwrite(sandbox: Path):
    r = _run("setup_probe.py", sandbox, "--out",
             "results/setup_probe_full.csv")
    assert "OVERWRITING the campaign evidence base" in r.stdout
    assert (sandbox / "results" / "setup_probe_full.csv").read_text(
        encoding="utf-8") == "sentinel\n"       # no data -> still not written


@pytest.mark.parametrize("script", ["appendix_b_probe.py",
                                    "s20_relative_value.py",
                                    "funding_carry_probe.py",
                                    "s02_nonoverlap_backtest.py",
                                    "extended_window_report.py"])
def test_outdir_results_is_named_as_an_overwrite(sandbox: Path, script: str):
    r = _run(script, sandbox, "--outdir", "results")
    assert "OVERWRITING the campaign evidence base" in r.stdout
    assert "Traceback" not in r.stderr, r.stderr[-800:]


def test_setup_probe_partial_run_is_labelled(sandbox: Path):
    """One symbol of flat synthetic bars: the probe runs, writes to out/, and
    every verdict-shaped line is replaced by the PARTIAL RUN label."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    d = sandbox / "data" / "native" / "ETHUSDT" / "1h"
    d.mkdir(parents=True)
    n = 4000
    t0 = 1_767_225_600_000                              # 2026-01-01 UTC
    rng = np.random.default_rng(1)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    pd.DataFrame({
        "open_time": t0 + 3_600_000 * np.arange(n, dtype=np.int64),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1000.0 + rng.random(n),
        "cvd_bar": rng.normal(0, 10, n), "n_trades": 500,
        "taker_buy_vol": 500.0, "oi": 1e6 + np.cumsum(rng.normal(0, 100, n)),
        "funding": np.round(rng.normal(1e-4, 5e-5, n), 8),
        "premium": rng.normal(0, 1e-4, n),
    }).to_parquet(d / "bars.parquet", index=False)
    r = _run("setup_probe.py", sandbox, "--tfs", "1h")
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert "PARTIAL RUN" in r.stdout and "DRY RUN on 1/26" in r.stdout
    assert "not a verdict on the setup library" in r.stdout
    assert "closed honestly" not in r.stdout
    assert "  VERDICT" not in r.stdout
    assert (sandbox / "out" / "setup_probe_full.csv").is_file()
    assert (sandbox / "results" / "setup_probe_full.csv").read_text(
        encoding="utf-8") == "sentinel\n"
