"""The CLI on a user's mistake: one line starting with ERROR:, exit code 2,
no traceback, no path the user did not type.

README promises "a sentence, not a traceback".  Before this file, a missing
--log file produced a FileNotFoundError traceback that printed the absolute
path of the clone (user name included) and exited 1 - the same code as a
FAIL verdict.  Exit codes are a contract: 0 PASS, 1 FAIL, 2 could-not-run."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from judge.__main__ import EXIT_ERROR, main
from judge.ingest import IngestError, load_trades

ROOT = Path(__file__).resolve().parent.parent
GOOD_LOG = ROOT / "results" / "s02_trades_LINKUSDT_16.csv"


def _run(capsys, *argv: str) -> tuple[int, str]:
    rc = main(["audit", *argv])
    out = capsys.readouterr()
    return rc, out.out + out.err


def _assert_clean(text: str, *paths: Path) -> None:
    assert "Traceback" not in text
    for p in paths:                       # no absolute path leaks
        assert str(p.resolve()) not in text
        assert p.resolve().as_posix() not in text
    assert str(Path.home()) not in text


# ── --log ────────────────────────────────────────────────────────────────────

def test_missing_log_is_one_line_exit_2(capsys, tmp_path):
    rc, text = _run(capsys, "--log", "my_trades.csv", "--out",
                    str(tmp_path / "r.md"))
    assert rc == EXIT_ERROR
    assert text.startswith("ERROR: file not found: my_trades.csv")
    _assert_clean(text, ROOT, tmp_path)


def test_log_is_a_directory(capsys, tmp_path):
    rc, text = _run(capsys, "--log", "results", "--out", str(tmp_path / "r.md"))
    assert rc == EXIT_ERROR
    assert "ERROR: results is a directory, not a file" in text
    _assert_clean(text, ROOT, tmp_path)


def test_log_unreadable_permission_denied(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "locked.csv"
    log.write_text("symbol,side,entry_time,exit_time\n", encoding="utf-8")

    def denied(self):
        raise PermissionError(13, "Permission denied", str(self))
    monkeypatch.setattr(Path, "read_bytes", denied)
    rc, text = _run(capsys, "--log", "locked.csv", "--out",
                    str(tmp_path / "r.md"))
    assert rc == EXIT_ERROR
    assert "ERROR: cannot read locked.csv: permission denied." in text
    _assert_clean(text, ROOT, tmp_path)


def test_empty_log(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("empty.csv").write_bytes(b"")
    rc, text = _run(capsys, "--log", "empty.csv", "--out", "r.md")
    assert rc == EXIT_ERROR
    assert text.startswith("ERROR: The file is empty")
    _assert_clean(text, ROOT, tmp_path)


def test_whitespace_only_log_is_empty_too():
    with pytest.raises(IngestError, match="empty"):
        load_trades(b"\n\n  \n")


def test_log_is_not_a_csv_binary(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("blob.csv").write_bytes(bytes(range(256)) * 8)
    rc, text = _run(capsys, "--log", "blob.csv", "--out", "r.md")
    assert rc == EXIT_ERROR
    assert "ERROR: This file is not UTF-8 text" in text
    _assert_clean(text, ROOT, tmp_path)


def test_log_is_an_xlsx_archive(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("trades.csv").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    rc, text = _run(capsys, "--log", "trades.csv", "--out", "r.md")
    assert rc == EXIT_ERROR
    assert "ERROR: This is a ZIP or Excel (.xlsx) archive, not a CSV" in text
    _assert_clean(text, ROOT, tmp_path)


def test_malformed_csv_is_still_a_sentence(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("t.csv").write_text("symbol,side\nETHUSDT,long\n", encoding="utf-8")
    rc, text = _run(capsys, "--log", "t.csv", "--out", "r.md")
    assert rc == EXIT_ERROR
    assert text.startswith("ERROR: Required columns not found")
    _assert_clean(text, ROOT, tmp_path)


# ── --fees / --out ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("fees", ["free", "0", "-3", "nan", "inf"])
def test_bad_fees_is_one_line_exit_2(capsys, tmp_path, fees):
    rc, text = _run(capsys, "--log", str(GOOD_LOG), "--fees", fees,
                    "--out", str(tmp_path / "r.md"))
    assert rc == EXIT_ERROR
    assert text.startswith("ERROR: --fees must be")
    _assert_clean(text, ROOT, tmp_path)


def test_out_is_a_directory(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("out").mkdir()
    Path("t.csv").write_bytes(GOOD_LOG.read_bytes())
    rc, text = _run(capsys, "--log", "t.csv", "--out", "out")
    assert rc == EXIT_ERROR
    assert "ERROR: --out out is a directory" in text
    _assert_clean(text, ROOT, tmp_path)


def test_out_parent_is_a_file(capsys, tmp_path, monkeypatch):
    """A bad --out fails before any market data is requested (the network is
    disabled in tests: reaching it would raise, not print ERROR)."""
    monkeypatch.chdir(tmp_path)
    Path("blocker").write_text("x", encoding="utf-8")
    Path("t.csv").write_bytes(GOOD_LOG.read_bytes())
    rc, text = _run(capsys, "--log", "t.csv", "--out", "blocker/report.md")
    assert rc == EXIT_ERROR
    assert "ERROR: cannot create the folder for --out blocker/report.md" in text
    _assert_clean(text, ROOT, tmp_path)


def test_report_write_failure_is_one_line(capsys, tmp_path, monkeypatch):
    """The audit ran; the report could not be written: ERROR, exit 2."""
    from conftest import make_klines, trade_log
    monkeypatch.chdir(tmp_path)
    Path("t.csv").write_bytes(trade_log())
    monkeypatch.setattr("judge.checks.ensure_klines",
                        lambda s, t0, t1, progress=None: make_klines(
                            t0 - 48 * 3_600_000, (t1 - t0) // 3_600_000 + 97))

    def bad_write(self, *a, **k):
        raise OSError(28, "No space left on device", str(self))
    monkeypatch.setattr(Path, "write_text", bad_write)
    rc, text = _run(capsys, "--log", "t.csv", "--out", "r.md")
    assert rc == EXIT_ERROR
    assert "ERROR: cannot write the report to r.md: No space left on device" in text
    _assert_clean(text, ROOT, tmp_path)


# ── the last resort: a bug is still one line, unless JUDGE_DEBUG is set ──────

def test_unexpected_exception_is_one_line_exit_2(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("t.csv").write_bytes(GOOD_LOG.read_bytes())
    monkeypatch.setattr("judge.__main__.load_trades",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.delenv("JUDGE_DEBUG", raising=False)
    rc, text = _run(capsys, "--log", "t.csv", "--out", "r.md")
    assert rc == EXIT_ERROR
    assert text.startswith("ERROR: unexpected problem (RuntimeError)")
    assert "JUDGE_DEBUG=1" in text
    _assert_clean(text, ROOT, tmp_path)


def test_judge_debug_reraises(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("t.csv").write_bytes(GOOD_LOG.read_bytes())
    monkeypatch.setattr("judge.__main__.load_trades",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setenv("JUDGE_DEBUG", "1")
    with pytest.raises(RuntimeError, match="boom"):
        main(["audit", "--log", "t.csv", "--out", "r.md"])


# ── the verdict path keeps its codes: 0 PASS / 1 FAIL ────────────────────────

def test_exit_codes_pass_and_fail(capsys, tmp_path, monkeypatch):
    from conftest import make_klines, trade_log
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("judge.checks.ensure_klines",
                        lambda s, t0, t1, progress=None: make_klines(
                            t0 - 48 * 3_600_000, (t1 - t0) // 3_600_000 + 97))
    Path("pass.csv").write_bytes(trade_log())                 # +0.5% each
    Path("fail.csv").write_bytes(trade_log(exit_=99.5))       # -0.5% each
    rc, text = _run(capsys, "--log", "pass.csv", "--out", "p.md")
    assert rc == 0 and "PASS —" in text and "report -> p.md" in text
    rc, text = _run(capsys, "--log", "fail.csv", "--out", "f.md")
    assert rc == 1 and "FAIL —" in text
    _assert_clean(text, ROOT, tmp_path)
    assert os.path.isfile("p.md") and os.path.isfile("f.md")
