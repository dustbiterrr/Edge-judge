"""CLI:  python -m judge audit --log trades.csv --out report.md
            [--fees binance-taker|binance-maker|<bps-per-side>] [--symbol SYM]
            [--date-format DMY|MDY]

Exit codes: 0 PASS, 1 FAIL, 2 the judge could not run (unreadable log,
malformed CSV, no market data, bad flag, unwritable report).  A user error
gets one line starting with "ERROR:" - never a traceback, never a path the
user did not type.  Set JUDGE_DEBUG=1 to see the traceback of an unexpected
failure.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from judge import FEE_PROFILES
from judge.checks import audit
from judge.ingest import IngestError, load_trades
from judge.marketdata import MarketDataError
from judge.report import PASS_MEANS, to_html, to_markdown, verdict_line

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


class CliError(Exception):
    """A problem the user can fix; printed as one line, exit code 2."""


def parse_fees(v: str) -> float:
    if v in FEE_PROFILES:
        return FEE_PROFILES[v]
    try:
        bps = float(v)
    except ValueError:
        raise CliError(f"--fees must be one of {list(FEE_PROFILES)} or a "
                       f"number (bps per side), got: {v}")
    if not math.isfinite(bps) or bps <= 0:
        raise CliError(f"--fees must be a finite number of bps per side "
                       f"> 0, got: {v}")
    return bps * 2 / 100.0                 # bps per side -> % round trip


def read_log(arg: str) -> bytes:
    """The bytes of --log, or a CliError that names the path as typed."""
    path = Path(arg)
    if not path.exists():
        raise CliError(f"file not found: {arg}")
    if path.is_dir():
        raise CliError(f"{arg} is a directory, not a file. Point --log at "
                       "the CSV itself.")
    try:
        return path.read_bytes()
    except PermissionError:
        raise CliError(f"cannot read {arg}: permission denied.")
    except OSError as e:
        raise CliError(f"cannot read {arg}: {e.strerror or type(e).__name__}.")


def prepare_out(arg: str) -> Path:
    """Where the report will go, checked before any market data is fetched
    so a bad --out fails in a millisecond, not after a download."""
    out = Path(arg)
    if out.is_dir():
        raise CliError(f"--out {arg} is a directory; give a file name such "
                       "as out/report.md or out/report.html.")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, FileExistsError) as e:
        raise CliError(f"cannot create the folder for --out {arg}: "
                       f"{e.strerror or type(e).__name__}.")
    if not os.access(out.parent, os.W_OK):
        raise CliError(f"cannot write to the folder of --out {arg}: "
                       "permission denied.")
    return out


def run(args: argparse.Namespace) -> int:
    fee_rt = parse_fees(args.fees)
    fee_label = (args.fees if args.fees in FEE_PROFILES
                 else f"custom {args.fees} bps/side")
    raw = read_log(args.log)
    out = prepare_out(args.out)
    try:
        trades, warns = load_trades(raw, filename=Path(args.log).name,
                                    symbol=args.symbol,
                                    date_format=args.date_format)
        res = audit(trades, fee_rt, progress=lambda m: print(f"  .. {m}"),
                    fee_label=fee_label)
    except (IngestError, MarketDataError) as e:
        raise CliError(str(e))
    res.warnings = warns + res.warnings

    try:
        if out.suffix == ".html":
            out.write_text(to_html(res), encoding="utf-8")
        else:
            out.write_text(to_markdown(res), encoding="utf-8")
    except OSError as e:
        raise CliError(f"cannot write the report to {args.out}: "
                       f"{e.strerror or type(e).__name__}.")
    if res.n_trades == 0:
        print("INSUFFICIENT — none of the trades could be priced against "
              "available market data; nothing to judge.")
    print(verdict_line(res))
    for c in res.checks:
        print(f"  {c.code} {c.status:<12} {c.name:<28} {c.key_number}")
        if c.code == "C2":
            print(f"      {res.cost_note}")
        if c.status == "UNVERIFIABLE":
            print(f"      {c.detail}")
    if res.verdict == "PASS":
        import textwrap
        print()
        print(textwrap.fill(PASS_MEANS, width=96, initial_indent="  ",
                            subsequent_indent="  "))
    print(f"report -> {args.out}")
    return EXIT_PASS if res.verdict == "PASS" else EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="judge")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit a trade-log CSV")
    a.add_argument("--log", required=True,
                   help="trade-log CSV: symbol, side, entry_time, exit_time; "
                        "optional signal_time, entry_price, exit_price, qty")
    a.add_argument("--out", default="report.md",
                   help="report path, .md or .html (default: report.md)")
    a.add_argument("--fees", default="binance-taker",
                   help="binance-taker (default), binance-maker, or a number "
                        "of bps per side")
    a.add_argument("--symbol", default=None,
                   help="symbol if the CSV has no symbol column")
    a.add_argument("--date-format", default=None, choices=["DMY", "MDY"],
                   dest="date_format",
                   help="day/month order for slash-style dates (03/02/2026); "
                        "required when every such date is ambiguous - the "
                        "judge never guesses. ISO and epoch need no flag.")
    args = p.parse_args(argv)

    try:
        return run(args)
    except CliError as e:
        print(f"ERROR: {e}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("ERROR: interrupted.")
        return EXIT_ERROR
    except Exception as e:                       # invariant: no tracebacks
        if os.environ.get("JUDGE_DEBUG"):
            raise
        print(f"ERROR: unexpected problem ({type(e).__name__}). If your CSV "
              "matches the template and this persists, simplify it to the "
              "template columns and retry; set JUDGE_DEBUG=1 to see the "
              "traceback.")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
