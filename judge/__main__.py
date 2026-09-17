"""CLI:  python -m judge audit --log trades.csv --out report.md
            [--fees binance-taker|binance-maker|<bps-per-side>] [--symbol SYM]
            [--date-format DMY|MDY]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from judge import FEE_PROFILES
from judge.checks import audit
from judge.ingest import IngestError, load_trades
from judge.marketdata import MarketDataError
from judge.report import to_html, to_markdown, verdict_line

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def parse_fees(v: str) -> float:
    if v in FEE_PROFILES:
        return FEE_PROFILES[v]
    try:
        bps = float(v)
    except ValueError:
        raise SystemExit(f"--fees must be one of {list(FEE_PROFILES)} or a "
                         f"number (bps per side), got: {v}")
    if not math.isfinite(bps) or bps <= 0:
        raise SystemExit(f"--fees must be a finite number of bps per side "
                         f"> 0, got: {v}")
    return bps * 2 / 100.0                 # bps per side -> % round trip


def main() -> int:
    p = argparse.ArgumentParser(prog="judge")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit a trade-log CSV")
    a.add_argument("--log", required=True)
    a.add_argument("--out", default="report.md")
    a.add_argument("--fees", default="binance-taker")
    a.add_argument("--symbol", default=None,
                   help="symbol if the CSV has no symbol column")
    a.add_argument("--date-format", default=None, choices=["DMY", "MDY"],
                   dest="date_format",
                   help="day/month order for slash-style dates (03/02/2026); "
                        "required when every such date is ambiguous - the "
                        "judge never guesses. ISO and epoch need no flag.")
    args = p.parse_args()

    fee_rt = parse_fees(args.fees)
    fee_label = (args.fees if args.fees in FEE_PROFILES
                 else f"custom {args.fees} bps/side")
    raw = Path(args.log).read_bytes()
    try:
        trades, warns = load_trades(raw, filename=Path(args.log).name,
                                    symbol=args.symbol,
                                    date_format=args.date_format)
        res = audit(trades, fee_rt, progress=lambda m: print(f"  .. {m}"),
                    fee_label=fee_label)
    except (IngestError, MarketDataError) as e:
        print(f"ERROR: {e}")
        return 2
    res.warnings = warns + res.warnings

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".html":
        out.write_text(to_html(res), encoding="utf-8")
    else:
        out.write_text(to_markdown(res), encoding="utf-8")
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
    print(f"report -> {out}")
    return 0 if res.verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
