"""
EdgeJudge — pre-registered trade-log audit.

The criteria are FROZEN in checks.py (v1.0) and are not configurable from any
UI or CLI flag.  The only user input besides the log is the fee profile —
a property of the user's exchange, not of the judge.  The product never
executes user code; it reads a CSV of trades, nothing else.
"""

from judge.checks import CRITERIA_VERSION, audit  # noqa: F401
from judge.ingest import IngestError, load_trades  # noqa: F401
from judge.report import to_html, to_markdown, verdict_line  # noqa: F401

FEE_PROFILES = {
    "binance-taker": 0.11,   # % round trip (0.055%/side)
    "binance-maker": 0.04,   # % round trip (0.020%/side)
}
