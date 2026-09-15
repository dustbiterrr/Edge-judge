"""
ingest.py — CSV -> validated trade table.  Human errors, not tracebacks.

Required columns (aliases accepted, case-insensitive):
    symbol      (or via `symbol=` argument if the file has none)
    side        long/short | buy/sell | +1/-1     (alias: dir)
    entry_time  parseable datetime (UTC assumed)
    exit_time   parseable datetime
Optional:
    entry_price, exit_price (aliases: entry_px, exit_px), qty

Every validation failure raises IngestError with a .message a human can act
on.  Non-fatal issues (overlaps, reconstructed prices) come back as warnings.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

MAX_BYTES = 10 * 1024 * 1024
MAX_SPAN_DAYS = 366

ALIASES = {
    "symbol": ["symbol", "ticker", "pair", "market"],
    "side": ["side", "dir", "direction", "position"],
    "entry_time": ["entry_time", "open_time", "entrytime", "time_in", "entry"],
    "exit_time": ["exit_time", "close_time", "exittime", "time_out", "exit"],
    "entry_price": ["entry_price", "entry_px", "open_price", "price_in"],
    "exit_price": ["exit_price", "exit_px", "close_price", "price_out"],
    "qty": ["qty", "quantity", "size", "amount"],
}

TEMPLATE_CSV = (
    "symbol,side,entry_time,exit_time,entry_price,exit_price,qty\n"
    "ETHUSDT,long,2026-01-05 10:00,2026-01-05 18:00,3010.5,3042.0,0.5\n"
    "ETHUSDT,short,2026-01-06 02:00,2026-01-06 09:00,3055.0,3021.2,0.5\n"
)


class IngestError(Exception):
    """Validation failure with a message written for a human."""


# epoch ranges: value magnitude -> unit.  Anything numeric outside these is
# rejected loudly — pandas would otherwise read it as NANOSECONDS and produce
# silent 1970 dates.
_EPOCH_UNITS = (
    (1e9, 1e11, "s"),
    (1e12, 1e14, "ms"),
    (1e14, 1e17, "us"),
    (1e17, 1e19, "ns"),
)


def _parse_time_col(raw: pd.Series, col: str,
                    warnings: list[str]) -> pd.Series:
    """Datetime strings via the mixed parser; all-numeric columns as unix
    epoch with the unit inferred from magnitude (noted in the report)."""
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().all():
        v = float(num.abs().median())
        if 1e7 <= v < 1e8:                      # yyyymmdd integer dates
            parsed = pd.to_datetime(raw.astype(str), format="%Y%m%d",
                                    errors="coerce", utc=True)
            if parsed.notna().all():
                warnings.append(f"'{col}': integer dates parsed as YYYYMMDD "
                                f"(UTC).")
                return parsed
        for lo, hi, unit in _EPOCH_UNITS:
            if lo <= v < hi:
                warnings.append(f"'{col}': timestamps parsed as epoch {unit} "
                                f"(UTC).")
                return pd.to_datetime(num, unit=unit, utc=True,
                                      errors="coerce")
        raise IngestError(
            f"Column '{col}' is numeric but does not match any epoch "
            f"range (seconds ~1.7e9, milliseconds ~1.7e12, micro ~1.7e15, "
            f"nano ~1.7e18). Use ISO datetimes like 2026-01-05 10:00 (UTC) "
            f"or unix epoch timestamps.")
    return pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")


def _find_col(cols: list[str], key: str) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for a in ALIASES[key]:
        if a in low:
            return low[a]
    return None


def _norm_side(v) -> float:
    s = str(v).strip().lower()
    if s in ("long", "buy", "1", "+1", "1.0"):
        return 1.0
    if s in ("short", "sell", "-1", "-1.0"):
        return -1.0
    return 0.0


def load_trades(raw: bytes, filename: str = "log.csv",
                symbol: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Returns (trades, warnings).  Raises IngestError on fatal problems."""
    warnings: list[str] = []

    if len(raw) > MAX_BYTES:
        raise IngestError(
            f"File is {len(raw) / 1e6:.1f} MB — the limit is 10 MB. "
            f"Export a shorter period or fewer columns.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise IngestError(f"Could not read this file as CSV ({type(e).__name__}). "
                          f"Export a plain comma-separated file and retry.")
    if df.empty:
        raise IngestError("The CSV has no rows.")

    cols = list(df.columns)
    mapping = {k: _find_col(cols, k) for k in ALIASES}

    missing = [k for k in ("side", "entry_time", "exit_time")
               if mapping[k] is None]
    if missing:
        raise IngestError(
            "Required columns not found: " + ", ".join(missing) + ".\n"
            f"Your file has: {', '.join(cols)}.\n"
            "Needed: symbol, side (long/short or +1/-1), entry_time, "
            "exit_time, and optionally entry_price/exit_price/qty. "
            "Download the template for the exact format.")

    out = pd.DataFrame(index=df.index)   # keep source index: scalar
    # assignments must broadcast to every row, not create an empty frame

    # symbol: column, else argument, else filename hint like ..._ETHUSDT_...
    if mapping["symbol"] is not None:
        out["symbol"] = df[mapping["symbol"]].astype(str).str.upper().str.strip()
    elif symbol:
        out["symbol"] = symbol.upper()
    else:
        m = re.search(r"([A-Z0-9]{2,15}USDT)", filename.upper())
        if m:
            out["symbol"] = m.group(1)
            warnings.append(f"No symbol column — took {m.group(1)} from the "
                            f"file name.")
        else:
            raise IngestError(
                "No symbol column found and the file name does not contain "
                "one. Add a 'symbol' column (e.g. ETHUSDT).")

    bad_syms = sorted(s for s in out["symbol"].unique()
                      if not re.fullmatch(r"[A-Z0-9]{2,15}USDT", s))
    if bad_syms:
        raise IngestError(
            "We currently support Binance USDT-perp futures only. "
            f"Found: {', '.join(bad_syms[:5])}. Symbols must look like "
            "ETHUSDT / SOLUSDT.")

    out["side"] = df[mapping["side"]].map(_norm_side)
    n_bad_side = int((out["side"] == 0).sum())
    if n_bad_side:
        ex = df.loc[out["side"] == 0, mapping["side"]].astype(str).unique()[:3]
        raise IngestError(
            f"{n_bad_side} rows have an unrecognized side value "
            f"(examples: {', '.join(ex)}). Use long/short, buy/sell or +1/-1.")

    for k in ("entry_time", "exit_time"):
        parsed = _parse_time_col(df[mapping[k]], mapping[k], warnings)
        bad = parsed.isna()
        if bad.any():
            rows = df.loc[bad, mapping[k]].astype(str).head(3).tolist()
            raise IngestError(
                f"{int(bad.sum())} rows in '{mapping[k]}' failed to parse as "
                f"datetimes. First broken values: {rows}. "
                "Use ISO format like 2026-01-05 10:00 (UTC).")
        out[k] = parsed

    if (out["exit_time"] <= out["entry_time"]).any():
        n = int((out["exit_time"] <= out["entry_time"]).sum())
        raise IngestError(f"{n} rows have exit_time <= entry_time — a trade "
                          "cannot close before it opens. Check the export.")

    now = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5)
    if (out["exit_time"] > now).any():
        n = int((out["exit_time"] > now).sum())
        latest = out.loc[out["exit_time"] > now, "exit_time"] \
            .max().strftime("%Y-%m-%d %H:%M")
        raise IngestError(
            f"{n} rows have exit_time in the future (latest: {latest} UTC). "
            "A closed trade cannot end after now — check the timestamp "
            "units and timezone of the export.")

    span = (out["exit_time"].max() - out["entry_time"].min()).days
    if span > MAX_SPAN_DAYS:
        raise IngestError(
            f"The log spans {span} days — the limit is 12 months (market-data "
            "volume). Trim the period and re-upload; the verdict on a recent "
            "12-month window is the one that matters anyway.")

    for k in ("entry_price", "exit_price", "qty"):
        if mapping[k] is not None:
            out[k] = pd.to_numeric(df[mapping[k]], errors="coerce")
        else:
            out[k] = np.nan
    # each provided price validates on its own — no gating on whether the
    # OTHER price column exists
    for k in ("entry_price", "exit_price"):
        if (out[k] <= 0).any():
            raise IngestError(f"Some {k} values are zero or negative.")

    have_px = out["entry_price"].notna() & out["exit_price"].notna()
    if have_px.all():
        pass
    elif have_px.any():
        warnings.append(f"{int((~have_px).sum())} rows lack prices — those "
                        "will be reconstructed from market data.")
    else:
        warnings.append("No prices in the log — all entries/exits will be "
                        "reconstructed from market data (next 1h bar open).")

    out = out.sort_values("entry_time").reset_index(drop=True)
    return out, warnings


def describe_overlaps(trades: pd.DataFrame) -> str | None:
    """Human preview of how the judge will resolve same-symbol overlaps."""
    from judge.overlap import resolve_overlaps
    _, dropped = resolve_overlaps(trades)
    if dropped == 0:
        return None
    return (f"{dropped} trades overlap an already-open position on the same "
            f"symbol. The judge keeps the FIRST position and drops entries "
            f"made while it was open (one-position-at-a-time rule) — "
            f"overlapping entries are how per-trade statistics get inflated.")
