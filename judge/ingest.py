"""
ingest.py — CSV -> validated trade table.  Human errors, not tracebacks.

Required columns (aliases accepted, case-insensitive):
    symbol      (or via `symbol=` argument if the file has none)
    side        long/short | buy/sell | +1/-1     (alias: dir)
    entry_time  parseable datetime (UTC assumed)
    exit_time   parseable datetime
Optional:
    entry_price, exit_price (aliases: entry_px, exit_px), qty
    signal_time (aliases: decision_time, signal_ts, sig_time, ...) - the
                moment the decision was made.  Enables the C6 look-ahead
                check; without it C6 is reported UNVERIFIABLE.
Every trade keeps `row`, its 1-based line number in the source CSV, so a
check can name the offending line rather than a position in a sorted table.

Every validation failure raises IngestError with a .message a human can act
on.  Non-fatal issues (overlaps, reconstructed prices) come back as warnings.

Dates: ISO (2026-02-03 04:00) and unix epoch are unambiguous.  Slash-style
dates (03/02/2026) are parsed under ONE rule for the whole file - day-first
or month-first - never row by row.  The rule is inferred only when some value
forces it (a field > 12); an all-ambiguous file is rejected unless the caller
passes date_format="DMY" | "MDY"; a file that forces both rules is rejected.
"""

from __future__ import annotations

import io
import re
import warnings as _warnings

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
    # decision time - time-like names only; a bare "signal" column usually
    # holds the signal VALUE (+1/-1), not a timestamp
    "signal_time": ["signal_time", "signal_ts", "signal_timestamp",
                    "decision_time", "decision_ts", "sig_time", "signal_at",
                    "decided_at"],
}

TEMPLATE_CSV = (
    "symbol,side,signal_time,entry_time,exit_time,entry_price,exit_price,qty\n"
    "ETHUSDT,long,2026-01-05 09:00,2026-01-05 10:00,2026-01-05 18:00,3010.5,3042.0,0.5\n"
    "ETHUSDT,short,2026-01-06 01:00,2026-01-06 02:00,2026-01-06 09:00,3055.0,3021.2,0.5\n"
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


# slash-style date: two 1-2 digit fields, then a 2- or 4-digit year
_SLASH_DATE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})(?:[ T]|$)")
DATE_FORMATS = ("DMY", "MDY")


def _slash_fields(series: pd.Series) -> list[tuple[int, int, int, str]]:
    """(first, second, source_row, value) for every slash-style value."""
    out = []
    for i, v in zip(series.index, series):
        m = _SLASH_DATE.match(v) if isinstance(v, str) else None
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(i), v.strip()))
    return out


def _infer_day_month_rule(columns: dict[str, pd.Series],
                          date_format: str | None,
                          warnings: list[str]) -> str | None:
    """One day/month rule for the whole file, or None when the file has no
    slash-style dates.  Never guesses: an ambiguous file is an IngestError."""
    if date_format is not None and date_format not in DATE_FORMATS:
        raise IngestError(f"date_format must be one of {DATE_FORMATS}, "
                          f"got {date_format!r}.")
    fields = [(a, b, row, v, col) for col, ser in columns.items()
              for a, b, row, v in _slash_fields(ser)]
    if not fields:
        return None
    force_dmy = [f for f in fields if f[0] > 12]    # first field cannot be a month
    force_mdy = [f for f in fields if f[1] > 12]    # second field cannot be a month
    n_slash = len(fields)

    def _ex(f):
        return f"'{f[3]}' (row {f[2] + 2}, column {f[4]})"

    if force_dmy and force_mdy:
        raise IngestError(
            "Dates in this file do not follow one rule: "
            f"{_ex(force_dmy[0])} can only be day-first, but "
            f"{_ex(force_mdy[0])} can only be month-first. The judge will "
            "not parse rows by different rules. Re-export with ISO dates "
            "(2026-02-03 04:00, UTC) or unix epoch timestamps.")
    if date_format is not None:
        if date_format == "DMY" and force_mdy:
            raise IngestError(
                f"date_format=DMY was requested but {_ex(force_mdy[0])} has "
                "a month field > 12 under that rule. Check the export.")
        if date_format == "MDY" and force_dmy:
            raise IngestError(
                f"date_format=MDY was requested but {_ex(force_dmy[0])} has "
                "a month field > 12 under that rule. Check the export.")
        rule = date_format
        name = ("day-first (DD/MM/YYYY)" if rule == "DMY"
                else "month-first (MM/DD/YYYY)")
        warnings.append(
            f"{n_slash} slash-style dates parsed as {name} - explicit "
            f"date_format={rule}, applied to every row (UTC).")
        return rule
    if force_dmy:
        warnings.append(
            f"{n_slash} slash-style dates parsed as day-first (DD/MM/YYYY): "
            f"{len(force_dmy)} value(s) such as {_ex(force_dmy[0])} have a "
            "first field > 12, so day-first is the only consistent rule; "
            "applied to every row (UTC).")
        return "DMY"
    if force_mdy:
        warnings.append(
            f"{n_slash} slash-style dates parsed as month-first (MM/DD/YYYY): "
            f"{len(force_mdy)} value(s) such as {_ex(force_mdy[0])} have a "
            "second field > 12, so month-first is the only consistent rule; "
            "applied to every row (UTC).")
        return "MDY"
    raise IngestError(
        f"Slash-style dates are ambiguous: every one of the {n_slash} values "
        f"(e.g. {_ex(fields[0])}) has both fields <= 12, so it could be "
        "day/month or month/day and the judge will not guess. Re-export with "
        "ISO dates (2026-02-03 04:00, UTC) or unix epoch timestamps, or state "
        "the format explicitly: --date-format DMY|MDY on the CLI, the date "
        "format selector in the web app.")


def _parse_time_col(raw: pd.Series, col: str,
                    warnings: list[str], rule: str | None = None) -> pd.Series:
    """Datetime strings via the mixed parser under ONE day/month rule for the
    file; all-numeric columns as unix epoch with the unit inferred from
    magnitude (noted in the report)."""
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
    # dayfirst=True makes dateutil swap day and month even in an ISO string
    # (2026-02-03 -> 2 March), so the rule is applied to slash-style values
    # ONLY; everything else is parsed with the default (ISO-safe) parser.
    slash = raw.map(lambda v: isinstance(v, str)
                    and _SLASH_DATE.match(v) is not None).astype(bool)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")        # dateutil dayfirst chatter
        parsed = pd.to_datetime(raw.where(~slash), errors="coerce", utc=True,
                                format="mixed")
        if slash.any():
            by_rule = pd.to_datetime(raw.where(slash), errors="coerce",
                                     utc=True, format="mixed",
                                     dayfirst=(rule == "DMY"))
            parsed = parsed.where(~slash, by_rule)
    # The rule is a contract, not a hint: verify every slash-style value
    # came out under it.  A parser that quietly used the other convention
    # for one row would otherwise produce exactly the two-rules-in-one-file
    # defect this guard exists to prevent.
    for a, b, row, v in _slash_fields(raw):
        ts = parsed.loc[row]
        if pd.isna(ts):
            continue
        day, month = (a, b) if rule == "DMY" else (b, a)
        if rule is None or ts.day != day or ts.month != month:
            raise IngestError(
                f"'{v}' (row {row + 2}, column '{col}') did not parse under "
                f"the file's {rule or 'unset'} date rule (got "
                f"{ts:%Y-%m-%d}). Re-export with ISO dates "
                "(2026-02-03 04:00, UTC) or unix epoch timestamps.")
    return parsed


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
                symbol: str | None = None,
                date_format: str | None = None,
                ) -> tuple[pd.DataFrame, list[str]]:
    """Returns (trades, warnings).  Raises IngestError on fatal problems.
    date_format: None (infer one rule, refuse if ambiguous) | "DMY" | "MDY"."""
    warnings: list[str] = []

    if len(raw) > MAX_BYTES:
        raise IngestError(
            f"File is {len(raw) / 1e6:.1f} MB — the limit is 10 MB. "
            f"Export a shorter period or fewer columns.")
    if not raw.strip():
        raise IngestError("The file is empty (0 bytes of data). Export the "
                          "trades again and retry.")
    if raw[:4] == b"PK\x03\x04":
        raise IngestError("This is a ZIP or Excel (.xlsx) archive, not a CSV. "
                          "Export as CSV (comma-separated text) and retry.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        raise IngestError("This file is not UTF-8 text - a binary file, or a "
                          "CSV in another encoding. Export as plain UTF-8 "
                          "comma-separated text and retry.")
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
    out["row"] = (df.index + 2).astype(int)     # CSV line number, header = 1

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

    time_keys = ["entry_time", "exit_time"]
    if mapping["signal_time"] is not None:
        time_keys.append("signal_time")
    time_cols = {mapping[k]: df[mapping[k]] for k in time_keys}
    rule = _infer_day_month_rule(time_cols, date_format, warnings)
    for k in time_keys:
        parsed = _parse_time_col(df[mapping[k]], mapping[k], warnings, rule)
        bad = parsed.isna()
        if bad.any():
            rows = df.loc[bad, mapping[k]].astype(str).head(3).tolist()
            hint = ("" if k != "signal_time" else
                    " A signal_time column must be complete: fill every row, "
                    "or drop the column and the look-ahead check is reported "
                    "UNVERIFIABLE instead of guessed.")
            raise IngestError(
                f"{int(bad.sum())} rows in '{mapping[k]}' failed to parse as "
                f"datetimes. First broken values: {rows}. "
                "Use ISO format like 2026-01-05 10:00 (UTC)." + hint)
        out[k] = parsed
    if mapping["signal_time"] is None:
        out["signal_time"] = pd.NaT

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

    # Say exactly what is missing.  A column that is absent and a cell that
    # is blank or non-numeric are different situations; a provided price is
    # never overwritten, only the missing cell is reconstructed.
    if mapping["entry_price"] is None and mapping["exit_price"] is None:
        warnings.append("No price columns in the log — every entry and exit "
                        "price will be reconstructed from market data (next "
                        "1h bar open after each timestamp).")
    else:
        for k in ("entry_price", "exit_price"):
            if mapping[k] is None:
                warnings.append(f"No {k} column — every {k} will be "
                                "reconstructed from market data (next 1h bar "
                                "open); the other side's prices are kept.")
                continue
            miss = out[k].isna()
            if miss.any():
                lines = [int(r) for r in out.loc[miss, "row"]]
                shown = ", ".join(map(str, lines[:5]))
                more = f", +{len(lines) - 5} more" if len(lines) > 5 else ""
                warnings.append(
                    f"{int(miss.sum())} of {len(out)} rows have a blank or "
                    f"non-numeric {k} (CSV line{'s' if len(lines) != 1 else ''} "
                    f"{shown}{more}) — only those cells are reconstructed "
                    "from market data; every provided price is kept.")

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
