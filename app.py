"""EdgeJudge — Streamlit MVP.  streamlit run app.py

Product invariants (enforced in code, not hoped for):
  * never executes user code — reads a trade-log CSV, nothing else;
  * verdict criteria are pre-registered in judge/ (v1.1.0; C1-C5 frozen
    since v1.0, C6 look-ahead added in v1.1.0) and are NOT configurable
    here - the user picks only a fee profile and, for slash-style dates,
    the day/month order;
  * a check the judge cannot run is shown as UNVERIFIABLE on screen, in
    the verdict line and in every report - never hidden behind a PASS;
  * no softening language: FAIL is FAIL with the reason in plain English.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from judge import FEE_PROFILES
from judge.checks import CRITERIA_VERSION, audit
from judge.ingest import TEMPLATE_CSV, IngestError, load_trades
from judge.marketdata import MarketDataError
from judge.report import (FAIL_RED, PASS_GREEN, chart_bootstrap, chart_equity,
                          chart_regimes, to_html, to_markdown, verdict_line,
                          what_this_means)

DEMO_LOG = Path("results/s02_trades_LINKUSDT_16.csv")
DEMO_SYMBOL = "LINKUSDT"


def run_pipeline(raw: bytes, filename: str, fee_rt: float,
                 symbol: str | None = None, status=None,
                 date_format: str | None = None):
    """The single audit path used by BOTH the upload flow and the demo
    button.  Returns (result | None, error_message | None)."""
    def prog(msg):
        if status is not None:
            status.write(f"⏳ {msg}")
    try:
        trades, warns = load_trades(raw, filename=filename, symbol=symbol,
                                    date_format=date_format)
        # overlap resolution happens ONCE, inside audit(); its warning covers
        # the dropped-trades story
        res = audit(trades, fee_rt, progress=prog)
        res.warnings = warns + res.warnings
        if res.n_trades == 0:
            return None, (
                "INSUFFICIENT — none of the trades could be priced against "
                "available market data, so there is nothing to judge. Check "
                "that entry/exit times are UTC, within the last 12 months, "
                "and that the symbol traded on Binance USDT-perp futures in "
                "that window.")
        return res, None
    except (IngestError, MarketDataError) as e:
        return None, str(e)
    except Exception as e:                       # invariant: no tracebacks
        return None, (f"Unexpected problem ({type(e).__name__}). If your CSV "
                      f"matches the template and this persists, the log has "
                      f"something we did not anticipate — simplify it to the "
                      f"template columns and retry.")


def fee_selector() -> float:
    prof = st.radio(
        "Fee profile (property of your exchange — the verdict criteria are "
        "fixed and not configurable)",
        ["Binance perp taker (0.055%/side)",
         "Binance perp maker (0.020%/side)",
         "Custom"],
        horizontal=True)
    if prof.startswith("Binance perp taker"):
        return FEE_PROFILES["binance-taker"]
    if prof.startswith("Binance perp maker"):
        return FEE_PROFILES["binance-maker"]
    bps = st.number_input("bps per side", min_value=0.0, max_value=50.0,
                          value=5.5, step=0.5)
    return bps * 2 / 100.0


_DATE_CHOICES = {
    "auto (ISO 2026-02-03 04:00, unix epoch, or unambiguous slash dates)": None,
    "DD/MM/YYYY (day first)": "DMY",
    "MM/DD/YYYY (month first)": "MDY",
}


def date_format_selector() -> str | None:
    choice = st.selectbox(
        "Date format in your file", list(_DATE_CHOICES),
        help="Slash-style dates such as 03/02/2026 are read under ONE rule "
             "for the whole file. If every such date is ambiguous the judge "
             "refuses rather than guessing - pick the rule here.")
    return _DATE_CHOICES[choice]


_BADGE = {"PASS": "🟢 PASS", "FAIL": "🔴 FAIL",
          "UNVERIFIABLE": "⚪ UNVERIFIABLE"}


def show_verdict(res):
    color = PASS_GREEN if res.verdict == "PASS" else FAIL_RED
    st.markdown(
        f"<h2 style='color:{color};font-family:ui-monospace,monospace'>"
        f"{verdict_line(res)}</h2>", unsafe_allow_html=True)
    for c in res.unverifiable:
        st.warning(f"{c.code} {c.name}: UNVERIFIABLE ({c.key_number}). "
                   f"{c.detail}")

    rows = []
    for c in res.checks:
        badge = _BADGE[c.status]
        rows.append({"#": c.code, "check": c.name, "result": badge,
                     "key number": c.key_number, "detail": c.detail})
    st.dataframe(rows, width="stretch", hide_index=True)

    if res.reconstructed:
        st.warning("prices reconstructed from market data "
                   "(next 1h bar open after each timestamp)")
    for w in res.warnings:
        if "reconstructed" not in w:
            st.caption(f"note: {w}")

    st.image(chart_bootstrap(res))
    st.image(chart_equity(res))
    st.image(chart_regimes(res))

    st.subheader("What this means")
    st.write(what_this_means(res))

    c1, c2 = st.columns(2)
    c1.download_button("Download report.md", to_markdown(res),
                       file_name="edgejudge_report.md")
    c2.download_button("Download report.html (self-contained)", to_html(res),
                       file_name="edgejudge_report.html")

    st.divider()
    st.caption(f"criteria v{CRITERIA_VERSION} (pre-registered, not "
               f"user-configurable) · not financial advice · methodology: "
               f"README")


def main():
    st.set_page_config(page_title="EdgeJudge", page_icon="⚖️", layout="wide")
    st.markdown("<h1 style='font-family:ui-monospace,monospace'>EdgeJudge"
                "</h1>", unsafe_allow_html=True)
    st.write("Upload a trade log. Get a pre-registered verdict: does this "
             "strategy show a real edge, or luck dressed up? "
             "We never run your code — only a CSV of your trades.")

    fee_rt = fee_selector()
    date_format = date_format_selector()

    left, right = st.columns([3, 2])
    with left:
        up = st.file_uploader(
            "Trade-log CSV — columns: symbol, side, entry_time, exit_time, "
            "[entry_price, exit_price, qty]",
            type=["csv"])
        st.download_button("Download CSV template", TEMPLATE_CSV,
                           file_name="edgejudge_template.csv")
    with right:
        st.markdown("**No log handy?**")
        demo = st.button("⚖️ Try the demo verdict", type="primary",
                         width="stretch",
                         help="Audits a real strategy from our own research "
                              "campaign — a VWAP-fade that looked profitable "
                              "until judged properly.")
        st.caption("Real trade log, real market data, real FAIL — in ~30 "
                   "seconds you see exactly what a verdict looks like.")

    res = None
    if demo:
        if not DEMO_LOG.is_file():
            st.error("Demo log missing from this deployment.")
        else:
            with st.status("Auditing the demo strategy "
                           "(S02 VWAP-fade, LINKUSDT, 6 months)…",
                           expanded=True) as status:
                res, err = run_pipeline(DEMO_LOG.read_bytes(), DEMO_LOG.name,
                                        fee_rt, symbol=DEMO_SYMBOL,
                                        status=status)
                if err:
                    st.error(err)
                else:
                    status.update(label="Demo audit complete",
                                  state="complete", expanded=False)
    elif up is not None:
        raw = up.getvalue()
        with st.status("Auditing your log…", expanded=True) as status:
            res, err = run_pipeline(raw, up.name, fee_rt, status=status,
                                    date_format=date_format)
            if err:
                st.error(err)
            else:
                status.update(label="Audit complete", state="complete",
                              expanded=False)

    if res is not None:
        try:
            show_verdict(res)
        except Exception:                        # invariant: no tracebacks
            st.error("Unexpected problem rendering the report. The verdict "
                     "itself is unaffected — download links below the charts "
                     "may be unavailable; re-run the audit or simplify the "
                     "log to the template columns.")


main()
