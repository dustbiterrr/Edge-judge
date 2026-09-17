"""
report.py — verdict line, markdown / self-contained HTML reports, charts.

No softening language.  FAIL is FAIL and says why in plain English.
Interpretation text is template-generated from the actual failures — no LLM.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone

import numpy as np
# Figure API, not pyplot: the global pyplot figure registry is not
# thread-safe under concurrent Streamlit sessions
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PASS_GREEN = "#2E6B4F"
FAIL_RED = "#B3402E"
UNVERIFIABLE_AMBER = "#8A6D1F"

VERSION_NOTES = ("v1.1.0 adds C6 look-ahead (signal_time vs entry_time); "
                 "without a signal_time column C6 is reported UNVERIFIABLE, "
                 "never passed. C1-C5 unchanged since v1.0; v1.0.1 fixed a "
                 "regime-lookup lookahead affecting mid-bar entries.")
V101_NOTE = VERSION_NOTES          # kept for callers of the old name

BADGE_COLOR = {"PASS": PASS_GREEN, "FAIL": FAIL_RED,
               "UNVERIFIABLE": UNVERIFIABLE_AMBER}


def _unverifiable_tail(res) -> str:
    unv = res.unverifiable
    if not unv:
        return ""
    return "; " + ", ".join(f"{c.code} {c.name.lower()} UNVERIFIABLE "
                            f"({c.key_number})" for c in unv)


def verdict_line(res) -> str:
    """One line that never hides an unverifiable check behind a PASS."""
    evald = res.evaluated
    tail = _unverifiable_tail(res)
    if res.verdict == "PASS":
        if tail:
            return (f"PASS — {len(evald)} of {len(evald)} evaluated checks "
                    f"passed on {res.n_trades} trades{tail} "
                    f"(criteria v{res.criteria_version}).")
        return (f"PASS — all {len(evald)} checks passed on {res.n_trades} "
                f"trades (criteria v{res.criteria_version}).")
    failed = [c for c in evald if c.status == "FAIL"]
    return (f"FAIL — {len(failed)} of {len(evald)} checks failed "
            f"({failed[0].code} {failed[0].name.lower()}: "
            f"{failed[0].key_number}){tail}.")


INTERPRET = {
    "C1": "There are not enough independent trades to judge anything — "
          "collect more history before drawing any conclusion.",
    "C2": "After realistic fees the average trade loses money. The raw "
          "signal, whatever it is, is smaller than the cost of trading it.",
    "C3": "Your direction choices perform no better than a coin flip on the "
          "same entries and exits. The PnL pattern is explainable by luck.",
    "C4": "The strategy made money in one half of the period and lost in the "
          "other. That is the signature of a market regime doing the work, "
          "not an edge.",
    "C5": "Profits are concentrated in one market regime. When that regime "
          "ends, the strategy has nothing.",
    "C6": "Entries are logged at or before the moment their signal could "
          "have been known. That is look-ahead: the backtest traded on "
          "information it did not yet have, and every other number in this "
          "report inherits it. Fix the entry timing before reading anything "
          "else.",
}

UNVERIFIABLE_TEXT = {
    "C6": "Look-ahead could NOT be checked: the log has no signal_time "
          "column, so the judge cannot tell whether entries used information "
          "from the bar they were booked on. A look-ahead-biased log passes "
          "all five statistical checks. Add a signal_time (decision time) "
          "column and re-run before trusting this verdict.",
}

PASS_VERIFIED = ("All six pre-registered checks passed. This does not "
                 "guarantee future profits, but the log shows a fee-surviving, "
                 "direction-informed, regime-robust pattern that a coin flip "
                 "does not explain, with entries timed after their signals. "
                 "Forward-test with small size before trusting it.")

PASS_PARTIAL = ("The five statistical checks passed, but the one check that "
                "would catch look-ahead could not run, so this is not a clean "
                "verdict.")

# Printed under every PASS - CLI, reports, web app.  The first question a
# reader asks under a PASS is "so I have an edge?"; this is the answer.
PASS_MEANS = (
    "What a PASS means: on the trades in this log, after the fee named "
    "above, your side choices beat a coin flip at the 2.5% level, the result "
    "held in both halves of the period, it was not carried by one regime, "
    "and - if C6 ran - entries were booked after their signals. What it does "
    "not mean: that the strategy was pre-registered, that these trades are "
    "out-of-sample, that your costs are right, that this was not the best of "
    "many variants you tried, that the data behind the signals existed when "
    "they fired, that the signal_time column is truthful, or that it will "
    "work next month. Read it together with the scope list in the README "
    "(\"What this is not\") and with the state of C6: a PASS with C6 "
    "UNVERIFIABLE holds only if the timing was honest, and the judge could "
    "not check the timing.")


def what_this_means(res) -> str:
    failed = [c.code for c in res.checks if c.status == "FAIL"]
    unv = [UNVERIFIABLE_TEXT.get(c.code, f"{c.code} could not be checked.")
           for c in res.unverifiable]
    if not failed:
        if not unv:
            return " ".join([PASS_VERIFIED, PASS_MEANS])
        return " ".join([PASS_PARTIAL] + unv + [PASS_MEANS])
    return " ".join([INTERPRET[c] for c in failed[:3]] + unv)


# ── charts (matplotlib -> PNG bytes; identical in app and HTML report) ──────

def _fig(figsize) -> tuple:
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return buf.getvalue()


def _placeholder(title: str) -> bytes:
    fig, ax = _fig((7, 2.2))
    ax.text(0.5, 0.5, "no evaluable trades — nothing to chart",
            ha="center", va="center", fontsize=11, color="#777")
    ax.set_title(title)
    ax.set_axis_off()
    return _png(fig)


def chart_bootstrap(res) -> bytes:
    if res.trades is None or len(res.trades) == 0:
        return _placeholder("C3 — does your direction beat a coin flip?")
    fig, ax = _fig((7, 3.2))
    ax.hist(res.bootstrap_totals, bins=50, color="#8FA6B8",
            label="1000 coin-flip logs")
    color = PASS_GREEN if res.client_pctl >= 97.5 else FAIL_RED
    ax.axvline(res.client_total, color=color, lw=2.5,
               label=f"your log ({res.client_pctl:.1f}th pctl)")
    ax.set_xlabel("total net PnL % (fees included)")
    ax.set_ylabel("count")
    ax.set_title("C3 — does your direction beat a coin flip?")
    ax.legend(frameon=False, fontsize=9)
    return _png(fig)


def chart_equity(res) -> bytes:
    if res.trades is None or len(res.trades) == 0:
        return _placeholder("C4 — equity, non-overlapping execution")
    tr = res.trades.sort_values("exit_time")
    eq = tr["net_pct"].cumsum().values
    t = tr["exit_time"].dt.tz_localize(None).values
    mid_time = (tr["entry_time"].min() + (tr["entry_time"].max()
                                          - tr["entry_time"].min()) / 2
                ).tz_localize(None)
    fig, ax = _fig((7, 3.2))
    ax.axvspan(t[0], np.datetime64(mid_time), color="#8FA6B8", alpha=0.12,
               label=f"1st half  mean {res.half_means[0]:+.3f}%/trade")
    ax.axvspan(np.datetime64(mid_time), t[-1], color="#C9B458", alpha=0.10,
               label=f"2nd half  mean {res.half_means[1]:+.3f}%/trade")
    ax.plot(t, eq, color="#233742", lw=1.6)
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_ylabel("cumulative net PnL %")
    ax.set_title("C4 — equity, non-overlapping execution")
    ax.legend(frameon=False, fontsize=9, loc="best")
    return _png(fig)


def chart_regimes(res) -> bytes:
    rt = res.regime_table
    if rt is None or len(rt) == 0 or res.trades is None or len(res.trades) == 0:
        return _placeholder("C5 — PnL by market regime (24h drift at entry)")
    fig, ax = _fig((7, 3.0))
    order = [r for r in ("UP", "FLAT", "DOWN") if r in set(rt["regime"])]
    rt = rt.set_index("regime").loc[order].reset_index()
    colors = [PASS_GREEN if v > 0 else FAIL_RED for v in rt["sum"]]
    ax.bar(rt["regime"], rt["sum"], color=colors)
    total = res.trades["net_pct"].sum()
    for i, r in rt.iterrows():
        share = (r["sum"] / total * 100) if abs(total) > 1e-9 else 0.0
        ax.text(i, r["sum"], f"{r['sum']:+.1f}%\n(n={int(r['count'])}, "
                f"{share:+.0f}% of total)", ha="center",
                va="bottom" if r["sum"] >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("net PnL % contribution")
    ax.set_title("C5 — PnL by market regime (24h drift at entry)")
    ax.margins(y=0.25)
    return _png(fig)


# ── documents ────────────────────────────────────────────────────────────────

def _criteria_rows(res) -> str:
    rows = []
    for c in res.checks:
        badge = c.status
        rows.append(f"| {c.code} | {c.name} | **{badge}** | {c.key_number} | "
                    f"{c.detail} |")
    return "\n".join(rows)


def to_markdown(res) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    warn = "\n".join(f"- {w}" for w in [res.cost_note, *res.warnings])
    return f"""# EdgeJudge report

**{verdict_line(res)}**

| # | check | result | key number | detail |
|---|-------|--------|------------|--------|
{_criteria_rows(res)}

## What this means

{what_this_means(res)}

## Notes

{warn}

---
criteria v{res.criteria_version} (pre-registered, not user-configurable) ·
fee round-trip {res.fee_rt:.3f}% · {res.n_trades} non-overlapping trades ·
generated {ts} · not financial advice · methodology: README
{VERSION_NOTES}
"""


def to_html(res) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    color = PASS_GREEN if res.verdict == "PASS" else FAIL_RED
    imgs = ""
    for png in (chart_bootstrap(res), chart_equity(res), chart_regimes(res)):
        b64 = base64.b64encode(png).decode()
        imgs += (f'<img src="data:image/png;base64,{b64}" '
                 f'style="max-width:100%;margin:12px 0"/>\n')
    rows = ""
    for c in res.checks:
        b = (c.status, BADGE_COLOR[c.status])
        rows += (f"<tr><td>{c.code}</td><td>{c.name}</td>"
                 f"<td style='color:{b[1]};font-weight:700'>{b[0]}</td>"
                 f"<td class='mono'>{c.key_number}</td>"
                 f"<td>{c.detail}</td></tr>\n")
    warn = "".join(f"<li>{html.escape(w)}</li>"
                   for w in [res.cost_note, *res.warnings])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>EdgeJudge report</title><style>
body{{font-family:Georgia,serif;max-width:860px;margin:32px auto;padding:0 16px;color:#233742}}
.mono{{font-family:ui-monospace,Consolas,monospace}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px 8px;font-size:14px}}
.verdict{{font-size:22px;font-weight:700;color:{color}}}
footer{{font-size:12px;color:#777;margin-top:24px}}
</style></head><body>
<h1>EdgeJudge</h1>
<p class="verdict">{verdict_line(res)}</p>
<table><tr><th>#</th><th>check</th><th>result</th><th>key number</th><th>detail</th></tr>
{rows}</table>
<h2>Charts</h2>
{imgs}
<h2>What this means</h2><p>{what_this_means(res)}</p>
<h2>Notes</h2><ul>{warn}</ul>
<footer>criteria v{res.criteria_version} (pre-registered, not user-configurable)
· fee RT {res.fee_rt:.3f}% · {res.n_trades} non-overlapping trades ·
generated {ts} · not financial advice · methodology: README<br>
{VERSION_NOTES}</footer>
</body></html>"""
