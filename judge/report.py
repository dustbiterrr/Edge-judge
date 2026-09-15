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

V101_NOTE = ("v1.0.1 fixes a regime-lookup lookahead affecting mid-bar "
             "entries; verdicts on bar-aligned logs are unchanged.")


def verdict_line(res) -> str:
    if res.verdict == "PASS":
        return (f"PASS — all 5 checks passed on {res.n_trades} trades "
                f"(criteria v{res.criteria_version}).")
    failed = [c for c in res.checks if not c.passed]
    return (f"FAIL — {len(failed)} of 5 checks failed "
            f"({failed[0].code} {failed[0].name.lower()}: "
            f"{failed[0].key_number}).")


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
}


def what_this_means(res) -> str:
    failed = [c.code for c in res.checks if not c.passed]
    if not failed:
        return ("All five pre-registered checks passed. This does not "
                "guarantee future profits, but the log shows a fee-surviving, "
                "direction-informed, regime-robust pattern that a coin flip "
                "does not explain. Forward-test with small size before "
                "trusting it.")
    return " ".join(INTERPRET[c] for c in failed[:3])


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
        badge = "PASS" if c.passed else "FAIL"
        rows.append(f"| {c.code} | {c.name} | **{badge}** | {c.key_number} | "
                    f"{c.detail} |")
    return "\n".join(rows)


def to_markdown(res) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    warn = "\n".join(f"- {w}" for w in res.warnings) or "- none"
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
{V101_NOTE}
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
        b = ("PASS", PASS_GREEN) if c.passed else ("FAIL", FAIL_RED)
        rows += (f"<tr><td>{c.code}</td><td>{c.name}</td>"
                 f"<td style='color:{b[1]};font-weight:700'>{b[0]}</td>"
                 f"<td class='mono'>{c.key_number}</td>"
                 f"<td>{c.detail}</td></tr>\n")
    warn = ("".join(f"<li>{html.escape(w)}</li>" for w in res.warnings)
            or "<li>none</li>")
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
{V101_NOTE}</footer>
</body></html>"""
