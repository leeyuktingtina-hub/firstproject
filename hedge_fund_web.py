"""
hedge_fund_web.py
=================
Flask web server that wraps virattt/ai-hedge-fund's 19-agent pipeline.
Serves a dashboard at http://localhost:5001/hedge-fund

Prerequisites:
    bash setup_hedge_fund.sh     # run once
    export ANTHROPIC_API_KEY=... # or put in .env
    python hedge_fund_web.py
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty

from flask import Flask, jsonify, render_template, request, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()

# ── Patch sys.path so the cloned source is importable ──────────────────────
HEDGE_FUND_ROOT = Path(__file__).parent / "hedge_fund_src"
HEDGE_FUND_SRC  = HEDGE_FUND_ROOT / "src"

_source_ready = HEDGE_FUND_SRC.exists()
if _source_ready:
    for p in (str(HEDGE_FUND_ROOT), str(HEDGE_FUND_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)

# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(32))

# ── Analyst catalogue (matches ai-hedge-fund's ANALYST_CONFIG keys) ─────────
ANALYSTS = [
    {"key": "warren_buffett",        "name": "Warren Buffett",        "style": "Quality at fair price",         "icon": "🏦"},
    {"key": "charlie_munger",        "name": "Charlie Munger",        "style": "Quality businesses, moat focus","icon": "🔭"},
    {"key": "ben_graham",            "name": "Ben Graham",            "style": "Deep value & margin of safety", "icon": "📊"},
    {"key": "peter_lynch",           "name": "Peter Lynch",           "style": "Ten-bagger opportunities",      "icon": "🚀"},
    {"key": "phil_fisher",           "name": "Phil Fisher",           "style": "Growth with research",          "icon": "🔬"},
    {"key": "bill_ackman",           "name": "Bill Ackman",           "style": "Activist investing",            "icon": "⚔️"},
    {"key": "cathie_wood",           "name": "Cathie Wood",           "style": "Disruptive innovation",         "icon": "🤖"},
    {"key": "michael_burry",         "name": "Michael Burry",         "style": "Contrarian deep value",         "icon": "🐻"},
    {"key": "stanley_druckenmiller", "name": "Stanley Druckenmiller", "style": "Macro & asymmetric bets",       "icon": "🌐"},
    {"key": "aswath_damodaran",      "name": "Aswath Damodaran",      "style": "Rigorous valuation",            "icon": "📐"},
    {"key": "nassim_taleb",          "name": "Nassim Taleb",          "style": "Tail risk & antifragility",     "icon": "🎲"},
    {"key": "mohnish_pabrai",        "name": "Mohnish Pabrai",        "style": "Low-risk, high-return clones",  "icon": "♟️"},
    {"key": "rakesh_jhunjhunwala",   "name": "Rakesh Jhunjhunwala",   "style": "Long-term growth conviction",   "icon": "🌏"},
    {"key": "fundamentals_agent",    "name": "Fundamentals",          "style": "Profitability & health metrics","icon": "🏗️"},
    {"key": "technicals_agent",      "name": "Technical Analysis",    "style": "Price, momentum & patterns",    "icon": "📈"},
    {"key": "valuation_agent",       "name": "Valuation (DCF)",       "style": "Intrinsic value & scenarios",   "icon": "💰"},
    {"key": "sentiment_agent",       "name": "Sentiment",             "style": "News & insider signals",        "icon": "📰"},
]

ANALYST_LOOKUP = {a["key"]: a for a in ANALYSTS}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/hedge-fund")
def hedge_fund_page():
    return render_template(
        "hedge_fund.html",
        analysts=ANALYSTS,
        source_ready=_source_ready,
    )


@app.route("/api/hedge-fund/analyze", methods=["POST"])
def analyze():
    """
    Run the multi-agent hedge fund analysis and stream progress as SSE.
    Accepts JSON body:
    {
        "tickers":           ["NVDA", "MSFT"],
        "start_date":        "2024-01-01",
        "end_date":          "2025-01-01",
        "initial_cash":      100000,
        "selected_analysts": ["warren_buffett", "technicals_agent", ...],
        "model_provider":    "ANTHROPIC",
        "model_name":        "claude-opus-4-6"
    }
    """
    if not _source_ready:
        return jsonify({"error": "Run bash setup_hedge_fund.sh first."}), 500

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set."}), 500

    data = request.get_json(force=True)
    tickers_raw   = data.get("tickers", [])
    tickers       = [t.strip().upper() for t in tickers_raw if t.strip()]
    start_date    = data.get("start_date", (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
    end_date      = data.get("end_date",   datetime.now().strftime("%Y-%m-%d"))
    initial_cash  = float(data.get("initial_cash", 100_000))
    sel_analysts  = data.get("selected_analysts", [a["key"] for a in ANALYSTS])
    model_provider = data.get("model_provider", "ANTHROPIC")
    model_name     = data.get("model_name", "claude-opus-4-6")

    if not tickers:
        return jsonify({"error": "Please enter at least one ticker."}), 400

    portfolio = {
        "cash": initial_cash,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            t: {"long": 0, "short": 0,
                "long_cost_basis": 0.0, "short_cost_basis": 0.0,
                "short_margin_used": 0.0}
            for t in tickers
        },
        "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
    }

    q: Queue = Queue()

    def _run():
        try:
            from src.main import run_hedge_fund  # type: ignore

            q.put({"type": "status", "message": f"Starting analysis for {', '.join(tickers)}…"})

            result = run_hedge_fund(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                portfolio=portfolio,
                show_reasoning=True,
                selected_analysts=sel_analysts,
                model_name=model_name,
                model_provider=model_provider,
            )
            q.put({"type": "done", "result": result})

        except ImportError as e:
            q.put({"type": "error", "message": f"Import error — did you run setup_hedge_fund.sh? ({e})"})
        except Exception as e:
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    def _stream():
        yield "data: " + json.dumps({"type": "status", "message": "Queuing analysis…"}) + "\n\n"
        while True:
            try:
                msg = q.get(timeout=300)   # 5-minute hard timeout
            except Empty:
                yield "data: " + json.dumps({"type": "error", "message": "Timeout — analysis took too long."}) + "\n\n"
                break

            yield "data: " + json.dumps(msg) + "\n\n"

            if msg["type"] in ("done", "error"):
                break

    return Response(
        stream_with_context(_stream()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    if not _source_ready:
        print("\n  ⚠️  hedge_fund_src/ not found.")
        print("  Run: bash setup_hedge_fund.sh\n")
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  Hedge Fund Agent → http://localhost:{port}/hedge-fund\n")
    app.run(debug=False, port=port)
