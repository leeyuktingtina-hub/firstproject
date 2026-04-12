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

# ── Provider catalogue ────────────────────────────────────────────────────────
PROVIDERS = {
    "Anthropic": {
        "name": "Anthropic", "icon": "🟣", "tag": "Cloud",
        "env_var": "ANTHROPIC_API_KEY", "local": False,
        "models": [
            {"id": "claude-opus-4-6",   "name": "Claude Opus 4.6 (Best)"},
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6 (Fast)"},
            {"id": "claude-haiku-4-6",  "name": "Claude Haiku 4.6 (Cheapest)"},
        ],
    },
    "OpenAI": {
        "name": "OpenAI", "icon": "🟢", "tag": "Cloud",
        "env_var": "OPENAI_API_KEY", "local": False,
        "models": [
            {"id": "gpt-4.1", "name": "GPT-4.1 (Recommended)"},
            {"id": "gpt-5.4", "name": "GPT-5.4 (Latest)"},
        ],
    },
    "DeepSeek": {
        "name": "DeepSeek", "icon": "🔵", "tag": "Cloud",
        "env_var": "DEEPSEEK_API_KEY", "local": False,
        "models": [
            {"id": "deepseek-reasoner", "name": "DeepSeek R1 (Reasoner)"},
            {"id": "deepseek-chat",     "name": "DeepSeek V3 (Chat)"},
        ],
    },
    "Google": {
        "name": "Google", "icon": "🔴", "tag": "Cloud",
        "env_var": "GOOGLE_API_KEY", "local": False,
        "models": [
            {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro"},
        ],
    },
    "Groq": {
        "name": "Groq", "icon": "⚡", "tag": "Cloud · Fast",
        "env_var": "GROQ_API_KEY", "local": False,
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Fast)"},
            {"id": "mixtral-8x7b-32768",       "name": "Mixtral 8x7B"},
        ],
    },
    "xAI": {
        "name": "xAI (Grok)", "icon": "✖️", "tag": "Cloud",
        "env_var": "XAI_API_KEY", "local": False,
        "models": [
            {"id": "grok-4-0709", "name": "Grok 4"},
        ],
    },
    "Mistral": {
        "name": "Mistral", "icon": "🌊", "tag": "Cloud",
        "env_var": "MISTRAL_API_KEY", "local": False,
        "models": [
            {"id": "mistral-small3.1", "name": "Mistral Small 3.1"},
        ],
    },
    "OpenRouter": {
        "name": "OpenRouter", "icon": "🔀", "tag": "Cloud · Multi",
        "env_var": "OPENROUTER_API_KEY", "local": False,
        "models": [
            {"id": "qwen/qwen3-235b-a22b-thinking-2507", "name": "Qwen 3 235B Thinking"},
            {"id": "z-ai/glm-4.5",     "name": "GLM-4.5"},
            {"id": "z-ai/glm-4.5-air", "name": "GLM-4.5 Air"},
        ],
    },
    "GigaChat": {
        "name": "GigaChat", "icon": "🤖", "tag": "Cloud",
        "env_var": "GIGACHAT_API_KEY", "local": False,
        "models": [
            {"id": "GigaChat-2-Max", "name": "GigaChat 2 Max"},
        ],
    },
    "Azure OpenAI": {
        "name": "Azure OpenAI", "icon": "☁️", "tag": "Enterprise",
        "env_var": "AZURE_OPENAI_API_KEY", "local": False,
        "models": [
            {"id": "azure-deployment", "name": "Custom Azure Deployment"},
        ],
    },
    "Ollama": {
        "name": "Ollama", "icon": "🦙", "tag": "Local · Free",
        "env_var": None, "local": True,
        "models": [
            {"id": "llama3.3:70b-instruct-q4_0", "name": "Llama 3.3 70B (Best)"},
            {"id": "llama3.1:latest",             "name": "Llama 3.1"},
            {"id": "gemma3:27b",                  "name": "Gemma 3 27B"},
            {"id": "gemma3:12b",                  "name": "Gemma 3 12B"},
            {"id": "gemma3:4b",                   "name": "Gemma 3 4B"},
            {"id": "qwen3:30b-a3b",               "name": "Qwen 3 30B"},
            {"id": "qwen3:8b",                    "name": "Qwen 3 8B"},
            {"id": "qwen3:4b",                    "name": "Qwen 3 4B"},
            {"id": "mistral-small3.1",            "name": "Mistral Small 3.1"},
            {"id": "gpt-oss:20b",                 "name": "GPT-OSS 20B"},
            {"id": "gpt-oss:120b",                "name": "GPT-OSS 120B"},
        ],
    },
}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/hedge-fund")
def hedge_fund_page():
    return render_template(
        "hedge_fund.html",
        analysts=ANALYSTS,
        providers=PROVIDERS,
        source_ready=_source_ready,
    )


@app.route("/api/hedge-fund/providers")
def get_providers():
    return jsonify(PROVIDERS)


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
        "model_provider":    "Anthropic",
        "model_name":        "claude-opus-4-6"
    }
    """
    if not _source_ready:
        return jsonify({"error": "Run bash setup_hedge_fund.sh first."}), 500

    data = request.get_json(force=True)
    tickers_raw   = data.get("tickers", [])
    tickers       = [t.strip().upper() for t in tickers_raw if t.strip()]
    start_date    = data.get("start_date", (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
    end_date      = data.get("end_date",   datetime.now().strftime("%Y-%m-%d"))
    initial_cash  = float(data.get("initial_cash", 100_000))
    sel_analysts  = data.get("selected_analysts", [a["key"] for a in ANALYSTS])
    model_provider = data.get("model_provider", "Anthropic")
    model_name     = data.get("model_name", "claude-opus-4-6")

    # Validate API key for the selected provider
    provider_info = PROVIDERS.get(model_provider, {})
    env_var = provider_info.get("env_var")
    if env_var and not os.environ.get(env_var):
        return jsonify({"error": f"{env_var} is not set. Export it or add it to your .env file."}), 500

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
