"""
Investment Agent Web App
========================
Flask web server that exposes the stock investment agent as a chatbot.

Usage:
    export ANTHROPIC_API_KEY="your-key"   # or DEEPSEEK_API_KEY / GROQ_API_KEY
    python app.py
    # Open http://localhost:5000
"""

import os
import json
import secrets

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from investment_agent import (
    get_stock_data,
    screen_watchlist,
    analyze_portfolio,
    get_market_overview,
    TOOLS,
    TOOL_MAP,
    SYSTEM_PROMPT,
    DEFAULT_WATCHLIST,
)
from backtest import run_backtest

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(32))

# Server-side conversation store  { session_id: {"provider": str, "history": [...]} }
_sessions: dict[str, dict] = {}

# ── Provider catalogue ────────────────────────────────────────────────────────
CHAT_PROVIDERS = {
    "Anthropic": {
        "name": "Anthropic", "icon": "🟣",
        "env_var": "ANTHROPIC_API_KEY",
        "models": [
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6 (Recommended)"},
            {"id": "claude-haiku-4-6",  "name": "Claude Haiku 4.6 (Cheapest)"},
            {"id": "claude-opus-4-6",   "name": "Claude Opus 4.6 (Best)"},
        ],
    },
    "DeepSeek": {
        "name": "DeepSeek", "icon": "🔵",
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "models": [
            {"id": "deepseek-chat",     "name": "DeepSeek V3 (Best value)"},
            {"id": "deepseek-reasoner", "name": "DeepSeek R1 (Reasoning)"},
        ],
    },
    "Groq": {
        "name": "Groq", "icon": "⚡",
        "env_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Free tier)"},
            {"id": "mixtral-8x7b-32768",       "name": "Mixtral 8x7B"},
        ],
    },
}

# ── Tool format conversion (Anthropic → OpenAI) ───────────────────────────────
def _to_openai_tools(tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]

OAI_TOOLS = _to_openai_tools(TOOLS)


# ── Agent loops ───────────────────────────────────────────────────────────────

def run_anthropic_turn(user_message: str, history: list, model_name: str) -> tuple[str, list]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
            history.append({"role": "user",      "content": user_message})
            history.append({"role": "assistant", "content": text})
            return text, history

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_fn = TOOL_MAP.get(block.name)
            result  = tool_fn(block.input) if tool_fn else {"error": f"Unknown tool: {block.name}"}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})


def run_openai_turn(user_message: str, history: list, model_name: str, provider_key: str) -> tuple[str, list]:
    from openai import OpenAI
    prov    = CHAT_PROVIDERS[provider_key]
    api_key = os.environ.get(prov["env_var"], "")
    client  = OpenAI(api_key=api_key, base_url=prov.get("base_url"))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=OAI_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            text = msg.content or ""
            history.append({"role": "user",      "content": user_message})
            history.append({"role": "assistant", "content": text})
            return text, history

        # Append assistant turn with tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            tool_fn = TOOL_MAP.get(tc.function.name)
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result = tool_fn(args) if tool_fn else {"error": f"Unknown tool: {tc.function.name}"}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })


def run_agent_turn(user_message: str, history: list, provider: str, model_name: str) -> tuple[str, list]:
    if provider == "Anthropic":
        return run_anthropic_turn(user_message, history, model_name)
    return run_openai_turn(user_message, history, model_name, provider)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = secrets.token_hex(16)
    return render_template("index.html", watchlist=DEFAULT_WATCHLIST, providers=CHAT_PROVIDERS)


@app.route("/api/chat/providers")
def chat_providers():
    return jsonify(CHAT_PROVIDERS)


@app.route("/api/chat", methods=["POST"])
def chat():
    data         = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    provider     = data.get("provider", "Anthropic")
    model_name   = data.get("model",    "claude-sonnet-4-6")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    prov_info = CHAT_PROVIDERS.get(provider)
    if not prov_info:
        return jsonify({"error": f"Unknown provider: {provider}"}), 400

    api_key = os.environ.get(prov_info["env_var"], "")
    if not api_key:
        return jsonify({"error": f"{prov_info['env_var']} is not set on the server."}), 500

    sid      = session.setdefault("sid", secrets.token_hex(16))
    sess     = _sessions.setdefault(sid, {"provider": provider, "history": []})

    # Reset history when provider changes
    if sess["provider"] != provider:
        sess["provider"] = provider
        sess["history"]  = []

    try:
        reply, sess["history"] = run_agent_turn(
            user_message, sess["history"], provider, model_name
        )
        return jsonify({"reply": reply})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid:
        _sessions.pop(sid, None)
    return jsonify({"status": "ok"})


@app.route("/backtest")
def backtest_page():
    return render_template("backtest.html", watchlist=DEFAULT_WATCHLIST)


@app.route("/api/backtest/run", methods=["POST"])
def backtest_run():
    data     = request.get_json(force=True)
    raw      = data.get("tickers", "NVDA,MSFT")
    tickers  = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400
    result = run_backtest(
        tickers       = tickers,
        start_date    = data.get("start_date", "2020-01-01"),
        end_date      = data.get("end_date",   "2024-12-31"),
        initial_cash  = float(data.get("initial_cash", 100_000)),
        position_size = float(data.get("position_size", 0.20)),
        stop_loss     = float(data.get("stop_loss",    0.08)),
        take_profit   = float(data.get("take_profit",  0.25)),
    )
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Investment Agent → http://localhost:{port}")
    print(f"  Backtester      → http://localhost:{port}/backtest\n")
    app.run(debug=False, host="0.0.0.0", port=port)
