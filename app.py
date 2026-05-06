"""
Investment Agent Web App
========================
Flask web server that exposes the stock investment agent as a chatbot.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python app.py
    # Open http://localhost:5000
"""

import os
import json
import secrets

from flask import Flask, render_template, request, jsonify, session
import anthropic
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

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(32))

# Server-side conversation store  { session_id: [message, ...] }
# Messages contain native Anthropic SDK objects — must stay in memory, not cookies.
_sessions: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Agent loop (web-adapted: server-side history, text-only return)
# ---------------------------------------------------------------------------

def run_agent_turn(user_message: str, history: list) -> tuple[str, list]:
    """
    Run one user turn through the investment agent.
    history is mutated in-place AND returned for convenience.
    Returns (assistant_text, history).
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    history.append({"role": "user", "content": user_message})
    messages = list(history)  # shallow copy for the API loop

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "\n".join(b.text for b in response.content if b.type == "text")
            # Persist the final assistant turn back into history
            history.append({"role": "assistant", "content": response.content})
            return text, history

        if response.stop_reason != "tool_use":
            break

        # Execute tools
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_fn = TOOL_MAP.get(block.name)
            result = tool_fn(block.input) if tool_fn else {"error": f"Unknown tool: {block.name}"}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

    # Fallback
    text = "\n".join(b.text for b in response.content if b.type == "text")
    return text, history


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = secrets.token_hex(16)
    return render_template("index.html", watchlist=DEFAULT_WATCHLIST)


@app.route("/api/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set on the server."}), 500

    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    sid = session.setdefault("sid", secrets.token_hex(16))
    history = _sessions.setdefault(sid, [])

    try:
        reply, history = run_agent_turn(user_message, history)
        _sessions[sid] = history
        return jsonify({"reply": reply})
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid Anthropic API key."}), 401
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid:
        _sessions.pop(sid, None)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Investment Agent running at http://localhost:{port}\n")
    app.run(debug=False, port=port)
