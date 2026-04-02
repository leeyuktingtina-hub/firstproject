"""
Stock Investment Agent
======================
An AI-powered investment assistant that monitors stocks, identifies buying
opportunities using technical analysis, and helps you make disciplined
medium/long-term investment decisions.

Usage:
    python investment_agent.py

Set your ANTHROPIC_API_KEY environment variable before running.
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Any

import anthropic
import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)

# ---------------------------------------------------------------------------
# AI-focused watchlist — feel free to add / remove tickers
# ---------------------------------------------------------------------------
DEFAULT_WATCHLIST = [
    "NVDA",   # NVIDIA — AI chips, data center
    "AMD",    # AMD — GPU / CPU competitor
    "MSFT",   # Microsoft — Azure AI, Copilot
    "GOOGL",  # Alphabet — Gemini, TPUs, Search AI
    "META",   # Meta — LLaMA, AI infrastructure
    "TSLA",   # Tesla — FSD, Dojo supercomputer
    "ASML",   # ASML — EUV lithography (AI chip supply chain)
    "AVGO",   # Broadcom — AI networking chips
    "TSM",    # TSMC — leading AI chip manufacturer (ADR)
    "ORCL",   # Oracle — cloud AI databases
    "CRM",    # Salesforce — AI CRM
    "PLTR",   # Palantir — AI data analytics
]

# ---------------------------------------------------------------------------
# Technical analysis helpers
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_macd(series: pd.Series) -> dict:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
        "bullish_crossover": bool(
            macd_line.iloc[-1] > signal_line.iloc[-1]
            and macd_line.iloc[-2] <= signal_line.iloc[-2]
        ),
    }


def compute_bollinger(series: pd.Series, period: int = 20) -> dict:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    price = float(series.iloc[-1])
    upper_val = float(upper.iloc[-1])
    lower_val = float(lower.iloc[-1])
    sma_val = float(sma.iloc[-1])
    band_width = (upper_val - lower_val) / sma_val if sma_val else 0
    pct_b = (price - lower_val) / (upper_val - lower_val) if (upper_val - lower_val) else 0.5
    return {
        "upper": round(upper_val, 2),
        "lower": round(lower_val, 2),
        "middle": round(sma_val, 2),
        "bandwidth": round(band_width, 4),
        "percent_b": round(pct_b, 4),
        "near_lower_band": pct_b < 0.2,
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_stock_data(ticker: str, period: str = "6mo") -> dict:
    """Fetch price history + technical indicators for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return {"error": f"No data found for {ticker}"}

        info = stock.info or {}
        close = hist["Close"]
        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else price
        change_pct = (price - prev_close) / prev_close * 100

        # Moving averages
        ma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        ma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # RSI
        rsi = compute_rsi(close) if len(close) >= 15 else None

        # MACD
        macd_data = compute_macd(close) if len(close) >= 30 else {}

        # Bollinger Bands
        bb_data = compute_bollinger(close) if len(close) >= 20 else {}

        # 52-week range
        week52_high = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        week52_low = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())
        pct_from_high = (price - week52_high) / week52_high * 100

        # Volume trend
        avg_vol_20 = float(hist["Volume"].tail(20).mean())
        latest_vol = float(hist["Volume"].iloc[-1])
        vol_ratio = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

        return {
            "ticker": ticker.upper(),
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "price": round(price, 2),
            "change_pct_today": round(change_pct, 2),
            "market_cap_b": round(info.get("marketCap", 0) / 1e9, 1),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "52w_high": round(week52_high, 2),
            "52w_low": round(week52_low, 2),
            "pct_from_52w_high": round(pct_from_high, 1),
            "ma_20": round(ma_20, 2) if ma_20 else None,
            "ma_50": round(ma_50, 2) if ma_50 else None,
            "ma_200": round(ma_200, 2) if ma_200 else None,
            "above_ma50": price > ma_50 if ma_50 else None,
            "above_ma200": price > ma_200 if ma_200 else None,
            "rsi_14": rsi,
            "macd": macd_data,
            "bollinger": bb_data,
            "volume_vs_avg": round(vol_ratio, 2),
            "analyst_target": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey", "N/A"),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def screen_watchlist(tickers: list[str]) -> dict:
    """Run a quick screen across all tickers and flag buying opportunities."""
    results = []
    for ticker in tickers:
        data = get_stock_data(ticker, period="6mo")
        if "error" in data:
            continue

        score = 0
        signals = []

        rsi = data.get("rsi_14")
        if rsi is not None:
            if rsi < 35:
                score += 3
                signals.append(f"RSI oversold ({rsi})")
            elif rsi < 45:
                score += 1
                signals.append(f"RSI approaching oversold ({rsi})")
            elif rsi > 70:
                score -= 2
                signals.append(f"RSI overbought ({rsi}) — avoid chasing")

        bb = data.get("bollinger", {})
        if bb.get("near_lower_band"):
            score += 2
            signals.append("Near Bollinger lower band — potential support")

        macd = data.get("macd", {})
        if macd.get("bullish_crossover"):
            score += 2
            signals.append("MACD bullish crossover")

        pct_from_high = data.get("pct_from_52w_high", 0)
        if pct_from_high < -30:
            score += 2
            signals.append(f"{pct_from_high:.0f}% below 52w high — significant pullback")
        elif pct_from_high < -15:
            score += 1
            signals.append(f"{pct_from_high:.0f}% below 52w high — pullback")

        if data.get("above_ma200") is False and data.get("above_ma50") is False:
            score -= 1
            signals.append("Below both MA50 and MA200 — downtrend caution")
        elif data.get("above_ma200") and not data.get("above_ma50"):
            signals.append("Below MA50 but above MA200 — possible dip in uptrend")
            score += 1

        results.append({
            "ticker": data["ticker"],
            "name": data["name"],
            "price": data["price"],
            "change_pct": data["change_pct_today"],
            "rsi": rsi,
            "pct_from_high": pct_from_high,
            "buy_score": score,
            "signals": signals,
            "recommendation": data.get("recommendation", "N/A"),
        })

    results.sort(key=lambda x: x["buy_score"], reverse=True)
    return {"screened": len(results), "stocks": results}


def analyze_portfolio(holdings: list[dict]) -> dict:
    """
    Analyze a portfolio of holdings.
    holdings: list of {"ticker": "NVDA", "shares": 10, "avg_cost": 500.0}
    """
    portfolio_data = []
    total_value = 0.0
    total_cost = 0.0

    for h in holdings:
        ticker = h["ticker"]
        shares = h.get("shares", 0)
        avg_cost = h.get("avg_cost", 0)
        data = get_stock_data(ticker, period="3mo")
        if "error" in data:
            continue

        price = data["price"]
        cost_basis = shares * avg_cost
        current_value = shares * price
        gain_loss = current_value - cost_basis
        gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0

        portfolio_data.append({
            "ticker": ticker,
            "name": data["name"],
            "shares": shares,
            "avg_cost": avg_cost,
            "current_price": price,
            "current_value": round(current_value, 2),
            "cost_basis": round(cost_basis, 2),
            "gain_loss": round(gain_loss, 2),
            "gain_loss_pct": round(gain_loss_pct, 1),
            "rsi_14": data.get("rsi_14"),
            "above_ma50": data.get("above_ma50"),
            "above_ma200": data.get("above_ma200"),
            "pct_from_52w_high": data.get("pct_from_52w_high"),
        })

        total_value += current_value
        total_cost += cost_basis

    total_gain_loss = total_value - total_cost
    total_gain_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0

    return {
        "holdings": portfolio_data,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_gain_loss": round(total_gain_loss, 2),
        "total_gain_loss_pct": round(total_gain_pct, 1),
        "num_positions": len(portfolio_data),
    }


def get_market_overview() -> dict:
    """Fetch key market indices to understand macro conditions."""
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "VIX (Fear Index)": "^VIX",
        "Semiconductor ETF": "SOXX",
        "AI/Tech ETF": "QQQ",
    }
    results = {}
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                continue
            price = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
            change = (price - prev) / prev * 100
            results[name] = {
                "price": round(price, 2),
                "change_pct": round(change, 2),
            }
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_stock_data",
        "description": (
            "Fetch real-time price, technical indicators (RSI, MACD, Bollinger Bands, "
            "moving averages), and fundamental data for a single stock ticker. "
            "Use this for deep analysis of a specific stock."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'NVDA', 'MSFT'"
                },
                "period": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y"],
                    "description": "History period for technical analysis (default: 6mo)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "screen_watchlist",
        "description": (
            "Screen a list of stock tickers for buying opportunities using technical "
            "analysis. Returns a ranked list sorted by buying opportunity score. "
            "Use this to find which stocks on the watchlist are at good entry points."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to screen"
                }
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "analyze_portfolio",
        "description": (
            "Analyze the user's current stock holdings. Returns P&L, current "
            "technical indicators for each position, and overall portfolio performance. "
            "Use this to give hold/trim/add advice on existing positions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "description": "List of current positions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "shares": {"type": "number"},
                            "avg_cost": {"type": "number", "description": "Average purchase price per share"}
                        },
                        "required": ["ticker", "shares", "avg_cost"]
                    }
                }
            },
            "required": ["holdings"]
        }
    },
    {
        "name": "get_market_overview",
        "description": (
            "Get current market conditions: S&P 500, NASDAQ, VIX fear index, "
            "semiconductor and tech ETFs. Use this to understand macro backdrop "
            "before making buy/sell recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

TOOL_MAP = {
    "get_stock_data": lambda args: get_stock_data(
        args["ticker"], args.get("period", "6mo")
    ),
    "screen_watchlist": lambda args: screen_watchlist(args["tickers"]),
    "analyze_portfolio": lambda args: analyze_portfolio(args["holdings"]),
    "get_market_overview": lambda args: get_market_overview(),
}

# ---------------------------------------------------------------------------
# System prompt — investing philosophy embedded here
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional stock investment advisor specialising in medium to long-term investing (holding periods of 6 months to 3+ years), with a focus on AI and technology stocks.

Your job is to help the user make disciplined, data-driven investment decisions. You have access to real-time stock data and technical analysis tools.

**User's investing challenges you must help with:**
1. They tend to BUY stocks when they are already elevated / overbought — help them identify better entry points
2. They take profits TOO EARLY (exit at +10% when the trend is still strong) — help them hold winners longer
3. They PANIC SELL during normal corrections — help them distinguish a healthy pullback from a genuine trend break
4. They are interested in AI stocks and the AI supply chain (chips, cloud, data, software)

**Your analysis framework:**
- RSI < 40: Potential buying zone (oversold). RSI < 30: Strong buy signal. RSI > 70: Overbought — DO NOT recommend buying here
- Price near Bollinger lower band: Support level, potential entry
- MACD bullish crossover: Momentum turning positive
- Price 15-30%+ below 52-week high: Meaningful pullback
- Price above MA200: Long-term uptrend intact (positive for holding)
- Price below MA50 but above MA200: Dip in an uptrend — often a good buying zone

**Holding guidance:**
- If RSI < 70, price above MA200, and thesis intact → HOLD / ADD
- If RSI > 75 and significantly above MA50 → consider trimming a portion, not selling everything
- A 10-15% drop in an uptrend is NORMAL — do not panic sell without checking the trend is broken

**Communication style:**
- Be direct and specific with price levels
- Always give a clear BUY / HOLD / AVOID recommendation
- Explain the "why" in plain language
- Warn the user if they are about to make an emotional mistake
- Start by checking market conditions before giving individual stock advice

Always call get_market_overview first when starting a session, then use the other tools to answer the user's question."""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_message: str, conversation_history: list) -> tuple[str, list]:
    """Run one turn of the investment agent. Returns (response_text, updated_history)."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    conversation_history.append({"role": "user", "content": user_message})

    messages = conversation_history.copy()

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect all content blocks for history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text response
            text_blocks = [b.text for b in response.content if b.type == "text"]
            final_text = "\n".join(text_blocks)
            # Update conversation_history with the full exchange
            conversation_history.append({"role": "assistant", "content": response.content})
            return final_text, conversation_history

        if response.stop_reason != "tool_use":
            break

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            print(f"{Fore.CYAN}  [Tool] {tool_name}({json.dumps(tool_input, separators=(',', ':'))[:80]}...){Style.RESET_ALL}")

            if tool_name in TOOL_MAP:
                result = TOOL_MAP[tool_name](tool_input)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

    # Fallback
    text_blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_blocks), conversation_history


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def print_banner():
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"  Stock Investment Agent — AI/Tech Focus")
    print(f"  Powered by Claude Opus 4.6 + Real-time Market Data")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Default watchlist: {', '.join(DEFAULT_WATCHLIST)}{Style.RESET_ALL}")
    print(f"\nExample questions:")
    print("  • Which AI stocks are at a good buying point right now?")
    print("  • I own 10 shares of NVDA at $500 avg cost, should I hold?")
    print("  • Analyse MSFT in detail")
    print("  • Screen my watchlist for opportunities")
    print("  • I'm tempted to sell after a 10% drop, what should I do?")
    print(f"\n{Fore.CYAN}Type 'quit' or 'exit' to stop.{Style.RESET_ALL}\n")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"{Fore.RED}Error: ANTHROPIC_API_KEY environment variable not set.{Style.RESET_ALL}")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    print_banner()

    conversation_history: list = []

    # First turn: greet with a market snapshot
    print(f"{Fore.YELLOW}Agent: {Style.RESET_ALL}", end="", flush=True)
    greeting = (
        "Hello! Please give me a quick market overview and tell me which AI stocks on "
        f"the watchlist {DEFAULT_WATCHLIST} look like good buying opportunities right now."
    )
    response, conversation_history = run_agent(greeting, conversation_history)
    print(response)
    print()

    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! Invest wisely.")
            break

        print(f"\n{Fore.YELLOW}Agent: {Style.RESET_ALL}", end="", flush=True)
        response, conversation_history = run_agent(user_input, conversation_history)
        print(response)
        print()


if __name__ == "__main__":
    main()
