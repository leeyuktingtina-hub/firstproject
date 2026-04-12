# Stock Investment Agent — How It All Works

## Overview

You have **two web apps** built on top of each other:

| App | Port | What it does |
|-----|------|--------------|
| Investment Chatbot | 5000 | Chat with Claude about your stocks |
| Hedge Fund Dashboard | 5001 | Run 19 AI analyst agents on any stock |

Both apps use **Claude** as the AI brain and **yfinance** for free real-time market data.

---

## App 1 — Investment Chatbot (`app.py`)

```
You (browser)
    ↓  type a question
Flask (app.py, port 5000)
    ↓  loads your conversation history (server-side session)
Claude Opus 4.6
    ↓  decides which tools to call (agentic loop)
    ├── get_stock_data(ticker)      → yfinance: price, RSI, MACD, Bollinger Bands
    ├── screen_watchlist(tickers)   → yfinance: scores 12 AI/tech stocks for buy signals
    ├── analyze_portfolio(holdings) → yfinance: P&L + technicals on your positions
    └── get_market_overview()       → yfinance: S&P500, NASDAQ, VIX, QQQ, SOXX
    ↓  tool results fed back to Claude
Claude reasons → writes response
    ↓
Your browser displays the reply
```

**Key design decisions:**
- Claude decides which tools to use — you don't have to specify, just ask naturally
- Conversation history is kept server-side (not in the browser) because Anthropic SDK objects aren't JSON-serializable
- System prompt encodes your investing style: RSI < 40 = buy zone, RSI > 70 = avoid, MA200 = hold signal, panic-sell guardrails

---

## App 2 — Hedge Fund Dashboard (`hedge_fund_web.py`)

```
You (browser)
    ↓  enter tickers, pick analysts, click Analyze
Flask (hedge_fund_web.py, port 5001)
    ↓  starts background thread
LangGraph (from virattt/ai-hedge-fund GitHub repo)
    ↓  orchestrates 19 agents in a directed graph
    │
    ├── 13 Investor Personality Agents (run in parallel)
    │     Warren Buffett    → quality at fair price
    │     Charlie Munger    → moat & quality businesses
    │     Ben Graham        → deep value & margin of safety
    │     Peter Lynch       → ten-bagger opportunities
    │     Phil Fisher       → growth with research
    │     Bill Ackman       → activist investing
    │     Cathie Wood       → disruptive innovation
    │     Michael Burry     → contrarian deep value
    │     Stanley Druckenmiller → macro & asymmetric bets
    │     Aswath Damodaran  → rigorous DCF valuation
    │     Nassim Taleb      → tail risk & antifragility
    │     Mohnish Pabrai    → low-risk high-return clones
    │     Rakesh Jhunjhunwala → long-term growth conviction
    │
    ├── 4 Quantitative Analyst Agents (run in parallel)
    │     Fundamentals      → profitability & health metrics
    │     Technical Analysis → price, momentum & patterns
    │     Valuation (DCF)   → intrinsic value & scenarios
    │     Sentiment         → news & insider signals
    │
    └── Portfolio Manager Agent (runs last)
          sees all 17 signals → makes final decision
          output: BUY / SELL / HOLD / SHORT / COVER
                  quantity, confidence %, reasoning
    ↓
Results stream back to browser in real-time (SSE)
    ↓
Dashboard shows: signal cards per analyst + decisions table
```

---

## What Each Layer Does

### LangGraph (the GitHub repo)
Provides the **skeleton**:
- Defines 19 agents and wires them into a directed graph
- Handles orchestration: which agent runs when, how data flows between them
- Manages state across the full analysis run

### Claude (the AI brain)
Provides the **intelligence inside every agent**:
- Each agent fetches data, then sends it to Claude with a personality prompt
- Example: *"You are Warren Buffett. Here are NVDA's financials. Give a bullish/bearish signal and explain your reasoning."*
- Claude reasons through the data and returns a structured signal
- Without Claude, the agents are empty shells

### yfinance (free market data)
Provides the **raw data** every agent needs:
- Real-time & historical prices
- Income statement, balance sheet, cash flow (quarterly & annual)
- Insider trades
- Company news
- Market cap, PE ratio, and 40+ other metrics

### yfinance_adapter.py (our patch)
The original GitHub repo requires a **paid API** (`financialdatasets.ai`).
We wrote a drop-in replacement that fetches the same data from yfinance for free.
`setup_hedge_fund.sh` copies it over the original file automatically.

---

## Data Flow — Full Picture

```
User picks: NVDA, 2024-01-01 to 2025-01-01, all analysts
                    ↓
         hedge_fund_web.py receives POST request
                    ↓
         Background thread starts
                    ↓
         LangGraph kicks off the pipeline
                    ↓
  ┌─────────────────────────────────────────┐
  │  Each of 19 agents does:                │
  │  1. Fetch data via yfinance_adapter.py  │
  │  2. Format data into a prompt           │
  │  3. Call Claude API                     │
  │  4. Parse Claude's response             │
  │  5. Return: signal + confidence + text  │
  └─────────────────────────────────────────┘
                    ↓
         Portfolio Manager Agent
         (sees all 19 signals, calls Claude again)
         → Final decision: BUY 150 shares, 85% confidence
                    ↓
         Result put into Queue
                    ↓
  SSE stream sends result to browser
                    ↓
  Dashboard renders analyst cards + decisions table
```

---

## File Structure

```
firstproject/
├── app.py                  # Investment chatbot Flask server (port 5000)
├── investment_agent.py     # Claude tool definitions + agentic loop
├── hedge_fund_web.py       # Hedge fund dashboard Flask server (port 5001)
├── yfinance_adapter.py     # Free replacement for paid financialdatasets.ai API
├── setup_hedge_fund.sh     # One-time setup: clone repo + patch data layer
├── requirements.txt        # All Python dependencies
├── .env                    # Your API key (never committed to git)
├── templates/
│   ├── index.html          # Chatbot UI (dark theme)
│   └── hedge_fund.html     # Hedge fund dashboard UI
└── hedge_fund_src/         # Cloned virattt/ai-hedge-fund (created by setup script)
    └── src/
        ├── main.py         # run_hedge_fund() entry point
        ├── graph/          # LangGraph pipeline definition
        ├── agents/         # 19 individual agent definitions
        └── tools/
            └── api.py      # ← replaced by our yfinance_adapter.py
```

---

## How to Run

### Investment Chatbot
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # or put in .env file
python app.py
# open http://localhost:5000
```

### Hedge Fund Dashboard
```bash
# First time only:
bash setup_hedge_fund.sh

# Every time:
export ANTHROPIC_API_KEY="sk-ant-..."   # or put in .env file
python hedge_fund_web.py
# open http://localhost:5001/hedge-fund
```

### Run both at the same time
```bash
python app.py &           # runs in background on port 5000
python hedge_fund_web.py  # runs in foreground on port 5001
```

---

## API Key — Why It's Needed

Every time an agent runs, it calls the **Anthropic API** (Claude). The API key:
1. Authenticates your requests
2. Tracks usage for billing
3. Is required even when running locally — the AI processing happens on Anthropic's servers, not your machine

**To avoid typing it every time**, create a `.env` file in the project folder:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```
Both apps load this automatically on startup.

Get a key at: **console.anthropic.com** → API Keys
