"""
backtest.py
===========
Rule-based backtesting engine using technical indicators to generate
buy/sell signals and simulate portfolio performance over historical data.
Mirrors the logic of the hedge fund agents without requiring LLM calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta


# ── Technical indicators ─────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(prices: pd.Series, fast=12, slow=26, signal=9):
    ema_f = prices.ewm(span=fast, adjust=False).mean()
    ema_s = prices.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig   # histogram


def _bollinger_pct(prices: pd.Series, period=20, std=2.0) -> pd.Series:
    mid   = prices.rolling(period).mean()
    band  = prices.rolling(period).std()
    lower = mid - std * band
    upper = mid + std * band
    return (prices - lower) / (upper - lower).replace(0, np.nan)


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signals(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns:
    close, signal (1=BUY, -1=SELL, 0=HOLD), score, rsi, ma50, ma200
    """
    warmup = (pd.Timestamp(start_date) - timedelta(days=120)).strftime("%Y-%m-%d")
    hist = yf.Ticker(ticker).history(start=warmup, end=end_date, auto_adjust=True)
    if hist.empty or len(hist) < 60:
        return pd.DataFrame()

    close = hist["Close"]

    rsi      = _rsi(close)
    macd_h   = _macd(close)
    bb_pct   = _bollinger_pct(close)
    ma20     = close.rolling(20).mean()
    ma50     = close.rolling(50).mean()
    ma200    = close.rolling(200).mean()

    # Each indicator scored 0–1 (1 = most bullish)
    rsi_score  = pd.cut(rsi, bins=[-1,30,45,55,70,101],
                        labels=[1.0, 0.7, 0.5, 0.3, 0.0]).astype(float)
    macd_score = (macd_h > 0).astype(float) * 0.6 + 0.2
    bb_score   = (1.0 - bb_pct.clip(0, 1))

    trend = pd.Series(0.4, index=close.index)
    trend[close > ma200] += 0.2
    trend[close > ma50]  += 0.2
    trend[close > ma20]  += 0.2
    trend = trend.clip(0, 1)

    score = (0.30 * rsi_score +
             0.25 * macd_score +
             0.20 * bb_score +
             0.25 * trend)

    signal = pd.Series(0, index=close.index)
    signal[score > 0.62]  =  1
    signal[score < 0.38]  = -1

    return pd.DataFrame({
        "close":  close,
        "signal": signal,
        "score":  score,
        "rsi":    rsi,
        "ma50":   ma50,
        "ma200":  ma200,
    }).loc[start_date:end_date].dropna(subset=["score"])


# ── Portfolio simulation ──────────────────────────────────────────────────────

def run_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100_000,
    position_size: float = 0.20,
    stop_loss: float    = 0.08,
    take_profit: float  = 0.25,
    commission: float   = 0.001,
) -> dict:

    signals = {t: generate_signals(t, start_date, end_date)
               for t in tickers}
    signals = {t: s for t, s in signals.items() if not s.empty}

    if not signals:
        return {"error": "No price data returned for the selected tickers."}

    all_dates = sorted(set().union(*[set(s.index) for s in signals.values()]))

    cash      = initial_cash
    positions = {}   # ticker -> {shares, cost_basis}
    history   = []
    trades    = []

    for date in all_dates:
        # ── Value open positions & check exits ──
        port_value = cash
        for ticker, pos in list(positions.items()):
            if ticker in signals and date in signals[ticker].index:
                price   = float(signals[ticker].loc[date, "close"])
                pnl_pct = (price - pos["cost_basis"]) / pos["cost_basis"]
                port_value += pos["shares"] * price

                if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                    proceeds = pos["shares"] * price * (1 - commission)
                    cash    += proceeds
                    trades.append({
                        "date":    str(date.date()),
                        "ticker":  ticker,
                        "action":  "SELL",
                        "price":   round(price, 2),
                        "shares":  pos["shares"],
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "reason":  "stop_loss" if pnl_pct <= -stop_loss else "take_profit",
                    })
                    del positions[ticker]

        # ── Process new signals ──
        for ticker, sig in signals.items():
            if date not in sig.index:
                continue
            row    = sig.loc[date]
            price  = float(row["close"])
            signal = int(row["signal"])

            if signal == 1 and ticker not in positions:
                max_invest = port_value * position_size
                shares     = int(max_invest / price)
                cost       = shares * price * (1 + commission)
                if shares > 0 and cash >= cost:
                    cash -= cost
                    positions[ticker] = {"shares": shares, "cost_basis": price}
                    trades.append({
                        "date":    str(date.date()),
                        "ticker":  ticker,
                        "action":  "BUY",
                        "price":   round(price, 2),
                        "shares":  shares,
                        "pnl_pct": 0,
                        "reason":  "signal",
                    })

            elif signal == -1 and ticker in positions:
                pos     = positions[ticker]
                pnl_pct = (price - pos["cost_basis"]) / pos["cost_basis"]
                cash   += pos["shares"] * price * (1 - commission)
                trades.append({
                    "date":    str(date.date()),
                    "ticker":  ticker,
                    "action":  "SELL",
                    "price":   round(price, 2),
                    "shares":  pos["shares"],
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "reason":  "signal",
                })
                del positions[ticker]

        history.append({"date": str(date.date()), "value": round(port_value, 2)})

    # Close remaining positions at last price
    final_value = cash
    for ticker, pos in positions.items():
        if ticker in signals:
            final_value += pos["shares"] * float(signals[ticker]["close"].iloc[-1])

    # Benchmark: SPY buy-and-hold
    spy = yf.Ticker("SPY").history(start=start_date, end=end_date, auto_adjust=True)
    spy_return = float((spy["Close"].iloc[-1] / spy["Close"].iloc[0] - 1) * 100) if not spy.empty else None

    metrics = _calc_metrics(history, initial_cash, final_value, trades, spy_return)

    # Per-ticker summary
    ticker_summary = {}
    for ticker, sig in signals.items():
        sells = [t for t in trades if t["ticker"] == ticker and t["action"] == "SELL"]
        ticker_summary[ticker] = {
            "buy_signals":      int((sig["signal"] == 1).sum()),
            "sell_signals":     int((sig["signal"] == -1).sum()),
            "current_signal":   "BUY" if int(sig["signal"].iloc[-1]) == 1
                                else ("SELL" if int(sig["signal"].iloc[-1]) == -1 else "HOLD"),
            "current_score":    round(float(sig["score"].iloc[-1]) * 100, 1),
            "current_rsi":      round(float(sig["rsi"].iloc[-1]), 1),
            "trades":           len(sells),
        }

    return {
        "metrics":          metrics,
        "portfolio_history": history[::3],   # thin to every 3rd day
        "trades":           trades[-100:],
        "ticker_summary":   ticker_summary,
        "spy_return":       round(spy_return, 2) if spy_return is not None else None,
    }


def _calc_metrics(history, initial_cash, final_value, trades, spy_return) -> dict:
    total_return = (final_value / initial_cash - 1) * 100
    vals         = np.array([h["value"] for h in history])

    if len(vals) < 2:
        return {"total_return": round(total_return, 2), "final_value": round(final_value, 2)}

    daily_ret = np.diff(vals) / vals[:-1]
    n_years   = len(vals) / 252
    cagr      = ((final_value / initial_cash) ** (1 / max(n_years, 0.1)) - 1) * 100

    rf_daily  = 0.05 / 252
    excess    = daily_ret - rf_daily
    sharpe    = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    running_max  = np.maximum.accumulate(vals)
    max_drawdown = float(((vals - running_max) / running_max).min() * 100)

    sell_trades = [t for t in trades if t["action"] == "SELL"]
    if sell_trades:
        wins     = [t["pnl_pct"] for t in sell_trades if t["pnl_pct"] > 0]
        losses   = [t["pnl_pct"] for t in sell_trades if t["pnl_pct"] <= 0]
        win_rate = len(wins) / len(sell_trades) * 100
        avg_win  = float(np.mean(wins))  if wins   else 0
        avg_loss = float(np.mean(losses)) if losses else 0
    else:
        win_rate = avg_win = avg_loss = 0

    return {
        "total_return":  round(total_return, 2),
        "cagr":          round(cagr, 2),
        "sharpe_ratio":  round(sharpe, 2),
        "max_drawdown":  round(max_drawdown, 2),
        "win_rate":      round(win_rate, 1),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "total_trades":  len(sell_trades),
        "final_value":   round(final_value, 2),
        "alpha":         round(total_return - spy_return, 2) if spy_return is not None else None,
    }
