"""
quant_scanner.py
================
Multi-factor quant scanner inspired by AQR Capital's factor investing approach.
Scores stocks on Momentum + Technical signals across US Tech, HK, and China.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta


# ── Stock universe ────────────────────────────────────────────────────────────

UNIVERSE = {
    "US": [
        "NVDA", "MSFT", "GOOGL", "META", "AMD", "TSLA", "AAPL", "AMZN",
        "AVGO", "QCOM", "MU", "AMAT", "LRCX", "TSM", "ARM", "SMCI", "MRVL",
        "ORCL", "CRM", "SNOW", "PLTR", "UBER", "COIN", "NFLX", "CRWD", "PANW",
        "DDOG", "NET", "NOW", "WDAY", "ADBE", "SHOP", "PYPL", "SQ",
        "RBLX", "HOOD", "SOFI", "IONQ", "SOUN", "INTC",
    ],
    "HK": [
        "0700.HK",  # Tencent
        "9988.HK",  # Alibaba HK
        "3690.HK",  # Meituan
        "9999.HK",  # NetEase HK
        "1810.HK",  # Xiaomi
        "0992.HK",  # Lenovo
        "9626.HK",  # Bilibili HK
        "0941.HK",  # China Mobile
        "1211.HK",  # BYD
        "2020.HK",  # ANTA Sports
        "9868.HK",  # XPeng HK
        "2015.HK",  # Li Auto HK
        "0175.HK",  # Geely
        "1024.HK",  # Kuaishou
        "9888.HK",  # Baidu HK
        "9961.HK",  # Trip.com HK
        "2318.HK",  # Ping An
        "2382.HK",  # Sunny Optical
        "6618.HK",  # JD Health
        "6690.HK",  # Haier Smart Home
    ],
    "CN": [
        "BABA",   # Alibaba ADR
        "JD",     # JD.com
        "PDD",    # Pinduoduo / Temu
        "BIDU",   # Baidu ADR
        "NIO",    # NIO
        "XPEV",   # XPeng ADR
        "LI",     # Li Auto ADR
        "NTES",   # NetEase ADR
        "BILI",   # Bilibili ADR
        "TME",    # Tencent Music
        "IQ",     # iQIYI
        "FUTU",   # Futu Holdings
        "TIGR",   # Tiger Brokers
        "YUMC",   # Yum China
        "ZH",     # Zhihu
    ],
}

# Static display names to avoid slow .info API calls
NAMES = {
    "NVDA": "NVIDIA", "MSFT": "Microsoft", "GOOGL": "Alphabet", "META": "Meta",
    "AMD": "AMD", "TSLA": "Tesla", "AAPL": "Apple", "AMZN": "Amazon",
    "AVGO": "Broadcom", "QCOM": "Qualcomm", "MU": "Micron", "AMAT": "Applied Materials",
    "LRCX": "Lam Research", "TSM": "TSMC", "ARM": "Arm Holdings", "SMCI": "Super Micro",
    "MRVL": "Marvell Tech", "ORCL": "Oracle", "CRM": "Salesforce", "SNOW": "Snowflake",
    "PLTR": "Palantir", "UBER": "Uber", "COIN": "Coinbase", "NFLX": "Netflix",
    "CRWD": "CrowdStrike", "PANW": "Palo Alto", "DDOG": "Datadog", "NET": "Cloudflare",
    "NOW": "ServiceNow", "WDAY": "Workday", "ADBE": "Adobe", "SHOP": "Shopify",
    "PYPL": "PayPal", "SQ": "Block", "RBLX": "Roblox", "HOOD": "Robinhood",
    "SOFI": "SoFi", "IONQ": "IonQ", "SOUN": "SoundHound", "INTC": "Intel",
    "0700.HK": "Tencent", "9988.HK": "Alibaba HK", "3690.HK": "Meituan",
    "9999.HK": "NetEase HK", "1810.HK": "Xiaomi", "0992.HK": "Lenovo",
    "9626.HK": "Bilibili HK", "0941.HK": "China Mobile", "1211.HK": "BYD",
    "2020.HK": "ANTA Sports", "9868.HK": "XPeng HK", "2015.HK": "Li Auto HK",
    "0175.HK": "Geely", "1024.HK": "Kuaishou", "9888.HK": "Baidu HK",
    "9961.HK": "Trip.com HK", "2318.HK": "Ping An", "2382.HK": "Sunny Optical",
    "6618.HK": "JD Health", "6690.HK": "Haier Smart",
    "BABA": "Alibaba ADR", "JD": "JD.com", "PDD": "Pinduoduo",
    "BIDU": "Baidu ADR", "NIO": "NIO", "XPEV": "XPeng ADR", "LI": "Li Auto ADR",
    "NTES": "NetEase ADR", "BILI": "Bilibili ADR", "TME": "Tencent Music",
    "IQ": "iQIYI", "FUTU": "Futu Holdings", "TIGR": "Tiger Brokers",
    "YUMC": "Yum China", "ZH": "Zhihu",
}

ALL_TICKERS = [(t, m) for m, tickers in UNIVERSE.items() for t in tickers]


# ── Technical indicators ──────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0


def _macd_bull(prices: pd.Series, fast=12, slow=26, signal=9) -> bool:
    ema_f = prices.ewm(span=fast, adjust=False).mean()
    ema_s = prices.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return float((macd - sig).iloc[-1]) > 0


def _bb_pct(prices: pd.Series, period=20, std=2.0) -> float:
    mid   = prices.rolling(period).mean()
    band  = prices.rolling(period).std()
    lower = mid - std * band
    upper = mid + std * band
    pct   = (prices - lower) / (upper - lower).replace(0, np.nan)
    v = float(pct.iloc[-1])
    return v if not np.isnan(v) else 0.5


# ── Score a single ticker ─────────────────────────────────────────────────────

def score_ticker(ticker: str, market: str) -> dict | None:
    try:
        end   = datetime.today()
        start = end - timedelta(days=300)
        hist  = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if hist.empty or len(hist) < 60:
            return None

        close = hist["Close"]
        price = float(close.iloc[-1])

        # Momentum
        ret_1m = float((close.iloc[-1] / close.iloc[-22]  - 1) * 100) if len(close) >= 22  else 0.0
        ret_3m = float((close.iloc[-1] / close.iloc[-63]  - 1) * 100) if len(close) >= 63  else 0.0
        ret_6m = float((close.iloc[-1] / close.iloc[-126] - 1) * 100) if len(close) >= 126 else 0.0

        # Technical
        rsi    = _rsi(close)
        macd_b = _macd_bull(close)
        bb     = _bb_pct(close)
        ma50   = float(close.rolling(50).mean().iloc[-1])
        ma200  = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50

        # ── Factor scores (0–1, higher = more bullish) ──
        # Momentum score
        mom = 0.5
        mom += 0.15 if ret_3m > 10  else (0.07 if ret_3m > 0   else (-0.15 if ret_3m < -20 else -0.07))
        mom += 0.15 if ret_6m > 20  else (0.07 if ret_6m > 0   else (-0.15 if ret_6m < -30 else -0.07))
        mom += 0.10 if ret_1m > 5   else (0.04 if ret_1m > 0   else (-0.10 if ret_1m < -10 else -0.04))
        mom  = max(0.0, min(1.0, mom))

        # RSI score (oversold = buy opportunity)
        rsi_score = 1.0 if rsi < 30 else (0.7 if rsi < 45 else (0.5 if rsi < 55 else (0.3 if rsi < 70 else 0.0)))

        # MACD
        macd_score = 0.7 if macd_b else 0.3

        # Bollinger: low in band = oversold = bullish
        bb_score = max(0.0, min(1.0, 1.0 - bb))

        # Trend
        trend = 0.4 + (0.3 if price > ma200 else 0) + (0.3 if price > ma50 else 0)

        composite = (
            0.25 * mom +
            0.20 * rsi_score +
            0.20 * macd_score +
            0.15 * bb_score +
            0.20 * trend
        )

        signal = "BUY" if composite > 0.62 else ("SELL" if composite < 0.38 else "HOLD")

        return {
            "ticker":   ticker,
            "market":   market,
            "name":     NAMES.get(ticker, ticker),
            "price":    round(price, 2),
            "ret_1m":   round(ret_1m, 1),
            "ret_3m":   round(ret_3m, 1),
            "ret_6m":   round(ret_6m, 1),
            "rsi":      round(rsi, 1),
            "macd_b":   macd_b,
            "score":    round(composite * 100, 1),
            "signal":   signal,
        }
    except Exception:
        return None


# ── Run full scan ─────────────────────────────────────────────────────────────

def run_scan(markets: list[str] | None = None) -> dict:
    tickers = [(t, m) for t, m in ALL_TICKERS if markets is None or m in markets]

    results = []
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(score_ticker, t, m): (t, m) for t, m in tickers}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: x["score"], reverse=True)

    buys  = [r for r in results if r["signal"] == "BUY"]
    holds = [r for r in results if r["signal"] == "HOLD"]
    sells = [r for r in results if r["signal"] == "SELL"]

    return {
        "scanned":    len(results),
        "total":      len(tickers),
        "top_picks":  results[:20],
        "all":        results,
        "buys":       len(buys),
        "holds":      len(holds),
        "sells":      len(sells),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
