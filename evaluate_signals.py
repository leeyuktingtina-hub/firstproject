"""
evaluate_signals.py
===================
Historical replay of the quant_scanner signal logic to measure win rate.

For every trading day since START, recompute the exact composite score the
scanner uses (momentum + RSI + MACD + Bollinger + trend), detect days where a
stock FLIPS into the BUY zone (score crosses above 0.62), then measure the
forward return 5 and 21 trading days later (≈7 / 30 calendar days).

Win definition (same as monitor.py outcome tracker):
    win     : +3% or better
    loss    : -5% or worse
    neutral : in between

CLI:  python evaluate_signals.py
Web:  run_evaluation() returns a JSON-ready dict (used by /api/winrate/run)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from quant_scanner import UNIVERSE

START     = "2020-06-01"   # includes warmup for 200d MA
EVAL_FROM = "2021-01-01"   # signals evaluated from here

BUY_TH, SELL_TH = 0.62, 0.38


def composite_series(close: pd.Series) -> pd.Series:
    """Vectorized version of score_ticker()'s composite score."""
    ret_1m = (close / close.shift(22)  - 1) * 100
    ret_3m = (close / close.shift(63)  - 1) * 100
    ret_6m = (close / close.shift(126) - 1) * 100

    # Momentum score
    mom = pd.Series(0.5, index=close.index)
    mom += np.select([ret_3m > 10, ret_3m > 0, ret_3m < -20], [0.15, 0.07, -0.15], -0.07)
    mom += np.select([ret_6m > 20, ret_6m > 0, ret_6m < -30], [0.15, 0.07, -0.15], -0.07)
    mom += np.select([ret_1m > 5,  ret_1m > 0, ret_1m < -10], [0.10, 0.04, -0.10], -0.04)
    mom = mom.clip(0.0, 1.0)

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).fillna(50.0)
    rsi_score = pd.Series(
        np.select([rsi < 30, rsi < 45, rsi < 55, rsi < 70], [1.0, 0.7, 0.5, 0.3], 0.0),
        index=close.index,
    )

    # MACD
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=9, adjust=False).mean()
    macd_score = pd.Series(np.where((macd - sig) > 0, 0.7, 0.3), index=close.index)

    # Bollinger %B
    mid   = close.rolling(20).mean()
    band  = close.rolling(20).std()
    bb    = ((close - (mid - 2 * band)) / (4 * band).replace(0, np.nan)).fillna(0.5)
    bb_score = (1.0 - bb).clip(0.0, 1.0)

    # Trend
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean().fillna(ma50)
    trend = 0.4 + (close > ma200) * 0.3 + (close > ma50) * 0.3

    return 0.25 * mom + 0.20 * rsi_score + 0.20 * macd_score + 0.15 * bb_score + 0.20 * trend


def evaluate_ticker(ticker: str, market: str, close: pd.Series) -> list[dict]:
    close = close.dropna()
    if len(close) < 260:
        return []

    score  = composite_series(close)
    signal = pd.Series(np.select([score > BUY_TH, score < SELL_TH], ["BUY", "SELL"], "HOLD"),
                       index=close.index)
    flips = (signal == "BUY") & (signal.shift(1).isin(["HOLD", "SELL"]))

    fwd5  = close.shift(-5)  / close - 1
    fwd21 = close.shift(-21) / close - 1

    events = []
    for dt in close.index[flips]:
        if dt < pd.Timestamp(EVAL_FROM, tz=dt.tz):
            continue
        f5, f21 = fwd5.get(dt), fwd21.get(dt)
        if pd.isna(f5):
            continue  # too recent to evaluate
        events.append({
            "ticker": ticker, "market": market, "date": dt.strftime("%Y-%m-%d"),
            "score": round(float(score[dt]) * 100, 1),
            "fwd5":  round(float(f5) * 100, 2),
            "fwd21": round(float(f21) * 100, 2) if not pd.isna(f21) else None,
        })
    return events


def _classify(r: float) -> str:
    return "win" if r >= 3 else ("loss" if r <= -5 else "neutral")


def _bucket_stats(rets: pd.Series) -> dict:
    n = len(rets)
    res = rets.apply(_classify)
    return {
        "signals":   int(n),
        "win_rate":  round((res == "win").mean() * 100, 1),
        "loss_rate": round((res == "loss").mean() * 100, 1),
        "neutral":   round((res == "neutral").mean() * 100, 1),
        "positive":  round((rets > 0).mean() * 100, 1),
        "avg_ret":   round(rets.mean(), 2),
        "med_ret":   round(rets.median(), 2),
    }


def run_evaluation() -> dict:
    """Full historical replay. Returns JSON-ready stats dict."""
    tickers = [(t, m) for m, ts in UNIVERSE.items() for t in ts]
    symbols = [t for t, _ in tickers]
    data = yf.download(symbols, start=START, auto_adjust=True, progress=False)["Close"]

    all_events = []
    for t, m in tickers:
        if t in data.columns:
            all_events += evaluate_ticker(t, m, data[t])

    if not all_events:
        return {"error": "No signals found (data download may have failed)"}

    df   = pd.DataFrame(all_events)
    df21 = df.dropna(subset=["fwd21"])

    by_market = {
        mkt: _bucket_stats(g["fwd21"])
        for mkt, g in df21.groupby("market")
    }
    by_year = {
        yr: _bucket_stats(g["fwd21"])
        for yr, g in df21.assign(year=df21["date"].str[:4]).groupby("year")
    }
    hi = df21[df21["score"] > 70]

    return {
        "eval_from":   EVAL_FROM,
        "universe":    len(symbols),
        "downloaded":  int(data.notna().any().sum()),
        "hold_7d":     _bucket_stats(df["fwd5"]),
        "hold_30d":    _bucket_stats(df21["fwd21"]),
        "high_score":  _bucket_stats(hi["fwd21"]) if len(hi) >= 10 else None,
        "by_market":   by_market,
        "by_year":     by_year,
        "win_def":     "win ≥ +3% · loss ≤ -5% · neutral in between",
    }


def main():
    r = run_evaluation()
    if "error" in r:
        print(r["error"])
        return
    print(f"\nBUY-signal replay {r['eval_from']} → today · {r['downloaded']}/{r['universe']} tickers")
    for label, key in [("7天持有", "hold_7d"), ("30天持有", "hold_30d")]:
        s = r[key]
        print(f"\n▶ {label}  信号数 {s['signals']}")
        print(f"   胜率(≥+3%) {s['win_rate']}% · 亏损(≤-5%) {s['loss_rate']}% · 正收益 {s['positive']}%")
        print(f"   平均 {s['avg_ret']:+.2f}% · 中位 {s['med_ret']:+.2f}%")
    print("\n分市场(30天):")
    for mkt, s in r["by_market"].items():
        print(f"   {mkt:3s} 信号{s['signals']:5d} · 胜率 {s['win_rate']}% · 平均 {s['avg_ret']:+.2f}%")
    print("\n分年份(30天):")
    for yr, s in r["by_year"].items():
        print(f"   {yr} 信号{s['signals']:5d} · 胜率 {s['win_rate']}% · 平均 {s['avg_ret']:+.2f}%")
    if r.get("high_score"):
        s = r["high_score"]
        print(f"\n高分信号(score>70, 30天): 信号{s['signals']} · 胜率 {s['win_rate']}% · 平均 {s['avg_ret']:+.2f}%")


if __name__ == "__main__":
    main()
