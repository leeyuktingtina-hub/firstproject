"""
evaluate_strategies.py
======================
Per-market strategy selection with proper train/test split (no look-ahead).

Method (standard quant practice):
  TRAIN 2021-01-01 → 2023-12-31 : compare candidate strategies per market,
                                   pick the best by avg 30d forward return.
  TEST  2024-01-01 → today      : report ONLY the chosen strategy's
                                   out-of-sample performance. Strategies are
                                   never tuned on this period.

Candidate strategies (all rules defined ex-ante, from classic literature):
  baseline   : composite score > 0.62 (current scanner logic)
  strict     : composite score > 0.70
  dip_buy    : RSI < 30 AND price > MA200   (mean-reversion in uptrend)
  oversold   : RSI < 30                     (pure mean reversion)
  trend_mom  : price > MA50 > MA200 AND 3m return > 10% AND MACD bullish
  regime     : baseline AND market index above its own MA200
               (index: US=SPY, HK=^HSI, CN=KWEB)

Hold period: 21 trading days (~30 calendar). win ≥ +3%, loss ≤ -5%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from quant_scanner import UNIVERSE
from evaluate_signals import composite_series

START       = "2020-06-01"
TRAIN_FROM  = "2021-01-01"
TEST_FROM   = "2024-01-01"

MARKET_INDEX = {"US": "SPY", "HK": "^HSI", "CN": "KWEB"}


def _indicators(close: pd.Series) -> dict:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).fillna(50.0)

    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=9, adjust=False).mean()

    return {
        "close":  close,
        "rsi":    rsi,
        "ma50":   close.rolling(50).mean(),
        "ma200":  close.rolling(200).mean(),
        "macd_b": (macd - sig) > 0,
        "ret_3m": (close / close.shift(63) - 1) * 100,
        "score":  composite_series(close),
    }


def _strategy_conditions(ind: dict, regime_ok: pd.Series) -> dict[str, pd.Series]:
    c = ind["close"]
    return {
        "baseline":  ind["score"] > 0.62,
        "strict":    ind["score"] > 0.70,
        "dip_buy":   (ind["rsi"] < 30) & (c > ind["ma200"]),
        "oversold":  ind["rsi"] < 30,
        "trend_mom": (c > ind["ma50"]) & (ind["ma50"] > ind["ma200"])
                     & (ind["ret_3m"] > 10) & ind["macd_b"],
        "regime":    (ind["score"] > 0.62) & regime_ok,
    }


def _stats(rets: pd.Series) -> dict | None:
    n = len(rets)
    if n == 0:
        return None
    return {
        "n":        int(n),
        "win":      round((rets >= 3).mean() * 100, 1),
        "loss":     round((rets <= -5).mean() * 100, 1),
        "positive": round((rets > 0).mean() * 100, 1),
        "avg":      round(rets.mean(), 2),
        "med":      round(rets.median(), 2),
    }


def main():
    tickers = [(t, m) for m, ts in UNIVERSE.items() for t in ts]
    symbols = [t for t, _ in tickers] + list(MARKET_INDEX.values())
    print(f"Downloading {len(symbols)} tickers …", flush=True)
    data = yf.download(symbols, start=START, auto_adjust=True, progress=False)["Close"]

    # Per-market regime series: index above its own 200d MA
    regime = {}
    for mkt, idx_tk in MARKET_INDEX.items():
        s = data[idx_tk].dropna()
        regime[mkt] = (s > s.rolling(200).mean()).reindex(data.index).ffill().fillna(False)

    # Collect signal events: rows of (market, strategy, date, fwd21)
    rows = []
    for t, m in tickers:
        if t not in data.columns:
            continue
        close = data[t].dropna()
        if len(close) < 260:
            continue
        ind   = _indicators(close)
        fwd21 = (close.shift(-21) / close - 1) * 100
        conds = _strategy_conditions(ind, regime[m].reindex(close.index).fillna(False))
        for strat, cond in conds.items():
            cond  = cond.fillna(False)
            entry = cond & ~cond.shift(1, fill_value=False)   # flip into condition
            for dt in close.index[entry]:
                if dt < pd.Timestamp(TRAIN_FROM, tz=dt.tz):
                    continue
                f = fwd21.get(dt)
                if pd.isna(f):
                    continue
                rows.append({"market": m, "strat": strat,
                             "date": dt.strftime("%Y-%m-%d"), "fwd21": float(f)})

    df = pd.DataFrame(rows)
    df["period"] = np.where(df["date"] < TEST_FROM, "train", "test")

    print(f"\n{'='*70}")
    print(f"  训练集 {TRAIN_FROM}→{TEST_FROM} 选策略 · 测试集 {TEST_FROM}→今天 验证")
    print(f"{'='*70}")

    chosen = {}
    for mkt in ["US", "HK", "CN"]:
        print(f"\n━━ {mkt} 市场 ━━ (训练集表现)")
        best_name, best_avg = None, -99
        for strat in ["baseline", "strict", "dip_buy", "oversold", "trend_mom", "regime"]:
            g = df[(df.market == mkt) & (df.strat == strat) & (df.period == "train")]
            s = _stats(g["fwd21"])
            if s is None:
                continue
            mark = ""
            if s["n"] >= 80 and s["avg"] > best_avg:
                best_name, best_avg, mark = strat, s["avg"], ""
            print(f"   {strat:10s} 信号{s['n']:5d} · 胜率{s['win']:5.1f}% · 亏损{s['loss']:5.1f}% · 平均{s['avg']:+6.2f}%")
        chosen[mkt] = best_name
        print(f"   → 训练集最优: {best_name}")

    print(f"\n{'='*70}")
    print(f"  ✅ 样本外验证（测试集 {TEST_FROM}→今天，选策略时未看过这段数据）")
    print(f"{'='*70}")
    for mkt, strat in chosen.items():
        if strat is None:
            print(f"\n━━ {mkt}: 无有效策略")
            continue
        g_test  = df[(df.market == mkt) & (df.strat == strat) & (df.period == "test")]
        g_base  = df[(df.market == mkt) & (df.strat == "baseline") & (df.period == "test")]
        s, b = _stats(g_test["fwd21"]), _stats(g_base["fwd21"])
        print(f"\n━━ {mkt} 选中策略: {strat}")
        if s:
            print(f"   测试集     信号{s['n']:5d} · 胜率{s['win']:5.1f}% · 亏损{s['loss']:5.1f}% · 正收益{s['positive']:5.1f}% · 平均{s['avg']:+6.2f}%")
        if b and strat != "baseline":
            print(f"   (对比基线  信号{b['n']:5d} · 胜率{b['win']:5.1f}% · 平均{b['avg']:+6.2f}%)")

    # Full test-period table for reference
    print(f"\n──全部策略测试集表现（供参考）──")
    for mkt in ["US", "HK", "CN"]:
        for strat in ["baseline", "strict", "dip_buy", "oversold", "trend_mom", "regime"]:
            g = df[(df.market == mkt) & (df.strat == strat) & (df.period == "test")]
            s = _stats(g["fwd21"])
            if s:
                print(f"   {mkt:3s} {strat:10s} 信号{s['n']:5d} · 胜率{s['win']:5.1f}% · 平均{s['avg']:+6.2f}%")


if __name__ == "__main__":
    main()
