"""
evaluate_longterm.py
====================
Long-horizon replay 1995 → today across full bull/bear cycles.

Universe: US tech stocks with history back to the 1990s (newer listings like
NVDA-1999 / AMZN-1997 simply enter when their data starts).

For each era we report, per strategy:
  - signal count, win rate (≥+3% in 21 trading days), avg forward return
  - the unconditional average 21-day return of the same stocks in that era
    ("random entry" baseline) — a strategy only has edge if it beats this.

Eras:
  1995-2000.03  dot-com bull        2000.03-2002.10  dot-com crash
  2002.10-2007.10 recovery bull     2007.10-2009.03  GFC bear
  2009.03-2020.01 QE long bull      2020          covid crash+rebound
  2021          stimulus top        2022          rate-hike bear
  2023-now      AI bull
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from evaluate_signals import composite_series

LONG_UNIVERSE = [
    "AAPL", "MSFT", "INTC", "AMD", "ORCL", "IBM", "CSCO", "TXN", "MU",
    "ADBE", "QCOM", "AMAT", "KLAC", "LRCX", "HPQ", "NVDA", "AMZN",
]

ERAS = [
    ("1995-2000.3 互联网泡沫牛市", "1995-01-01", "2000-03-10"),
    ("2000-2002 互联网崩盘熊市",   "2000-03-10", "2002-10-09"),
    ("2002-2007 复苏牛市",         "2002-10-09", "2007-10-09"),
    ("2007-2009 金融危机熊市",     "2007-10-09", "2009-03-09"),
    ("2009-2020 QE长牛",           "2009-03-09", "2020-01-01"),
    ("2020 疫情崩盘+反弹",         "2020-01-01", "2021-01-01"),
    ("2021 放水牛尾",              "2021-01-01", "2022-01-01"),
    ("2022 加息熊市",              "2022-01-01", "2023-01-01"),
    ("2023-今 AI牛市",             "2023-01-01", "2099-01-01"),
]

BUY_TH = 0.62


def _series_for(close: pd.Series) -> dict:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).fillna(50.0)
    ma200 = close.rolling(200).mean()
    score = composite_series(close)
    return {"rsi": rsi, "ma200": ma200, "score": score}


def main():
    print(f"Downloading {len(LONG_UNIVERSE)} long-history tickers since 1994 …", flush=True)
    data = yf.download(LONG_UNIVERSE, start="1994-01-01", auto_adjust=True, progress=False)["Close"]

    rows = []        # one row per signal event
    uncond = []      # unconditional daily fwd21 returns (random-entry baseline)

    for t in LONG_UNIVERSE:
        if t not in data.columns:
            continue
        close = data[t].dropna()
        if len(close) < 300:
            continue
        ind   = _series_for(close)
        fwd21 = ((close.shift(-21) / close - 1) * 100).rename("fwd21")

        strategies = {
            "momentum(现行美股策略)": ind["score"] > BUY_TH,
            "oversold(RSI<30)":      ind["rsi"] < 30,
            "dip_buy(RSI<30且>MA200)": (ind["rsi"] < 30) & (close > ind["ma200"]),
        }
        for name, cond in strategies.items():
            cond  = cond.fillna(False)
            entry = cond & ~cond.shift(1, fill_value=False)
            for dt in close.index[entry]:
                f = fwd21.get(dt)
                if pd.isna(f):
                    continue
                rows.append({"strat": name, "date": dt.strftime("%Y-%m-%d"), "fwd21": float(f)})

        # Random-entry baseline: every trading day counts once
        df_u = fwd21.dropna()
        uncond.append(pd.DataFrame({"date": df_u.index.strftime("%Y-%m-%d"), "fwd21": df_u.values}))

    df  = pd.DataFrame(rows)
    dfu = pd.concat(uncond, ignore_index=True)

    print(f"\n共 {len(df)} 次信号 · {len(LONG_UNIVERSE)} 只长历史美股 · 持有21交易日(约30天)")
    print("胜=≥+3% · 基线=同期同股票随机买入的平均收益（跑赢基线才有真实优势）")

    for era, d1, d2 in ERAS:
        in_era_u = dfu[(dfu.date >= d1) & (dfu.date < d2)]
        base_avg = in_era_u["fwd21"].mean() if len(in_era_u) else float("nan")
        print(f"\n━━ {era} ━━  基线(随机买入){base_avg:+.2f}%")
        for name in ["momentum(现行美股策略)", "oversold(RSI<30)", "dip_buy(RSI<30且>MA200)"]:
            g = df[(df.strat == name) & (df.date >= d1) & (df.date < d2)]
            if len(g) < 10:
                print(f"   {name:26s} 信号不足({len(g)})")
                continue
            win = (g.fwd21 >= 3).mean() * 100
            avg = g.fwd21.mean()
            edge = avg - base_avg
            mark = "✅" if edge > 0 else "❌"
            print(f"   {name:26s} 信号{len(g):5d} · 胜率{win:5.1f}% · 平均{avg:+6.2f}% · 超额{edge:+6.2f}% {mark}")

    # Overall summary
    print(f"\n━━ 全期 1995→今 汇总 ━━  基线{dfu.fwd21.mean():+.2f}%")
    for name in ["momentum(现行美股策略)", "oversold(RSI<30)", "dip_buy(RSI<30且>MA200)"]:
        g = df[df.strat == name]
        win = (g.fwd21 >= 3).mean() * 100
        print(f"   {name:26s} 信号{len(g):5d} · 胜率{win:5.1f}% · 平均{g.fwd21.mean():+6.2f}% · 超额{g.fwd21.mean()-dfu.fwd21.mean():+6.2f}%")


if __name__ == "__main__":
    main()
