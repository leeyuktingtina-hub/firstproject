"""
evaluate_portfolio.py
=====================
Portfolio-level backtest 1995 → today: does the strategy beat buy-and-hold SPY?

Realistic account simulation (per-signal averages overstate what an account
can capture — signals cluster and capital is finite):
  - max 10 concurrent positions, each 10% of current equity
  - signal on day D → enter at day D+1 close (no look-ahead)
  - hold 21 trading days, exit at close; no slippage modelled (noted in output)
  - idle cash earns 0

Variants tested:
  momentum  : composite score > 0.62 flips (current US production strategy)
  dip_buy   : RSI<30 AND price>MA200 flips (best per-signal edge since 1995)
  combined  : either signal

Benchmark: SPY buy-and-hold (dividends included via auto_adjust).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from evaluate_signals import composite_series
from evaluate_longterm import LONG_UNIVERSE, ERAS

HOLD_DAYS  = 21
MAX_POS    = 10
POS_FRAC   = 0.10


def _entry_signals(close: pd.Series) -> dict[str, pd.Series]:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).fillna(50.0)
    ma200 = close.rolling(200).mean()
    score = composite_series(close)

    momentum = (score > 0.62).fillna(False)
    dip      = ((rsi < 30) & (close > ma200)).fillna(False)
    flips    = lambda c: c & ~c.shift(1, fill_value=False)
    return {"momentum": flips(momentum), "dip_buy": flips(dip),
            "combined": flips(momentum) | flips(dip)}


def simulate(data: pd.DataFrame, signals: dict[str, pd.DataFrame], variant: str) -> pd.Series:
    """Daily equity curve for one variant."""
    sig    = signals[variant]              # DataFrame [date x ticker] bool
    dates  = data.index
    cash   = 1.0
    pos: list[dict] = []                   # {ticker, shares, exit_i}
    equity = np.full(len(dates), np.nan)

    for i, dt in enumerate(dates):
        px = data.loc[dt]

        # exits
        still = []
        for p in pos:
            if i >= p["exit_i"]:
                price = px.get(p["ticker"])
                if pd.isna(price):          # no price today → exit next day
                    p["exit_i"] = i + 1
                    still.append(p)
                else:
                    cash += p["shares"] * price
            else:
                still.append(p)
        pos = still

        # mark to market
        eq = cash
        for p in pos:
            price = px.get(p["ticker"])
            if not pd.isna(price):
                p["last_px"] = price
            eq += p["shares"] * p.get("last_px", 0)
        equity[i] = eq

        # entries: signals from PREVIOUS day, executed at today's close
        if i == 0:
            continue
        prev = dates[i - 1]
        held = {p["ticker"] for p in pos}
        for tk in sig.columns:
            if len(pos) >= MAX_POS or cash <= 1e-9:
                break
            if tk in held or not sig.at[prev, tk]:
                continue
            price = px.get(tk)
            if pd.isna(price):
                continue
            alloc = min(cash, eq * POS_FRAC)
            pos.append({"ticker": tk, "shares": alloc / price,
                        "exit_i": i + HOLD_DAYS, "last_px": price})
            cash -= alloc

    return pd.Series(equity, index=dates).ffill()


def _maxdd(eq: pd.Series) -> float:
    return float(((eq / eq.cummax()) - 1).min() * 100)


def _cagr(eq: pd.Series) -> float:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return float(((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100)


def main():
    symbols = LONG_UNIVERSE + ["SPY"]
    print(f"Downloading {len(symbols)} tickers since 1994 …", flush=True)
    data = yf.download(symbols, start="1994-01-01", auto_adjust=True, progress=False)["Close"]

    sig_frames: dict[str, pd.DataFrame] = {v: pd.DataFrame(index=data.index) for v in ["momentum", "dip_buy", "combined"]}
    for t in LONG_UNIVERSE:
        if t not in data.columns:
            continue
        close = data[t].dropna()
        if len(close) < 300:
            continue
        for v, s in _entry_signals(close).items():
            sig_frames[v][t] = s.reindex(data.index, fill_value=False)
    for v in sig_frames:
        sig_frames[v] = sig_frames[v].fillna(False)

    spy = data["SPY"].dropna()
    spy = spy / spy.iloc[0]

    curves = {"SPY买入持有": spy}
    for v in ["momentum", "dip_buy", "combined"]:
        print(f"simulating {v} …", flush=True)
        curves[v] = simulate(data.drop(columns=["SPY"]), sig_frames, v)

    print(f"\n━━ 组合级回测 1995→今 · 最多{MAX_POS}仓 · 每仓{int(POS_FRAC*100)}% · 持有{HOLD_DAYS}日 · 未计滑点 ━━")
    print(f"{'策略':14s} {'终值(1元变多少)':>14s} {'年化收益':>8s} {'最大回撤':>8s}")
    for name, eq in curves.items():
        eq = eq.dropna()
        print(f"{name:14s} {eq.iloc[-1]:>12.1f}x {_cagr(eq):>7.1f}% {_maxdd(eq):>7.1f}%")

    print("\n━━ 分时代收益（策略 vs SPY，同期区间收益%）━━")
    for era, d1, d2 in ERAS:
        line = f"{era:20s}"
        for name in ["SPY买入持有", "momentum", "dip_buy", "combined"]:
            eq = curves[name].dropna()
            seg = eq[(eq.index >= d1) & (eq.index < d2)]
            if len(seg) < 10:
                line += f" {'—':>9s}"
            else:
                r = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
                line += f" {r:>+8.0f}%"
        print(line)
    print(f"{'(列顺序: SPY / momentum / dip_buy / combined)'}")


if __name__ == "__main__":
    main()
