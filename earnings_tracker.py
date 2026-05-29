"""
earnings_tracker.py
===================
Earnings calendar and time-zone arbitrage linkage map.
Tracks when major tech stocks report and which linked stocks to watch.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Linkage map ───────────────────────────────────────────────────────────────
# For each major reporter: competitors, suppliers, HK/CN linked stocks

LINKAGE_MAP = {
    "NVDA": {
        "name":    "NVIDIA",
        "theme":   "AI GPU / Data Center",
        "desc":    "全球AI算力龙头，财报超预期时整个AI产业链联动",
        "competitors": ["AMD", "INTC"],
        "suppliers":   ["AMAT", "LRCX", "MU", "TSM"],
        "hk_cn":       ["0700.HK", "9888.HK", "BIDU", "9999.HK"],
        "hk_cn_names": ["腾讯(AI算力用户)", "百度HK(AI算力)", "百度ADR", "网易HK"],
    },
    "AMD": {
        "name":    "AMD",
        "theme":   "GPU/CPU 半导体",
        "desc":    "NVDA直接竞争者，财报联动半导体板块",
        "competitors": ["NVDA", "INTC"],
        "suppliers":   ["TSM", "AMAT", "MU"],
        "hk_cn":       ["0700.HK", "9888.HK"],
        "hk_cn_names": ["腾讯", "百度HK"],
    },
    "TSLA": {
        "name":    "Tesla",
        "theme":   "电动车",
        "desc":    "EV需求信号，强烈联动中国新能源车板块",
        "competitors": ["NIO", "XPEV", "LI"],
        "suppliers":   ["PANW", "MU"],
        "hk_cn":       ["1211.HK", "9868.HK", "2015.HK", "0175.HK", "NIO", "XPEV", "LI"],
        "hk_cn_names": ["比亚迪HK", "小鹏HK", "理想HK", "吉利", "蔚来ADR", "小鹏ADR", "理想ADR"],
    },
    "AAPL": {
        "name":    "Apple",
        "theme":   "消费电子",
        "desc":    "消费需求风向标，联动全球供应链",
        "competitors": ["MSFT", "GOOGL"],
        "suppliers":   ["TSM", "QCOM", "AVGO", "AMAT"],
        "hk_cn":       ["2382.HK", "0992.HK", "1810.HK"],
        "hk_cn_names": ["舜宇光学(摄像头)", "联想(PC需求)", "小米"],
    },
    "MSFT": {
        "name":    "Microsoft",
        "theme":   "云计算 / 企业AI",
        "desc":    "Azure增长驱动企业科技，联动云服务板块",
        "competitors": ["AMZN", "GOOGL"],
        "suppliers":   ["NVDA", "AMD"],
        "hk_cn":       ["0700.HK", "9988.HK", "9999.HK"],
        "hk_cn_names": ["腾讯云", "阿里云HK", "网易云"],
    },
    "AMZN": {
        "name":    "Amazon",
        "theme":   "云计算 / 电商",
        "desc":    "AWS全球最大云，联动电商和云服务",
        "competitors": ["MSFT", "GOOGL"],
        "suppliers":   ["NVDA", "AMD"],
        "hk_cn":       ["9988.HK", "JD", "PDD"],
        "hk_cn_names": ["阿里巴巴HK", "京东ADR", "拼多多"],
    },
    "GOOGL": {
        "name":    "Alphabet",
        "theme":   "搜索 / 广告 / AI",
        "desc":    "广告市场健康度指标，联动数字广告和AI板块",
        "competitors": ["META", "MSFT"],
        "suppliers":   ["NVDA", "TSM"],
        "hk_cn":       ["BIDU", "TME", "BILI", "IQ"],
        "hk_cn_names": ["百度ADR(搜索)", "腾讯音乐(广告)", "哔哩哔哩", "爱奇艺"],
    },
    "META": {
        "name":    "Meta",
        "theme":   "社交媒体 / 数字广告",
        "desc":    "数字广告景气度，联动中国内容平台",
        "competitors": ["GOOGL", "SNAP"],
        "suppliers":   ["NVDA"],
        "hk_cn":       ["TME", "BILI", "IQ", "9626.HK"],
        "hk_cn_names": ["腾讯音乐", "哔哩哔哩ADR", "爱奇艺", "哔哩哔哩HK"],
    },
    "TSM": {
        "name":    "台积电",
        "theme":   "晶圆代工",
        "desc":    "全球最大芯片代工，财报是半导体行业风向标",
        "competitors": ["INTC"],
        "suppliers":   ["AMAT", "LRCX"],
        "hk_cn":       ["0700.HK", "1810.HK", "0992.HK", "AMD", "NVDA"],
        "hk_cn_names": ["腾讯(芯片用户)", "小米", "联想", "AMD", "英伟达"],
    },
    "AVGO": {
        "name":    "Broadcom",
        "theme":   "定制AI芯片 / 网络",
        "desc":    "超大规模算力定制芯片，联动AI基础设施",
        "competitors": ["NVDA", "QCOM"],
        "suppliers":   ["TSM", "AMAT"],
        "hk_cn":       ["0700.HK", "9888.HK", "BIDU"],
        "hk_cn_names": ["腾讯", "百度HK", "百度ADR"],
    },
    "QCOM": {
        "name":    "Qualcomm",
        "theme":   "手机芯片",
        "desc":    "手机SoC龙头，联动手机产业链",
        "competitors": ["NVDA", "AMD", "AVGO"],
        "suppliers":   ["TSM", "AMAT"],
        "hk_cn":       ["1810.HK", "0992.HK", "2382.HK"],
        "hk_cn_names": ["小米(手机)", "联想", "舜宇光学"],
    },
    "MU": {
        "name":    "Micron",
        "theme":   "内存/存储芯片",
        "desc":    "HBM内存供应商，AI服务器需求代理",
        "competitors": ["NVDA", "AMD"],
        "suppliers":   ["AMAT", "LRCX"],
        "hk_cn":       ["0700.HK", "9888.HK", "BIDU"],
        "hk_cn_names": ["腾讯", "百度HK", "百度ADR"],
    },
}

TRACKED_TICKERS = list(LINKAGE_MAP.keys())


# ── Fetch earnings calendar ───────────────────────────────────────────────────

def _fetch_calendar(ticker: str) -> dict | None:
    try:
        t    = yf.Ticker(ticker)
        cal  = t.calendar
        info = t.info

        upcoming_date = None
        if cal is not None:
            dates = cal.get("Earnings Date") or cal.get("Earnings Dates")
            if dates is not None:
                if hasattr(dates, '__iter__') and not isinstance(dates, str):
                    dates_list = list(dates)
                    if dates_list:
                        d = dates_list[0]
                        if hasattr(d, 'date'):
                            upcoming_date = d.date()
                        elif isinstance(d, str):
                            upcoming_date = datetime.strptime(d[:10], "%Y-%m-%d").date()

        # Recent earnings beat/miss from quarterly data
        beat_pct  = None
        beat_flag = None
        try:
            qe = t.quarterly_earnings
            if qe is not None and not qe.empty and len(qe) >= 1:
                row       = qe.iloc[0]
                actual    = float(row.get("Earnings", 0) or row.get("actual", 0))
                estimate  = float(row.get("Estimate", 0) or row.get("estimate", 0))
                if estimate and estimate != 0:
                    beat_pct  = round((actual - estimate) / abs(estimate) * 100, 1)
                    beat_flag = beat_pct > 10
        except Exception:
            pass

        lm = LINKAGE_MAP.get(ticker, {})
        return {
            "ticker":        ticker,
            "name":          lm.get("name", ticker),
            "theme":         lm.get("theme", ""),
            "desc":          lm.get("desc", ""),
            "upcoming_date": str(upcoming_date) if upcoming_date else None,
            "days_away":     (upcoming_date - datetime.today().date()).days if upcoming_date else None,
            "beat_pct":      beat_pct,
            "beat_flag":     beat_flag,
            "competitors":   lm.get("competitors", []),
            "suppliers":     lm.get("suppliers", []),
            "hk_cn":         lm.get("hk_cn", []),
            "hk_cn_names":   lm.get("hk_cn_names", []),
        }
    except Exception:
        return None


def get_earnings_data() -> dict:
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch_calendar, t): t for t in TRACKED_TICKERS}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    upcoming = [r for r in results if r["upcoming_date"] and r["days_away"] is not None and 0 <= r["days_away"] <= 45]
    upcoming.sort(key=lambda x: x["days_away"])

    recent_beats = [r for r in results if r["beat_pct"] is not None and r["beat_pct"] > 10]
    recent_beats.sort(key=lambda x: x["beat_pct"], reverse=True)

    return {
        "upcoming":      upcoming,
        "recent_beats":  recent_beats,
        "all":           results,
        "fetched_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
