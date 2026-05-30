"""
earnings_tracker.py
===================
Earnings calendar and time-zone arbitrage linkage map.
Covers top 50+ US tech/AI stocks by market cap.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Expanded stock universe (top 50+ tech/AI by market cap) ──────────────────

TRACKED_STOCKS = {
    # Mega-cap tech
    "AAPL":  {"name": "Apple",          "theme": "消费电子"},
    "MSFT":  {"name": "Microsoft",      "theme": "云计算/企业AI"},
    "NVDA":  {"name": "NVIDIA",         "theme": "AI GPU/数据中心"},
    "GOOGL": {"name": "Alphabet",       "theme": "搜索/广告/AI"},
    "META":  {"name": "Meta",           "theme": "社交媒体/广告"},
    "AMZN":  {"name": "Amazon",         "theme": "云计算/电商"},
    "TSLA":  {"name": "Tesla",          "theme": "电动车/自动驾驶"},
    "AVGO":  {"name": "Broadcom",       "theme": "定制AI芯片/网络"},
    "TSM":   {"name": "台积电",          "theme": "晶圆代工"},
    "ORCL":  {"name": "Oracle",         "theme": "云数据库/AI"},
    # Semiconductors
    "AMD":   {"name": "AMD",            "theme": "GPU/CPU半导体"},
    "QCOM":  {"name": "Qualcomm",       "theme": "手机芯片"},
    "INTC":  {"name": "Intel",          "theme": "CPU/晶圆"},
    "AMAT":  {"name": "Applied Materials", "theme": "半导体设备"},
    "LRCX":  {"name": "Lam Research",   "theme": "半导体设备"},
    "KLAC":  {"name": "KLA Corp",       "theme": "半导体检测"},
    "ASML":  {"name": "ASML",           "theme": "EUV光刻机"},
    "MU":    {"name": "Micron",         "theme": "内存/HBM"},
    "ARM":   {"name": "Arm Holdings",   "theme": "芯片架构"},
    "MRVL":  {"name": "Marvell Tech",   "theme": "AI网络芯片"},
    "SMCI":  {"name": "Super Micro",    "theme": "AI服务器"},
    "ADI":   {"name": "Analog Devices", "theme": "模拟芯片"},
    "TXN":   {"name": "Texas Instruments", "theme": "模拟芯片"},
    "SNPS":  {"name": "Synopsys",       "theme": "芯片EDA软件"},
    "CDNS":  {"name": "Cadence",        "theme": "芯片EDA软件"},
    # Cloud / SaaS
    "CRM":   {"name": "Salesforce",     "theme": "AI CRM"},
    "NOW":   {"name": "ServiceNow",     "theme": "企业AI平台"},
    "ADBE":  {"name": "Adobe",          "theme": "AI创意软件"},
    "WDAY":  {"name": "Workday",        "theme": "HR/财务SaaS"},
    "SNOW":  {"name": "Snowflake",      "theme": "云数据仓库"},
    "DDOG":  {"name": "Datadog",        "theme": "云监控/AI"},
    "NET":   {"name": "Cloudflare",     "theme": "网络安全/AI"},
    "CRWD":  {"name": "CrowdStrike",    "theme": "网络安全AI"},
    "PANW":  {"name": "Palo Alto",      "theme": "网络安全"},
    "ZS":    {"name": "Zscaler",        "theme": "零信任安全"},
    # AI / Data
    "PLTR":  {"name": "Palantir",       "theme": "AI数据分析"},
    "AI":    {"name": "C3.ai",          "theme": "企业AI"},
    "SOUN":  {"name": "SoundHound",     "theme": "语音AI"},
    "IONQ":  {"name": "IonQ",           "theme": "量子计算"},
    # Internet / Consumer
    "NFLX":  {"name": "Netflix",        "theme": "流媒体/AI内容"},
    "UBER":  {"name": "Uber",           "theme": "出行/AI调度"},
    "SHOP":  {"name": "Shopify",        "theme": "电商SaaS"},
    "COIN":  {"name": "Coinbase",       "theme": "加密货币"},
    "HOOD":  {"name": "Robinhood",      "theme": "零售投资"},
    "SQ":    {"name": "Block",          "theme": "金融科技"},
    "PYPL":  {"name": "PayPal",         "theme": "支付/金融科技"},
    "RBLX":  {"name": "Roblox",         "theme": "元宇宙/游戏"},
    "SPOT":  {"name": "Spotify",        "theme": "音乐流媒体"},
    "ABNB":  {"name": "Airbnb",         "theme": "旅游/平台"},
    "LYFT":  {"name": "Lyft",           "theme": "出行"},
}

# ── Linkage map ───────────────────────────────────────────────────────────────

LINKAGE_MAP = {
    "NVDA": {
        "desc":      "全球AI算力龙头，财报超预期时整个AI产业链联动",
        "competitors": ["AMD", "INTC"],
        "suppliers":   ["AMAT", "LRCX", "MU", "TSM", "KLAC"],
        "hk_cn":       ["0700.HK", "9888.HK", "BIDU", "9999.HK"],
        "hk_cn_names": ["腾讯(AI算力用户)", "百度HK", "百度ADR", "网易HK"],
    },
    "AVGO": {
        "desc":      "超大规模定制AI芯片龙头，财报反映超大规模算力需求",
        "competitors": ["NVDA", "MRVL", "QCOM"],
        "suppliers":   ["TSM", "AMAT", "LRCX"],
        "hk_cn":       ["0700.HK", "9888.HK", "BIDU", "1810.HK"],
        "hk_cn_names": ["腾讯", "百度HK", "百度ADR", "小米"],
    },
    "AMD": {
        "desc":      "NVDA直接竞争者，财报联动半导体板块",
        "competitors": ["NVDA", "INTC"],
        "suppliers":   ["TSM", "AMAT", "MU"],
        "hk_cn":       ["0700.HK", "9888.HK"],
        "hk_cn_names": ["腾讯", "百度HK"],
    },
    "TSLA": {
        "desc":      "EV需求信号，强烈联动中国新能源车板块",
        "competitors": ["NIO", "XPEV", "LI"],
        "suppliers":   ["PANW", "MU"],
        "hk_cn":       ["1211.HK", "9868.HK", "2015.HK", "0175.HK", "NIO", "XPEV", "LI"],
        "hk_cn_names": ["比亚迪HK", "小鹏HK", "理想HK", "吉利", "蔚来ADR", "小鹏ADR", "理想ADR"],
    },
    "AAPL": {
        "desc":      "消费需求风向标，联动全球供应链",
        "competitors": ["MSFT", "GOOGL"],
        "suppliers":   ["TSM", "QCOM", "AVGO", "AMAT"],
        "hk_cn":       ["2382.HK", "0992.HK", "1810.HK"],
        "hk_cn_names": ["舜宇光学(摄像头)", "联想(PC需求)", "小米"],
    },
    "MSFT": {
        "desc":      "Azure增长驱动企业科技，联动云服务板块",
        "competitors": ["AMZN", "GOOGL"],
        "suppliers":   ["NVDA", "AMD"],
        "hk_cn":       ["0700.HK", "9988.HK", "9999.HK"],
        "hk_cn_names": ["腾讯云", "阿里云HK", "网易云"],
    },
    "AMZN": {
        "desc":      "AWS全球最大云，联动电商和云服务",
        "competitors": ["MSFT", "GOOGL"],
        "suppliers":   ["NVDA", "AMD"],
        "hk_cn":       ["9988.HK", "JD", "PDD"],
        "hk_cn_names": ["阿里巴巴HK", "京东ADR", "拼多多"],
    },
    "GOOGL": {
        "desc":      "广告市场健康度指标，联动数字广告和AI板块",
        "competitors": ["META", "MSFT"],
        "suppliers":   ["NVDA", "TSM"],
        "hk_cn":       ["BIDU", "TME", "BILI", "IQ"],
        "hk_cn_names": ["百度ADR(搜索)", "腾讯音乐", "哔哩哔哩", "爱奇艺"],
    },
    "META": {
        "desc":      "数字广告景气度，联动中国内容平台",
        "competitors": ["GOOGL", "SNAP"],
        "suppliers":   ["NVDA"],
        "hk_cn":       ["TME", "BILI", "IQ", "9626.HK"],
        "hk_cn_names": ["腾讯音乐", "哔哩哔哩ADR", "爱奇艺", "哔哩哔哩HK"],
    },
    "TSM": {
        "desc":      "全球最大芯片代工，财报是半导体行业风向标",
        "competitors": ["INTC"],
        "suppliers":   ["AMAT", "LRCX", "KLAC", "ASML"],
        "hk_cn":       ["0700.HK", "1810.HK", "0992.HK"],
        "hk_cn_names": ["腾讯(芯片用户)", "小米", "联想"],
    },
    "QCOM": {
        "desc":      "手机SoC龙头，联动手机产业链",
        "competitors": ["NVDA", "AMD", "AVGO"],
        "suppliers":   ["TSM", "AMAT"],
        "hk_cn":       ["1810.HK", "0992.HK", "2382.HK"],
        "hk_cn_names": ["小米(手机)", "联想", "舜宇光学"],
    },
    "MU": {
        "desc":      "HBM内存供应商，AI服务器需求代理",
        "competitors": ["NVDA", "AMD"],
        "suppliers":   ["AMAT", "LRCX"],
        "hk_cn":       ["0700.HK", "9888.HK", "BIDU"],
        "hk_cn_names": ["腾讯", "百度HK", "百度ADR"],
    },
    "NFLX": {
        "desc":      "流媒体订阅增长，联动内容/广告平台",
        "competitors": ["DIS", "SPOT"],
        "suppliers":   ["NVDA", "AMZN"],
        "hk_cn":       ["BILI", "IQ", "TME", "9626.HK"],
        "hk_cn_names": ["哔哩哔哩", "爱奇艺", "腾讯音乐", "哔哩哔哩HK"],
    },
    "CRM": {
        "desc":      "企业AI软件需求，联动SaaS板块",
        "competitors": ["NOW", "ADBE", "WDAY"],
        "suppliers":   ["NVDA", "AMZN"],
        "hk_cn":       ["0700.HK", "9988.HK"],
        "hk_cn_names": ["腾讯企业服务", "阿里企业云"],
    },
    "ORCL": {
        "desc":      "云AI数据库，联动企业云和数据中心",
        "competitors": ["MSFT", "AMZN", "SNOW"],
        "suppliers":   ["NVDA", "AMD"],
        "hk_cn":       ["9988.HK", "0700.HK"],
        "hk_cn_names": ["阿里云HK", "腾讯云"],
    },
}


# ── Fetch earnings date (multi-method fallback) ───────────────────────────────

def _get_upcoming_earnings_date(ticker_obj) -> str | None:
    """Try multiple yfinance API methods to get next earnings date."""
    today = datetime.today().date()

    # Method 1: earnings_dates DataFrame (yfinance 1.x)
    try:
        ed = ticker_obj.get_earnings_dates(limit=12)
        if ed is not None and not ed.empty:
            for idx in ed.index:
                d = idx.date() if hasattr(idx, 'date') else idx
                if d >= today:
                    return str(d)
    except Exception:
        pass

    # Method 2: calendar dict
    try:
        cal = ticker_obj.calendar
        if cal and isinstance(cal, dict):
            dates = cal.get("Earnings Date") or cal.get("Earnings Dates") or []
            if dates:
                dl = list(dates) if hasattr(dates, '__iter__') else [dates]
                for d in dl:
                    date_obj = d.date() if hasattr(d, 'date') else None
                    if date_obj and date_obj >= today:
                        return str(date_obj)
    except Exception:
        pass

    # Method 3: info earningsTimestamp
    try:
        info = ticker_obj.get_info()
        ts   = info.get("earningsTimestamp") or info.get("earningsDate")
        if ts and isinstance(ts, (int, float)):
            d = datetime.utcfromtimestamp(ts).date()
            if d >= today:
                return str(d)
    except Exception:
        pass

    return None


def _fetch_one(ticker: str) -> dict | None:
    try:
        t    = yf.Ticker(ticker)
        meta = TRACKED_STOCKS.get(ticker, {})
        lm   = LINKAGE_MAP.get(ticker, {})
        today = datetime.today().date()

        upcoming_date = _get_upcoming_earnings_date(t)
        days_away = None
        if upcoming_date:
            days_away = (datetime.strptime(upcoming_date, "%Y-%m-%d").date() - today).days

        # Recent earnings beat
        beat_pct  = None
        beat_flag = None
        try:
            qe = t.quarterly_earnings
            if qe is not None and not qe.empty:
                row      = qe.iloc[0]
                actual   = float(row.get("Earnings", 0) or row.get("actual", 0) or 0)
                estimate = float(row.get("Estimate", 0) or row.get("estimate", 0) or 0)
                if estimate and estimate != 0:
                    beat_pct  = round((actual - estimate) / abs(estimate) * 100, 1)
                    beat_flag = beat_pct > 10
        except Exception:
            pass

        return {
            "ticker":        ticker,
            "name":          meta.get("name", ticker),
            "theme":         meta.get("theme", ""),
            "desc":          lm.get("desc", ""),
            "upcoming_date": upcoming_date,
            "days_away":     days_away,
            "beat_pct":      beat_pct,
            "beat_flag":     beat_flag,
            "competitors":   lm.get("competitors", []),
            "suppliers":     lm.get("suppliers", []),
            "hk_cn":         lm.get("hk_cn", []),
            "hk_cn_names":   lm.get("hk_cn_names", []),
        }
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_earnings_data() -> dict:
    tickers = list(TRACKED_STOCKS.keys())
    results = []

    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    upcoming = sorted(
        [r for r in results if r["days_away"] is not None and 0 <= r["days_away"] <= 60],
        key=lambda x: x["days_away"]
    )
    recent_beats = sorted(
        [r for r in results if r["beat_pct"] is not None and r["beat_pct"] > 10],
        key=lambda x: x["beat_pct"], reverse=True
    )

    return {
        "upcoming":     upcoming,
        "recent_beats": recent_beats,
        "all":          sorted(results, key=lambda x: x["ticker"]),
        "fetched_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_tracked": len(tickers),
    }
