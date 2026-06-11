"""
monitor.py
==========
24/7 background market monitor — runs scans automatically like a quant fund,
detects signal changes, stores a signal feed, and pushes alerts to Telegram.

Signals detected:
  1. Signal flip   — a stock crosses into BUY or SELL zone (score threshold)
  2. Oversold      — RSI drops below 30 (potential entry)
  3. Overbought    — RSI rises above 75 (trim warning)
  4. Big move      — 1-day price move beyond ±6%
  5. Earnings soon — tracked stock reports within 3 days (once per day)

Telegram setup (optional, for phone push):
  1. Create a bot: chat with @BotFather on Telegram → /newbot → get token
  2. Get your chat id: chat with @userinfobot → it replies your id
  3. Set env vars in Railway: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import os
import json
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta

import requests

from quant_scanner import run_scan

# ── Config ────────────────────────────────────────────────────────────────────

SCAN_INTERVAL_MIN  = int(os.environ.get("MONITOR_INTERVAL_MIN", "30"))   # minutes between scans
SIGNALS_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals_history.json")
STATE_FILE         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_state.json")
MAX_SIGNALS_KEPT   = 300

# Hong Kong timezone for display
HKT = timezone(timedelta(hours=8))


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def get_signal_feed() -> dict:
    """Public API for the web page."""
    state    = _load_json(STATE_FILE, {})
    signals  = _load_json(SIGNALS_FILE, [])
    channels = push_channels_status()

    # Compute simple win-rate from resolved outcomes
    outcomes = state.get("outcomes", [])
    resolved = [o for o in outcomes if o.get("resolved")]
    wins     = sum(1 for o in resolved if o.get("result") == "win")
    win_rate = round(wins / len(resolved) * 100) if resolved else None

    return {
        "signals":        signals[:100],
        "last_scan":      state.get("last_scan"),
        "next_scan":      state.get("next_scan"),
        "scan_count":     state.get("scan_count", 0),
        "interval_min":   SCAN_INTERVAL_MIN,
        "channels":       channels,
        "push_on":        any(channels.values()),
        "telegram_on":    channels["telegram"],
        "monitor_running": _monitor_started,
        "win_rate":        win_rate,
        "signals_tracked": len(resolved),
    }


# ── Push channels ─────────────────────────────────────────────────────────────
# Configure ANY of these (set env vars in Railway Variables):
#
# 1. Email (Gmail):
#      SMTP_EMAIL    = your gmail address (sender)
#      SMTP_PASSWORD = Gmail "App Password" (Google Account → Security →
#                      2-Step Verification → App passwords)
#      ALERT_EMAIL   = where to send alerts (can be same as SMTP_EMAIL)
#
# 2. Bark (iPhone app, simplest):
#      BARK_KEY      = the key shown in the Bark app after install
#
# 3. Telegram:
#      TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID

def send_telegram(text: str) -> bool:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def send_email(subject: str, body: str) -> bool:
    import smtplib
    from email.mime.text import MIMEText
    from email.header import Header

    sender   = os.environ.get("SMTP_EMAIL", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    to_addr  = os.environ.get("ALERT_EMAIL", sender)
    if not sender or not password:
        return False
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"]    = sender
        msg["To"]      = to_addr
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(sender, password)
            server.sendmail(sender, [to_addr], msg.as_string())
        return True
    except Exception:
        return False


def send_bark(title: str, body: str) -> bool:
    key = os.environ.get("BARK_KEY", "")
    if not key:
        return False
    try:
        resp = requests.post(
            f"https://api.day.app/{key}",
            json={"title": title, "body": body, "group": "量化信号", "sound": "bell"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def push_channels_status() -> dict:
    return {
        "email":    bool(os.environ.get("SMTP_EMAIL") and os.environ.get("SMTP_PASSWORD")),
        "bark":     bool(os.environ.get("BARK_KEY")),
        "telegram": bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")),
    }


# ── Signal detection ──────────────────────────────────────────────────────────

def _detect_signals(current: list[dict], previous: dict) -> list[dict]:
    """Compare current scan vs previous snapshot, emit signal events."""
    now_str = datetime.now(HKT).strftime("%Y-%m-%d %H:%M HKT")
    events  = []

    for stock in current:
        tk    = stock["ticker"]
        prev  = previous.get(tk, {})
        sig   = stock["signal"]
        psig  = prev.get("signal")
        rsi   = stock["rsi"]
        prsi  = prev.get("rsi")
        ret1m = stock.get("ret_1m", 0)

        strat  = stock.get("strategy", "momentum")
        as_of  = stock.get("as_of", "")
        price  = stock["price"]
        common = {
            "time": now_str, "ticker": tk, "name": stock["name"], "market": stock["market"],
            "strategy": strat, "as_of": as_of,
            "price": price, "score": stock["score"], "rsi": rsi,
        }

        # NOTE: every rule below requires a previous snapshot (psig/prsi not None).
        # The first scan after a restart only establishes the baseline silently —
        # otherwise every redeploy re-fires all currently-active signals.

        # 1. Signal flip into BUY
        if sig == "BUY" and psig in ("HOLD", "SELL"):
            if strat == "mean_reversion":
                title  = f"{tk} 超卖反弹买点"
                detail = f"触发原因：RSI从{prsi}跌破30至{rsi}（超卖）· 昨收价 {price}"
            else:
                title  = f"{tk} 进入买入区"
                detail = f"触发原因：综合评分升至{stock['score']}（突破62动量买入线）· RSI {rsi} · 昨收价 {price}"
            events.append({**common, "type": "BUY_SIGNAL", "emoji": "🟢", "title": title, "detail": detail})

        # 2. Signal flip into SELL
        elif sig == "SELL" and psig in ("HOLD", "BUY"):
            if strat == "mean_reversion":
                detail = f"触发原因：RSI升至{rsi}（突破70超买）· 昨收价 {price}"
            else:
                detail = f"触发原因：综合评分跌至{stock['score']}（跌破38卖出线）· RSI {rsi} · 昨收价 {price}"
            events.append({**common, "type": "SELL_SIGNAL", "emoji": "🔴",
                           "title": f"{tk} 进入卖出/回避区", "detail": detail})

        # 3. RSI oversold crossing — US only; for HK/CN this IS the BUY signal
        if strat == "momentum" and prsi is not None and rsi < 30 and prsi >= 30:
            events.append({**common, "type": "OVERSOLD", "emoji": "💎",
                           "title": f"{tk} RSI超卖 ({rsi})",
                           "detail": f"触发原因：RSI从{prsi}跌破30 · 昨收价 {price} · 近1月{'+' if ret1m>0 else ''}{ret1m}% · 关注分批埋伏机会"})

        # 4. RSI overbought crossing
        if prsi is not None and rsi > 75 and prsi <= 75:
            events.append({**common, "type": "OVERBOUGHT", "emoji": "⚠️",
                           "title": f"{tk} RSI超买 ({rsi})",
                           "detail": f"触发原因：RSI从{prsi}升破75 · 昨收价 {price} · 持有者考虑止盈1/3，勿追高"})

    return events


def _format_signal_push(events: list[dict]) -> tuple[str, str]:
    """Return (plain, html) push text with actionable guidance.

    Position sizes are fixed per strategy (historical replay showed score
    level above the threshold adds no extra edge, so no score-based sizing).
    """
    plain_lines = [f"📡 量化监控 · {len(events)} 条新信号\n"]
    html_lines  = [f"📡 <b>量化监控</b> · {len(events)} 条新信号\n"]

    for e in events[:10]:
        price = e.get("price", 0)
        t     = e["type"]

        if t == "BUY_SIGNAL":
            stop = round(price * 0.92, 2)
            if e.get("strategy") == "mean_reversion":
                tp = round(price * 1.10, 2)
                guidance_p = f"  💡操作：{price}附近试探仓5-8% · 止损{stop}(-8%) · 止盈{tp}(+10%)或RSI回到55"
                guidance_h = f"  💡{price}附近试探仓5-8% | 🛑{stop}(-8%) | 🎯{tp}(+10%)或RSI>55"
            else:
                tp = round(price * 1.25, 2)
                guidance_p = f"  💡操作：{price}附近建仓8-12% · 止损{stop}(-8%) · 止盈{tp}(+25%)"
                guidance_h = f"  💡{price}附近建仓8-12% | 🛑{stop}(-8%) | 🎯{tp}(+25%)"
        elif t == "SELL_SIGNAL":
            guidance_p = guidance_h = f"  💡操作：持有者{price}附近减仓至5%以下或清仓"
        elif t == "OVERSOLD":
            stop = round(price * 0.92, 2)
            guidance_p = f"  💡操作：{price}附近可小仓5-8%试探 · 止损{stop} · 等RSI回升>35再加仓"
            guidance_h = f"  💡{price}附近小仓5-8%试探 | 🛑{stop} | 等RSI>35再加仓"
        elif t == "OVERBOUGHT":
            guidance_p = guidance_h = f"  💡操作：持有者{price}附近止盈1/3锁利 · 未持有勿追高"
        else:
            guidance_p = guidance_h = ""

        plain_lines += [f"{e['emoji']} {e['title']} [{e['market']}]", f"  {e['detail']}", guidance_p, ""]
        html_lines  += [f"{e['emoji']} <b>{e['title']}</b> [{e['market']}]", f"  {e['detail']}", guidance_h, ""]

    if len(events) > 10:
        note = f"…另 {len(events)-10} 条信号，详见 /signals"
        plain_lines.append(note); html_lines.append(note)

    as_of = next((e.get("as_of") for e in events if e.get("as_of")), "")
    data_note = f"📅 数据基准：{as_of} 日线收盘价（非实时，每个交易日收盘后更新）"
    plain_lines.append(data_note)
    html_lines.append(data_note)
    return "\n".join(plain_lines), "\n".join(html_lines)


def _notify(events: list[dict]):
    """Push events to all configured channels (batched into one message)."""
    if not events:
        return
    plain, html = _format_signal_push(events)
    subject = f"📡 {events[0]['title']}" + (f" 等{len(events)}条" if len(events) > 1 else "")
    send_email(subject, plain)
    # Bark gets the full plain text too — title-only pushes were unreadable
    send_bark(f"📡 {len(events)}条新信号", plain)
    send_telegram(html)


# ── Market session summaries ──────────────────────────────────────────────────
# Sent once per session at open (+15 min) and close (-10 min)

_SUMMARY_JOBS = [
    {"label": "港A股开盘", "icon": "📈", "markets": ["HK", "CN"], "hh": 9,  "mm": 45, "tag": "hkcn_open"},
    {"label": "A股收盘",   "icon": "🔔", "markets": ["CN"],       "hh": 14, "mm": 50, "tag": "cn_close"},
    {"label": "港股收盘",  "icon": "🔔", "markets": ["HK"],       "hh": 15, "mm": 50, "tag": "hk_close"},
    {"label": "美股开盘",  "icon": "📈", "markets": ["US"],       "hh": 21, "mm": 45, "tag": "us_open"},
    {"label": "美股收盘",  "icon": "🔔", "markets": ["US"],       "hh": 3,  "mm": 50, "tag": "us_close"},
]
_SUMMARY_WINDOW_MIN = 20  # fire if within ±20 min of scheduled time


def _should_send_summary(job: dict, state: dict, now: datetime) -> bool:
    last = state.get(f"summary_{job['tag']}")
    if last:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() < 6 * 3600:
                return False
        except Exception:
            pass
    now_m = now.hour * 60 + now.minute
    tgt_m = job["hh"] * 60 + job["mm"]
    diff  = abs(now_m - tgt_m)
    return min(diff, 1440 - diff) <= _SUMMARY_WINDOW_MIN


def _send_market_summary(job: dict, current: list[dict], state: dict, now: datetime):
    stocks = [s for s in current if s["market"] in job["markets"]]
    if not stocks:
        return

    buys  = sorted([s for s in stocks if s["signal"] == "BUY"],  key=lambda x: -x["score"])
    sells = [s for s in stocks if s["signal"] == "SELL"]
    holds = [s for s in stocks if s["signal"] == "HOLD"]
    oversold   = sorted([s for s in stocks if s["rsi"] < 35], key=lambda x: x["rsi"])
    overbought = sorted([s for s in stocks if s["rsi"] > 72], key=lambda x: -x["rsi"])

    # Market health: % of stocks with positive 1m momentum
    up_count = sum(1 for s in stocks if s.get("ret_1m", 0) > 0)
    health   = round(up_count / len(stocks) * 100) if stocks else 0
    mood     = "偏强" if health >= 60 else ("偏弱" if health <= 40 else "中性")

    now_s = now.strftime("%H:%M HKT")
    icon  = job["icon"]
    label = job["label"]

    header_p = f"{icon} {label}扫描 ({now_s})"
    header_h = f"{icon} <b>{label}扫描</b> ({now_s})"
    stats    = f"覆盖{len(stocks)}只 | 🟢{len(buys)}买 ⚪{len(holds)}持 🔴{len(sells)}卖 | 市场{mood}({health}%涨)"

    plain_lines = [header_p, stats, ""]
    html_lines  = [header_h, stats, ""]

    if buys:
        plain_lines.append("🎯 买入区")
        html_lines.append("🎯 <b>买入区</b>")
        for s in buys[:5]:
            stop = round(s["price"] * 0.92, 2)
            if s.get("strategy") == "mean_reversion":
                pos, tag = "5-8%试探", "超卖反弹"
            else:
                pos, tag = "8-12%", "动量"
            line_p = f"  {s['ticker']} [{tag}] 评分{s['score']} RSI{s['rsi']} 昨收{s['price']} | 建仓{pos} 止损{stop}"
            line_h = f"  <b>{s['ticker']}</b> [{tag}] 评分{s['score']} RSI{s['rsi']} 昨收{s['price']} | 建仓{pos} 🛑{stop}"
            plain_lines.append(line_p); html_lines.append(line_h)
        plain_lines.append(""); html_lines.append("")

    if oversold:
        plain_lines.append("💎 RSI超卖（潜在机会）")
        html_lines.append("💎 <b>RSI超卖（潜在机会）</b>")
        for s in oversold[:3]:
            line = f"  {s['ticker']} RSI={s['rsi']} 昨收{s['price']} 1月{s.get('ret_1m',0):+.1f}%"
            plain_lines.append(line); html_lines.append(line)
        plain_lines.append(""); html_lines.append("")

    if overbought:
        plain_lines.append("⚠️ RSI超买（考虑止盈）")
        html_lines.append("⚠️ <b>RSI超买（考虑止盈）</b>")
        for s in overbought[:3]:
            line = f"  {s['ticker']} RSI={s['rsi']} 昨收{s['price']}"
            plain_lines.append(line); html_lines.append(line)
        plain_lines.append(""); html_lines.append("")

    plain_lines.append("⚠️ 数据为昨日收盘，仅供参考")
    html_lines.append("⚠️ 数据为昨日收盘，仅供参考")

    plain = "\n".join(plain_lines)
    html  = "\n".join(html_lines)

    send_email(f"{icon} {label}扫描摘要", plain)
    send_bark(f"{icon} {label}", f"🟢{len(buys)}买 ⚪{len(holds)}持 🔴{len(sells)}卖 | 市场{mood}")
    send_telegram(html)

    state[f"summary_{job['tag']}"] = now.isoformat()


def _maybe_send_summaries(current: list[dict], state: dict):
    now = datetime.now(HKT)
    for job in _SUMMARY_JOBS:
        if _should_send_summary(job, state, now):
            _send_market_summary(job, current, state, now)


# ── Monitor loop ──────────────────────────────────────────────────────────────

_monitor_started = False
_monitor_lock    = threading.Lock()


def _scan_once():
    state    = _load_json(STATE_FILE, {})
    previous = state.get("snapshot", {})

    result  = run_scan()
    current = result.get("all", [])
    if not current:
        return

    events = _detect_signals(current, previous)

    # Append to signal history (newest first)
    if events:
        history = _load_json(SIGNALS_FILE, [])
        history = events + history
        _save_json(SIGNALS_FILE, history[:MAX_SIGNALS_KEPT])
        _notify(events)

    # Track buy-signal outcomes: record new, resolve old (7-day check)
    price_map = {s["ticker"]: s["price"] for s in current}
    outcomes  = state.get("outcomes", [])
    now_dt    = datetime.now(HKT)
    for e in events:
        if e["type"] == "BUY_SIGNAL":
            outcomes.append({
                "ticker": e["ticker"], "price_entry": e["price"],
                "fired_at": now_dt.isoformat(), "resolved": False,
            })
    for o in outcomes:
        if o.get("resolved"):
            continue
        try:
            fired = datetime.fromisoformat(o["fired_at"])
            if (now_dt - fired).days >= 7 and o["ticker"] in price_map:
                current_price = price_map[o["ticker"]]
                change = (current_price - o["price_entry"]) / o["price_entry"]
                o["result"]        = "win" if change > 0.03 else ("loss" if change < -0.05 else "neutral")
                o["price_exit"]    = current_price
                o["change_pct"]    = round(change * 100, 1)
                o["resolved"]      = True
                o["resolved_at"]   = now_dt.isoformat()
        except Exception:
            pass
    state["outcomes"] = outcomes[-200:]  # keep last 200

    # Check if we should send a market open/close summary
    _maybe_send_summaries(current, state)

    # Save snapshot for next comparison
    now = datetime.now(HKT)
    state.update({
        "snapshot":   {s["ticker"]: {"signal": s["signal"], "rsi": s["rsi"], "score": s["score"]} for s in current},
        "last_scan":  now.strftime("%Y-%m-%d %H:%M HKT"),
        "next_scan":  (now + timedelta(minutes=SCAN_INTERVAL_MIN)).strftime("%Y-%m-%d %H:%M HKT"),
        "scan_count": state.get("scan_count", 0) + 1,
    })
    _save_json(STATE_FILE, state)


def _monitor_loop():
    # Initial delay so app boot isn't slowed down
    time.sleep(20)
    while True:
        try:
            _scan_once()
        except Exception:
            traceback.print_exc()
        time.sleep(SCAN_INTERVAL_MIN * 60)


def start_monitor():
    """Start the background monitor thread (idempotent)."""
    global _monitor_started
    with _monitor_lock:
        if _monitor_started:
            return
        t = threading.Thread(target=_monitor_loop, daemon=True, name="quant-monitor")
        t.start()
        _monitor_started = True
        print(f"  📡 Quant monitor started — scanning every {SCAN_INTERVAL_MIN} min")
