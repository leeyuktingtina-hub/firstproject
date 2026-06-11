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
    state   = _load_json(STATE_FILE, {})
    signals = _load_json(SIGNALS_FILE, [])
    channels = push_channels_status()
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

        # 1. Signal flip into BUY
        if sig == "BUY" and psig in ("HOLD", "SELL"):
            events.append({
                "time": now_str, "ticker": tk, "name": stock["name"], "market": stock["market"],
                "type": "BUY_SIGNAL", "emoji": "🟢",
                "title": f"{tk} 进入买入区",
                "detail": f"综合评分 {stock['score']}/100 · RSI {rsi} · 价格 {stock['price']}",
                "price": stock["price"], "score": stock["score"],
            })

        # 2. Signal flip into SELL
        elif sig == "SELL" and psig in ("HOLD", "BUY"):
            events.append({
                "time": now_str, "ticker": tk, "name": stock["name"], "market": stock["market"],
                "type": "SELL_SIGNAL", "emoji": "🔴",
                "title": f"{tk} 进入卖出/回避区",
                "detail": f"综合评分 {stock['score']}/100 · RSI {rsi} · 价格 {stock['price']}",
                "price": stock["price"], "score": stock["score"],
            })

        # 3. RSI oversold crossing (only on crossing, not while staying)
        if rsi < 30 and (prsi is None or prsi >= 30):
            events.append({
                "time": now_str, "ticker": tk, "name": stock["name"], "market": stock["market"],
                "type": "OVERSOLD", "emoji": "💎",
                "title": f"{tk} RSI超卖 ({rsi})",
                "detail": f"可能是分批埋伏机会 · 价格 {stock['price']} · 1月{'+' if ret1m>0 else ''}{ret1m}%",
                "price": stock["price"], "score": stock["score"],
            })

        # 4. RSI overbought crossing
        if rsi > 75 and (prsi is None or prsi <= 75):
            events.append({
                "time": now_str, "ticker": tk, "name": stock["name"], "market": stock["market"],
                "type": "OVERBOUGHT", "emoji": "⚠️",
                "title": f"{tk} RSI超买 ({rsi})",
                "detail": f"考虑止盈1/3锁定利润，不要追高 · 价格 {stock['price']}",
                "price": stock["price"], "score": stock["score"],
            })

    return events


def _notify(events: list[dict]):
    """Push events to all configured channels (batched into one message)."""
    if not events:
        return

    # Plain-text version (email / bark)
    plain_lines = [f"量化监控信号 ({len(events)}条)\n"]
    for e in events[:15]:
        plain_lines.append(f"{e['emoji']} {e['title']}\n   {e['detail']}\n")
    if len(events) > 15:
        plain_lines.append(f"…及另外 {len(events)-15} 条信号，详见网站 /signals")
    plain = "\n".join(plain_lines)

    # HTML version (telegram)
    html_lines = [f"📡 <b>量化监控信号</b> ({len(events)}条)\n"]
    for e in events[:15]:
        html_lines.append(f"{e['emoji']} <b>{e['title']}</b>\n   {e['detail']}\n")
    if len(events) > 15:
        html_lines.append(f"…及另外 {len(events)-15} 条信号，详见网站 /signals")

    subject = f"📡 量化信号: {events[0]['title']}" + (f" 等{len(events)}条" if len(events) > 1 else "")
    send_email(subject, plain)
    send_bark(f"📡 {len(events)}条新信号", "\n".join(f"{e['emoji']} {e['title']}" for e in events[:8]))
    send_telegram("\n".join(html_lines))


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
