"""
yfinance_adapter.py
====================
Drop-in replacement for ai-hedge-fund's src/tools/api.py.
Uses yfinance (free, no API key) instead of financialdatasets.ai.

Copied to hedge_fund_src/src/tools/api.py by setup_hedge_fund.sh.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data models matching ai-hedge-fund's src/data/models.py
# ---------------------------------------------------------------------------

class Price:
    def __init__(self, open, close, high, low, volume, time):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.time = time


class FinancialMetrics:
    def __init__(self, ticker, report_period, period="ttm", currency="USD", **kwargs):
        self.ticker = ticker
        self.report_period = report_period
        self.period = period
        self.currency = currency
        # All optional fields
        defaults = [
            "market_cap", "enterprise_value", "price_to_earnings_ratio",
            "price_to_book_ratio", "price_to_sales_ratio",
            "enterprise_value_to_ebitda_ratio", "enterprise_value_to_revenue_ratio",
            "free_cash_flow_yield", "peg_ratio",
            "gross_margin", "operating_margin", "net_margin",
            "return_on_equity", "return_on_assets", "return_on_invested_capital",
            "asset_turnover", "inventory_turnover", "receivables_turnover",
            "days_sales_outstanding", "operating_cycle", "working_capital_turnover",
            "current_ratio", "quick_ratio", "cash_ratio", "operating_cash_flow_ratio",
            "debt_to_equity", "debt_to_assets", "interest_coverage",
            "revenue_growth", "earnings_growth", "book_value_growth",
            "earnings_per_share_growth", "free_cash_flow_growth",
            "operating_income_growth", "ebitda_growth",
            "earnings_per_share", "book_value_per_share",
            "free_cash_flow_per_share", "payout_ratio",
        ]
        for f in defaults:
            setattr(self, f, kwargs.get(f))


class LineItem:
    def __init__(self, ticker, report_period, period="ttm", currency="USD", **kwargs):
        self.ticker = ticker
        self.report_period = report_period
        self.period = period
        self.currency = currency
        for k, v in kwargs.items():
            setattr(self, k, v)


class InsiderTrade:
    def __init__(self, ticker, **kwargs):
        self.ticker = ticker
        fields = [
            "issuer", "name", "title", "is_board_director", "transaction_date",
            "transaction_shares", "transaction_price_per_share", "transaction_value",
            "shares_owned_before_transaction", "shares_owned_after_transaction",
            "security_title", "filing_date",
        ]
        for f in fields:
            setattr(self, f, kwargs.get(f))


class CompanyNews:
    def __init__(self, ticker, title="", author="", source="", date="", url="", sentiment="neutral"):
        self.ticker = ticker
        self.title = title
        self.author = author
        self.source = source
        self.date = date
        self.url = url
        self.sentiment = sentiment


# ---------------------------------------------------------------------------
# Internal cache & helpers
# ---------------------------------------------------------------------------

_ticker_cache: dict[str, yf.Ticker] = {}


def _t(symbol: str) -> yf.Ticker:
    if symbol not in _ticker_cache:
        _ticker_cache[symbol] = yf.Ticker(symbol)
    return _ticker_cache[symbol]


def _safe(val):
    """Convert NaN / Inf / non-numeric → None; keep floats."""
    if val is None:
        return None
    try:
        if isinstance(val, (pd.Series, pd.DataFrame)):
            return None
        fval = float(val)
        return None if (np.isnan(fval) or np.isinf(fval)) else fval
    except (TypeError, ValueError):
        return None


def _row(df: pd.DataFrame, *keys) -> pd.Series:
    """Return first matching row from a DataFrame by index label."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series(dtype=float)


def _col_str(col) -> str:
    if hasattr(col, "strftime"):
        return col.strftime("%Y-%m-%d")
    return str(col)


# ---------------------------------------------------------------------------
# Public API — mirrors ai-hedge-fund's src/tools/api.py signatures exactly
# ---------------------------------------------------------------------------

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    hist = _t(ticker).history(start=start_date, end=end_date, auto_adjust=True)
    if hist.empty:
        return []
    return [
        Price(
            open=float(row["Open"]),
            close=float(row["Close"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            volume=int(row["Volume"]),
            time=ts.strftime("%Y-%m-%d"),
        )
        for ts, row in hist.iterrows()
    ]


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    t = _t(ticker)
    info = t.info or {}

    try:
        income  = t.income_stmt
        balance = t.balance_sheet
        cashflow = t.cashflow
    except Exception:
        income = balance = cashflow = pd.DataFrame()

    cols = income.columns if not income.empty else (
        balance.columns if not balance.empty else [pd.Timestamp(end_date)]
    )

    results = []
    for col in list(cols)[:limit]:
        revenue   = _safe(_row(income,  "Total Revenue").get(col))
        gp        = _safe(_row(income,  "Gross Profit").get(col))
        op_inc    = _safe(_row(income,  "Operating Income", "EBIT").get(col))
        net_inc   = _safe(_row(income,  "Net Income").get(col))
        int_exp   = _safe(_row(income,  "Interest Expense").get(col))

        tot_assets = _safe(_row(balance, "Total Assets").get(col))
        equity     = _safe(_row(balance, "Stockholders Equity",
                                         "Total Stockholders Equity").get(col))
        tot_debt   = _safe(_row(balance, "Total Debt", "Long Term Debt").get(col))
        cur_assets = _safe(_row(balance, "Current Assets").get(col))
        cur_liab   = _safe(_row(balance, "Current Liabilities").get(col))
        inventory  = _safe(_row(balance, "Inventory").get(col))

        op_cf  = _safe(_row(cashflow, "Operating Cash Flow").get(col))
        capex  = _safe(_row(cashflow, "Capital Expenditure").get(col))
        fcf    = ((op_cf + capex) if (op_cf is not None and capex is not None)
                  else _safe(info.get("freeCashflow")))

        shares = info.get("sharesOutstanding") or 1

        def _div(a, b):
            return (a / b) if (a is not None and b and b != 0) else None

        results.append(FinancialMetrics(
            ticker=ticker,
            report_period=_col_str(col),
            period=period,
            market_cap              = _safe(info.get("marketCap")),
            price_to_earnings_ratio = _safe(info.get("trailingPE")),
            price_to_book_ratio     = _safe(info.get("priceToBook")),
            price_to_sales_ratio    = _safe(info.get("priceToSalesTrailing12Months")),
            peg_ratio               = _safe(info.get("pegRatio")),
            gross_margin            = _div(gp, revenue)     or _safe(info.get("grossMargins")),
            operating_margin        = _div(op_inc, revenue) or _safe(info.get("operatingMargins")),
            net_margin              = _div(net_inc, revenue) or _safe(info.get("profitMargins")),
            return_on_equity        = _div(net_inc, equity) or _safe(info.get("returnOnEquity")),
            return_on_assets        = _div(net_inc, tot_assets) or _safe(info.get("returnOnAssets")),
            debt_to_equity          = _div(tot_debt, equity) or _safe(info.get("debtToEquity")),
            current_ratio           = _div(cur_assets, cur_liab) or _safe(info.get("currentRatio")),
            quick_ratio             = (_div((cur_assets or 0) - (inventory or 0), cur_liab)
                                       or _safe(info.get("quickRatio"))),
            interest_coverage       = _div(op_inc, abs(int_exp) if int_exp else None),
            earnings_per_share      = _safe(info.get("trailingEps")) or _div(net_inc, shares),
            book_value_per_share    = _div(equity, shares),
            free_cash_flow_per_share= _div(fcf, shares),
            payout_ratio            = _safe(info.get("payoutRatio")),
            revenue_growth          = _safe(info.get("revenueGrowth")),
            earnings_growth         = _safe(info.get("earningsGrowth")),
        ))

    return results


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    t = _t(ticker)
    info = t.info or {}

    try:
        income   = t.income_stmt
        balance  = t.balance_sheet
        cashflow = t.cashflow
    except Exception:
        income = balance = cashflow = pd.DataFrame()

    # Maps from canonical name → yfinance row labels (in priority order)
    INCOME = {
        "revenue":                          ["Total Revenue"],
        "gross_profit":                     ["Gross Profit"],
        "operating_income":                 ["Operating Income", "EBIT"],
        "net_income":                       ["Net Income"],
        "ebitda":                           ["EBITDA", "Normalized EBITDA"],
        "interest_expense":                 ["Interest Expense"],
        "depreciation_and_amortization":    ["Depreciation And Amortization",
                                             "Reconciled Depreciation"],
        "ebit":                             ["EBIT", "Operating Income"],
    }
    BALANCE = {
        "total_assets":             ["Total Assets"],
        "total_liabilities":        ["Total Liabilities Net Minority Interest"],
        "shareholders_equity":      ["Stockholders Equity", "Total Stockholders Equity"],
        "total_debt":               ["Total Debt", "Long Term Debt"],
        "cash_and_equivalents":     ["Cash And Cash Equivalents",
                                     "Cash Cash Equivalents And Short Term Investments"],
        "inventory":                ["Inventory"],
        "current_assets":           ["Current Assets"],
        "current_liabilities":      ["Current Liabilities"],
        "goodwill_and_intangibles": ["Goodwill And Other Intangible Assets", "Goodwill"],
    }
    CASHFLOW = {
        "operating_cash_flow":                      ["Operating Cash Flow"],
        "capital_expenditure":                      ["Capital Expenditure"],
        "dividends_and_other_cash_distributions":   ["Cash Dividends Paid"],
        "issuance_or_purchase_of_equity_shares":    ["Repurchase Of Capital Stock",
                                                     "Common Stock Issuance"],
    }

    cols = income.columns if not income.empty else (
        balance.columns if not balance.empty else [pd.Timestamp(end_date)]
    )

    results = []
    for col in list(cols)[:limit]:
        kwargs: dict = {}
        for item in line_items:
            key = item.lower().replace(" ", "_").replace("-", "_")

            if key in INCOME:
                kwargs[key] = _safe(_row(income, *INCOME[key]).get(col))

            elif key in BALANCE:
                kwargs[key] = _safe(_row(balance, *BALANCE[key]).get(col))

            elif key in CASHFLOW:
                kwargs[key] = _safe(_row(cashflow, *CASHFLOW[key]).get(col))

            elif key == "free_cash_flow":
                op  = _safe(_row(cashflow, "Operating Cash Flow").get(col))
                cap = _safe(_row(cashflow, "Capital Expenditure").get(col))
                kwargs[key] = ((op + cap) if (op is not None and cap is not None)
                               else _safe(info.get("freeCashflow")))

            elif key == "working_capital":
                ca = _safe(_row(balance, "Current Assets").get(col))
                cl = _safe(_row(balance, "Current Liabilities").get(col))
                kwargs[key] = ((ca - cl) if (ca is not None and cl is not None) else None)

            elif key == "outstanding_shares":
                kwargs[key] = _safe(info.get("sharesOutstanding"))

            elif key in ("net_income_to_common_shareholders",):
                kwargs[key] = _safe(_row(income, "Net Income").get(col))

            else:
                kwargs[key] = None

        results.append(LineItem(
            ticker=ticker,
            report_period=_col_str(col),
            period=period,
            **kwargs,
        ))

    return results


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    try:
        df = _t(ticker).insider_transactions
        if df is None or df.empty:
            return []
        out = []
        for _, row in df.head(limit).iterrows():
            out.append(InsiderTrade(
                ticker=ticker,
                name=str(row.get("Insider", "")),
                title=str(row.get("Position", "")),
                transaction_date=str(row.get("Start Date", "")),
                transaction_shares=_safe(row.get("Shares")),
                transaction_value=_safe(row.get("Value")),
            ))
        return out
    except Exception:
        return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 50,
) -> list[CompanyNews]:
    try:
        news = _t(ticker).news or []
        return [
            CompanyNews(
                ticker=ticker,
                title=item.get("title", ""),
                source=item.get("publisher", ""),
                url=item.get("link", ""),
                date=datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).strftime("%Y-%m-%d"),
            )
            for item in news[:limit]
        ]
    except Exception:
        return []


def get_market_cap(ticker: str, end_date: str) -> Optional[float]:
    try:
        return _safe(_t(ticker).info.get("marketCap"))
    except Exception:
        return None


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    hist = _t(ticker).history(start=start_date, end=end_date, auto_adjust=True)
    if hist.empty:
        return pd.DataFrame()
    hist = hist[["Open", "High", "Low", "Close", "Volume"]].rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    hist.index = pd.to_datetime(hist.index)
    return hist.apply(pd.to_numeric, errors="coerce")
