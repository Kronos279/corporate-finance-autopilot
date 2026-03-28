from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from app.config import settings
from app.models.company import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    HistoricalFinancials,
    IncomeStatement,
    KeyStats,
    PricePoint,
)
logger = logging.getLogger(__name__)


def _safe(series: pd.Series, keys: list[str], default=None):
    """Safely extract a float value from a pandas Series trying multiple key names."""
    for key in keys:
        if key in series.index:
            val = series[key]
            if pd.notna(val):
                return float(val)
    return default


def get_profile(ticker: str) -> CompanyProfile:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        logger.warning("yfinance profile error for %s: %s", ticker, e)
        info = {}

    city = info.get("city", "")
    state = info.get("state", "")
    country = info.get("country", "")
    hq_parts = [p for p in [city, state, country] if p]

    # Extract CEO from companyOfficers list
    ceo_name = None
    for officer in info.get("companyOfficers") or []:
        title = (officer.get("title") or "").lower()
        if "ceo" in title or "chief executive officer" in title:
            ceo_name = officer.get("name")
            break

    # Build logo URL from company website domain via logo.dev
    website = info.get("website")
    logo_url = None
    if website:
        try:
            from urllib.parse import urlparse
            domain = urlparse(website).netloc or website
            domain = domain.removeprefix("www.")
            if settings.logo_dev_token:
                logo_url = f"https://img.logo.dev/{domain}?token={settings.logo_dev_token}&format=png"
        except Exception:
            pass

    return CompanyProfile(
        ticker=ticker.upper(),
        name=info.get("longName") or info.get("shortName") or ticker.upper(),
        sector=info.get("sector"),
        industry=info.get("industry"),
        description=info.get("longBusinessSummary"),
        market_cap=info.get("marketCap"),
        employees=info.get("fullTimeEmployees"),
        website=website,
        headquarters=", ".join(hq_parts) if hq_parts else None,
        ceo=ceo_name,
        logo_url=logo_url,
        exchange=info.get("exchange"),
        currency=info.get("currency", "USD"),
    )


def get_financials(ticker: str) -> HistoricalFinancials:
    try:
        stock = yf.Ticker(ticker)
        inc_df = stock.income_stmt
        bs_df = stock.balance_sheet
        cf_df = stock.cashflow
    except Exception as e:
        logger.warning("yfinance financials error for %s: %s", ticker, e)
        return HistoricalFinancials(ticker=ticker.upper())

    income_stmts: list[IncomeStatement] = []
    balance_sheets: list[BalanceSheet] = []
    cash_flows: list[CashFlowStatement] = []

    # ── Income Statements ──
    if inc_df is not None and not inc_df.empty:
        for col in inc_df.columns:
            year = col.year if hasattr(col, "year") else int(str(col)[:4])
            s = inc_df[col]
            revenue = _safe(s, ["Total Revenue", "Operating Revenue", "Revenue"])
            if not revenue or revenue == 0:
                continue
            cost_of_rev = _safe(s, ["Cost Of Revenue", "Reconciled Cost Of Revenue"], 0)
            gross_profit = _safe(s, ["Gross Profit"]) or (revenue - cost_of_rev)
            opex = _safe(s, ["Operating Expense"], 0)
            op_income = _safe(s, ["Operating Income", "Total Operating Income As Reported"]) or (gross_profit - opex)
            ebitda = _safe(s, ["EBITDA", "Normalized EBITDA"]) or op_income
            net_income = _safe(s, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"], 0)
            eps = _safe(s, ["Basic EPS", "Diluted EPS"])

            income_stmts.append(IncomeStatement(
                fiscal_year=year,
                revenue=revenue,
                cost_of_revenue=cost_of_rev,
                gross_profit=gross_profit,
                operating_expenses=opex,
                operating_income=op_income,
                ebitda=ebitda,
                net_income=net_income,
                eps=eps,
            ))

    # ── Balance Sheets ──
    if bs_df is not None and not bs_df.empty:
        for col in bs_df.columns:
            year = col.year if hasattr(col, "year") else int(str(col)[:4])
            s = bs_df[col]
            balance_sheets.append(BalanceSheet(
                fiscal_year=year,
                total_assets=_safe(s, ["Total Assets"], 0),
                total_liabilities=_safe(s, ["Total Liabilities Net Minority Interest", "Total Liabilities"], 0),
                total_equity=_safe(s, ["Stockholders Equity", "Total Equity Gross Minority Interest"], 0),
                cash_and_equivalents=_safe(s, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"], 0),
                total_debt=_safe(s, ["Total Debt", "Long Term Debt And Capital Lease Obligation"], 0),
                current_assets=_safe(s, ["Current Assets"], 0),
                current_liabilities=_safe(s, ["Current Liabilities"], 0),
            ))

    # ── Cash Flow Statements ──
    if cf_df is not None and not cf_df.empty:
        for col in cf_df.columns:
            year = col.year if hasattr(col, "year") else int(str(col)[:4])
            s = cf_df[col]
            cash_flows.append(CashFlowStatement(
                fiscal_year=year,
                operating_cash_flow=_safe(s, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], 0),
                capex=_safe(s, ["Capital Expenditure", "Purchase Of PPE"], 0),
                dividends_paid=_safe(s, ["Common Stock Dividend Paid", "Cash Dividends Paid"], 0),
                share_buybacks=_safe(s, ["Repurchase Of Capital Stock", "Common Stock Payments"], 0),
            ))

    # Sort ascending by year
    income_stmts.sort(key=lambda x: x.fiscal_year)
    balance_sheets.sort(key=lambda x: x.fiscal_year)
    cash_flows.sort(key=lambda x: x.fiscal_year)

    return HistoricalFinancials(
        ticker=ticker.upper(),
        income_statements=income_stmts,
        balance_sheets=balance_sheets,
        cash_flows=cash_flows,
    )


def get_price_history(ticker: str, period: str = "5y") -> list[PricePoint]:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
    except Exception as e:
        logger.warning("yfinance price history error for %s: %s", ticker, e)
        return []

    points: list[PricePoint] = []
    if df is not None and not df.empty:
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            close = row.get("Close", 0)
            if pd.isna(close):
                continue
            vol = row.get("Volume", 0)
            points.append(PricePoint(
                date=date_str,
                open=round(float(row.get("Open", 0) or 0), 2),
                high=round(float(row.get("High", 0) or 0), 2),
                low=round(float(row.get("Low", 0) or 0), 2),
                close=round(float(close), 2),
                volume=int(vol if pd.notna(vol) else 0),
            ))
    return points


def get_key_stats(ticker: str) -> KeyStats:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        logger.warning("yfinance key stats error for %s: %s", ticker, e)
        info = {}

    return KeyStats(
        ticker=ticker.upper(),
        pe_ratio=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        ev_to_ebitda=info.get("enterpriseToEbitda"),
        price_to_book=info.get("priceToBook"),
        dividend_yield=info.get("dividendYield"),
        beta=info.get("beta"),
        fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
        fifty_two_week_low=info.get("fiftyTwoWeekLow"),
        current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
        shares_outstanding=info.get("sharesOutstanding"),
    )
