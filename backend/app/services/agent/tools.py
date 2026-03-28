from __future__ import annotations

import json
import logging

from app.config import settings
from app.models.company import Assumptions
from app.services.advisory.strategy import (
    analyze_capital_structure,
    suggest_funding_options,
    suggest_strategic_options,
)
from app.services.financial_engine.model_builder import dcf_valuation, derive_assumptions, forecast
from app.services.financial_engine.sensitivity import sensitivity_table
from app.services.ingestion.fred import get_macro_snapshot
from app.services.ingestion.sec_edgar import get_cik, get_company_facts, get_filings_list
from app.services.ingestion.yahoo_finance import (
    get_financials,
    get_key_stats,
    get_price_history,
    get_profile,
)

logger = logging.getLogger(__name__)

# ── OpenAI-compatible tool definitions for Together AI ──

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_company_profile",
            "description": "Fetch company profile: name, sector, market cap, description, headquarters, employees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker (e.g. AAPL, MSFT)"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_financials",
            "description": "Fetch 4-5 years of historical income statements, balance sheets, and cash flow statements.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_key_stats",
            "description": "Fetch valuation metrics: P/E, EV/EBITDA, beta, dividend yield, current price, shares outstanding.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_price_history",
            "description": "Fetch historical stock price data (OHLCV).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "period": {"type": "string", "description": "1y, 2y, 5y, 10y, max", "default": "5y"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_macro_data",
            "description": "Fetch macroeconomic context: 10-year treasury rate, fed funds rate, GDP growth, CPI inflation.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_financial_model",
            "description": "Build a 3-case financial model (base/upside/downside) with 5-year forecasts and DCF valuation based on real historical data.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sensitivity",
            "description": "Run 2D sensitivity analysis (revenue growth vs WACC) showing implied share price grid.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_advisory",
            "description": "Generate strategic advisory: capital structure analysis, funding options, and strategic recommendations.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_sec_filings",
            "description": "Search SEC EDGAR for recent company filings (10-K, 10-Q).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "form_type": {"type": "string", "default": "10-K"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Semantic search over ingested company documents (SEC filings, financials, profile). Use this to find specific facts, risk factors, or context from filings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "ticker": {"type": "string", "description": "Filter by ticker (optional)"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


async def execute_tool(tool_name: str, args: dict, ticker: str = "") -> dict:
    """Execute a named tool and return a JSON-serialisable result dict."""
    tk = args.get("ticker", ticker).upper()

    try:
        if tool_name == "fetch_company_profile":
            return get_profile(tk).model_dump()

        if tool_name == "fetch_financials":
            return get_financials(tk).model_dump()

        if tool_name == "fetch_key_stats":
            return get_key_stats(tk).model_dump()

        if tool_name == "fetch_price_history":
            period = args.get("period", "5y")
            points = get_price_history(tk, period)
            return {
                "ticker": tk,
                "data_points": len(points),
                "first": points[0].model_dump() if points else None,
                "last": points[-1].model_dump() if points else None,
            }

        if tool_name == "fetch_macro_data":
            return get_macro_snapshot().model_dump()

        if tool_name == "build_financial_model":
            financials = get_financials(tk)
            key_stats = get_key_stats(tk)
            macro = get_macro_snapshot()
            assumptions_dict = derive_assumptions(financials, key_stats=key_stats, macro=macro)
            starting_rev = (
                financials.income_statements[-1].revenue
                if financials.income_statements
                else settings.default_starting_revenue
            )
            current_price = key_stats.current_price or settings.default_current_price
            shares = key_stats.shares_outstanding or settings.default_shares_outstanding
            net_debt_val = 0.0
            if financials.balance_sheets:
                net_debt_val = financials.balance_sheets[-1].net_debt

            exit_mult = key_stats.ev_to_ebitda

            scenarios = {}
            for scenario_name, assumption_set in assumptions_dict.items():
                fcst = forecast(assumption_set, years=5, starting_revenue=starting_rev)
                dcf = dcf_valuation(
                    fcst,
                    assumption_set.wacc,
                    assumption_set.terminal_growth_rate,
                    current_price=current_price,
                    net_debt=net_debt_val,
                    shares_outstanding=shares,
                    exit_ev_ebitda=exit_mult,
                )
                scenarios[scenario_name] = {
                    "assumptions": assumption_set.model_dump(),
                    "forecast": [f.model_dump() for f in fcst],
                    "dcf": dcf.model_dump(),
                }
            return {"ticker": tk, "scenarios": scenarios}

        if tool_name == "run_sensitivity":
            financials = get_financials(tk)
            key_stats = get_key_stats(tk)
            starting_rev = (
                financials.income_statements[-1].revenue
                if financials.income_statements
                else settings.default_starting_revenue
            )
            net_debt_val = financials.balance_sheets[-1].net_debt if financials.balance_sheets else 0.0
            tbl = sensitivity_table(
                Assumptions(),
                param_x="revenue_growth_rate",
                param_y="wacc",
                range_x=[0.04, 0.06, 0.08, 0.10, 0.12],
                range_y=[0.08, 0.09, 0.10, 0.11, 0.12],
                starting_revenue=starting_rev,
                net_debt=net_debt_val,
                shares_outstanding=key_stats.shares_outstanding or settings.default_shares_outstanding,
                current_price=key_stats.current_price or settings.default_current_price,
            )
            return {"ticker": tk, "sensitivity": tbl.model_dump()}

        if tool_name == "generate_advisory":
            financials = get_financials(tk)
            macro = get_macro_snapshot()
            return {
                "ticker": tk,
                "capital_structure": analyze_capital_structure(financials),
                "funding_options": [r.model_dump() for r in suggest_funding_options(financials, macro)],
                "strategic_options": [r.model_dump() for r in suggest_strategic_options(financials)],
            }

        if tool_name == "search_sec_filings":
            form_type = args.get("form_type", "10-K")
            cik = get_cik(tk)
            if not cik:
                return {"error": f"CIK not found for {tk}"}
            filings = get_filings_list(cik, form_type)
            return {"ticker": tk, "cik": cik, "filings": filings}

        if tool_name == "rag_search":
            from app.services.rag import search as rag_search_fn
            query = args.get("query", "")
            rag_ticker = args.get("ticker", tk) or tk
            top_k = args.get("top_k", 5)
            results = await rag_search_fn(query, ticker=rag_ticker, top_k=top_k)
            return {"query": query, "ticker": rag_ticker, "results": results}

        return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.exception("Tool execution error for %s", tool_name)
        return {"error": str(e), "tool": tool_name}
