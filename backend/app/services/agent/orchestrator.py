from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable

from openai import AsyncOpenAI

from app.config import settings
from app.models.company import AgentResponse, AgentTrace
from app.services.agent.tools import TOOL_DEFINITIONS, execute_tool
from app.services.financial_engine.model_builder import dcf_valuation, derive_assumptions, forecast
from app.services.ingestion.fred import get_macro_snapshot
from app.services.ingestion.yahoo_finance import get_financials, get_key_stats, get_profile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior corporate finance analyst at a top-tier investment bank.
You have access to real-time financial data tools and can perform sophisticated analysis.

CRITICAL INSTRUCTIONS:
1. ANSWER THE USER'S ACTUAL QUESTION. Do not run every tool. Only call tools that are directly relevant to what the user asked.
2. Be conversational, specific, and opinionated. Give clear, actionable insights—not generic data dumps.
3. If the user asks about products, talk about PRODUCTS. If they ask about valuation, focus on VALUATION. Do not repeat the same boilerplate.
4. DO NOT start every response with the company profile summary. The user already knows what company they're asking about.
5. Reference specific numbers from tool results to support your points. Don't just list metrics—interpret them.
6. Structure your response with clear headings and bullet points when appropriate.
7. If you've already gathered data from previous tool calls in this conversation, USE IT—don't re-fetch the same data.
8. Give a direct answer FIRST, then supporting evidence. Never make the user wait through data dumps to get your actual opinion.

Available tools:
• fetch_company_profile – company info (name, sector, market cap, description, CEO)
• fetch_financials – 4-5 years of income statements, balance sheets, cash flows
• fetch_key_stats – valuation metrics (P/E, EV/EBITDA, beta, current price, shares outstanding)
• fetch_price_history – historical stock price data
• fetch_macro_data – treasury rates, GDP, CPI, fed funds rate
• build_financial_model – 3-scenario DCF model with data-driven assumptions
• run_sensitivity – 2D sensitivity analysis (e.g., growth vs WACC)
• generate_advisory – capital structure analysis & strategic recommendations
• search_sec_filings – SEC EDGAR filings lookup (10-K, 10-Q, etc.)
• rag_search – semantic search over ingested documents

TOOL SELECTION GUIDE:
- "Tell me about products/business" → fetch_company_profile (maybe search_sec_filings for detail)
- "Is it undervalued?" → fetch_key_stats + build_financial_model
- "What are the risks?" → fetch_financials + search_sec_filings + fetch_macro_data
- "Should I invest?" → build_financial_model + fetch_key_stats + generate_advisory
- "How are margins trending?" → fetch_financials (focus on margin analysis)
- "What about the balance sheet?" → fetch_financials (focus on balance sheet)
- General analysis → use your judgment, but be selective

NEVER call all tools. Pick 2-4 that are most relevant to the question."""

MAX_ITERATIONS = settings.agent_max_iterations


async def run(
    ticker: str,
    query: str | None = None,
    on_trace: Callable[[AgentTrace], Awaitable[None]] | None = None,
    conversation_history: list[dict] | None = None,
) -> AgentResponse:
    """Run the agentic analysis loop. If Together AI key isn't set, falls back to direct mode."""
    traces: list[AgentTrace] = []
    step = 1

    async def _log(trace_type: str, detail: str, data: dict | None = None) -> None:
        nonlocal step
        trace = AgentTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            step=step,
            type=trace_type,
            detail=detail,
            data=data,
        )
        traces.append(trace)
        step += 1
        if on_trace:
            await on_trace(trace)

    # ── Fallback: no LLM key ──
    if not settings.together_api_key:
        return await _run_direct(ticker, query, _log)

    # ── LLM-powered agentic loop ──
    client = AsyncOpenAI(
        api_key=settings.together_api_key,
        base_url=settings.together_base_url,
    )

    user_msg = query if query else f"Provide a comprehensive financial analysis of {ticker.upper()}."
    user_msg = f"[Company: {ticker.upper()}] {user_msg}"

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Include conversation history for follow-up context
    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_msg})

    await _log("planning", f"Starting agentic analysis for {ticker.upper()}")

    for _ in range(MAX_ITERATIONS):
        try:
            response = await client.chat.completions.create(
                model=settings.together_model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
        except Exception as e:
            logger.error("LLM API error: %s", e)
            await _log("error", f"LLM API error: {e}")
            break

        choice = response.choices[0]
        assistant_msg = choice.message

        # ── Tool-calling turn ──
        if assistant_msg.tool_calls:
            # Build serialisable assistant message for the conversation
            tc_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_msg.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": tc_dicts,
            })

            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                await _log("tool_call", f"Calling {fn_name}", {"args": fn_args})

                result = await execute_tool(fn_name, fn_args, ticker=ticker)
                result_str = json.dumps(result, default=str)
                # Truncate large payloads to stay within LLM context
                truncated = result_str[:settings.agent_tool_result_max_chars]

                await _log("tool_result", f"Result from {fn_name}", {"preview": truncated[:500]})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": truncated,
                })
        else:
            # ── Final response ──
            final = assistant_msg.content or "Analysis complete."
            await _log("complete", "Analysis finished.")
            return AgentResponse(response=final, traces=traces)

    # Max iterations reached
    await _log("complete", "Analysis finished (max iterations).")
    return AgentResponse(
        response="Analysis complete. The agent reached the maximum number of iterations.",
        traces=traces,
    )


async def _run_direct(
    ticker: str,
    query: str | None,
    _log: Callable,
) -> AgentResponse:
    """Deterministic fallback that calls every tool without an LLM."""
    await _log("planning", f"Running direct analysis for {ticker.upper()} (no LLM key configured)")

    profile = get_profile(ticker)
    await _log("tool_result", f"Fetched profile: {profile.name}", {"name": profile.name, "sector": profile.sector})

    financials = get_financials(ticker)
    n_years = len(financials.income_statements)
    await _log("tool_result", f"Fetched {n_years} years of financials")

    key_stats = get_key_stats(ticker)
    await _log("tool_result", f"Current price: ${key_stats.current_price or 0:.2f}")

    macro = get_macro_snapshot()
    await _log("tool_result", "Fetched macro data", macro.model_dump())

    starting_rev = (
        financials.income_statements[-1].revenue
        if financials.income_statements
        else settings.default_starting_revenue
    )
    assumptions = derive_assumptions(financials, key_stats=key_stats, macro=macro)
    base_fcst = forecast(assumptions["base"], starting_revenue=starting_rev)
    net_debt = financials.balance_sheets[-1].net_debt if financials.balance_sheets else 0
    shares = key_stats.shares_outstanding or settings.default_shares_outstanding
    dcf = dcf_valuation(
        base_fcst,
        assumptions["base"].wacc,
        assumptions["base"].terminal_growth_rate,
        current_price=key_stats.current_price or settings.default_current_price,
        net_debt=net_debt,
        shares_outstanding=shares,
        exit_ev_ebitda=key_stats.ev_to_ebitda,
    )
    await _log("tool_result", f"DCF implied price: ${dcf.implied_share_price:.2f}")

    await _log("complete", "Analysis complete (direct mode)")

    latest_stmt = financials.income_statements[-1] if financials.income_statements else None
    response_text = (
        f"Analysis for {profile.name} ({ticker.upper()}) completed.\n\n"
        f"Current Price: ${key_stats.current_price or 0:.2f}\n"
        f"Implied Price (Base DCF): ${dcf.implied_share_price:.2f}\n"
        f"Upside/Downside: {dcf.upside_pct:.1%}\n"
    )
    if latest_stmt:
        response_text += (
            f"\nLatest Revenue: ${latest_stmt.revenue:,.0f}\n"
            f"Gross Margin: {latest_stmt.gross_margin:.1%}\n"
            f"Net Margin: {latest_stmt.net_margin:.1%}\n"
        )
    if query:
        response_text += f"\nRegarding '{query}': Enable Together AI API key for AI-powered answers."

    return AgentResponse(response=response_text, traces=[])
