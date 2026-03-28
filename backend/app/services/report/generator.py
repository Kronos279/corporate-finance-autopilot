from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from openai import AsyncOpenAI

from app.config import settings
from app.models.company import GeneratedReport, ReportSection, ReportType
from app.services.ingestion.fred import get_macro_snapshot
from app.services.ingestion.yahoo_finance import get_financials, get_key_stats, get_profile

logger = logging.getLogger(__name__)

# ── Prompt templates per report type ──

REPORT_PROMPTS: dict[ReportType, str] = {
    ReportType.INFO_MEMO: (
        "Generate a comprehensive Information Memorandum with these sections:\n"
        "1. Executive Summary\n2. Company Overview\n3. Industry & Market Analysis\n"
        "4. Financial Analysis\n5. Growth Strategy\n6. Risk Factors\n7. Valuation Summary"
    ),
    ReportType.SHAREHOLDER_REPORT: (
        "Generate a Monthly Shareholder Report with these sections:\n"
        "1. Chairman's Letter\n2. Financial Performance\n3. Operational Highlights\n"
        "4. Market & Industry Update\n5. Capital Allocation\n6. Outlook"
    ),
    ReportType.EQUITY_NOTE: (
        "Generate a sell-side Equity Research Note with these sections:\n"
        "1. Investment Thesis\n2. Recent Developments\n3. Financial Model Summary\n"
        "4. Valuation Analysis\n5. Risk/Reward Assessment\n6. Recommendation"
    ),
    ReportType.PRESENTATION: (
        "Generate a Corporate Presentation outline with these sections:\n"
        "1. Company Snapshot\n2. Business Model\n3. Market Opportunity\n"
        "4. Competitive Advantages\n5. Financial Track Record\n6. Strategic Roadmap\n"
        "7. Investment Highlights"
    ),
}


def _build_data_context(ticker: str) -> str:
    """Assemble a text summary of all available company data for the LLM."""
    profile = get_profile(ticker)
    financials = get_financials(ticker)
    stats = get_key_stats(ticker)
    macro = get_macro_snapshot()

    parts: list[str] = [
        f"Company: {profile.name} ({profile.ticker})",
        f"Sector: {profile.sector} | Industry: {profile.industry}",
    ]
    if profile.market_cap:
        parts.append(f"Market Cap: ${profile.market_cap:,.0f}")
    if profile.employees:
        parts.append(f"Employees: {profile.employees:,}")
    if profile.description:
        parts.append(f"Description: {profile.description[:600]}")

    parts.append("\n=== KEY METRICS ===")
    if stats.current_price:
        parts.append(f"Current Price: ${stats.current_price:.2f}")
    if stats.pe_ratio:
        parts.append(f"P/E: {stats.pe_ratio:.1f}")
    if stats.ev_to_ebitda:
        parts.append(f"EV/EBITDA: {stats.ev_to_ebitda:.1f}")
    if stats.beta:
        parts.append(f"Beta: {stats.beta:.2f}")
    if stats.dividend_yield:
        parts.append(f"Dividend Yield: {stats.dividend_yield:.2%}")

    parts.append("\n=== RECENT FINANCIALS ===")
    for stmt in financials.income_statements[-3:]:
        parts.append(
            f"FY{stmt.fiscal_year}: Rev ${stmt.revenue:,.0f} | "
            f"GM {stmt.gross_margin:.1%} | NI ${stmt.net_income:,.0f} | "
            f"NM {stmt.net_margin:.1%}"
        )

    if financials.balance_sheets:
        bs = financials.balance_sheets[-1]
        parts.extend([
            "\n=== LATEST BALANCE SHEET ===",
            f"Total Assets: ${bs.total_assets:,.0f}",
            f"Total Debt: ${bs.total_debt:,.0f}",
            f"Cash: ${bs.cash_and_equivalents:,.0f}",
            f"D/E: {bs.debt_to_equity:.2f}",
        ])

    parts.append("\n=== MACRO ===")
    if macro.treasury_10y is not None:
        parts.append(f"10Y Treasury: {macro.treasury_10y:.2%}")
    if macro.fed_funds_rate is not None:
        parts.append(f"Fed Funds: {macro.fed_funds_rate:.2%}")
    if macro.gdp_growth is not None:
        parts.append(f"GDP Growth: {macro.gdp_growth:.2%}")
    if macro.cpi_inflation is not None:
        parts.append(f"CPI: {macro.cpi_inflation:.2%}")

    return "\n".join(parts)


async def _generate_with_llm(
    ticker: str, report_type: ReportType, data_context: str,
) -> list[ReportSection]:
    """Call Together AI to generate report sections using real company data."""
    prompt_instructions = REPORT_PROMPTS.get(report_type, REPORT_PROMPTS[ReportType.INFO_MEMO])

    client = AsyncOpenAI(
        api_key=settings.together_api_key,
        base_url=settings.together_base_url,
    )

    response = await client.chat.completions.create(
        model=settings.together_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior investment banking analyst writing professional "
                    "corporate finance documents. Use a formal, analytical tone. "
                    "Reference specific numbers from the data provided. "
                    'Return ONLY a JSON array of objects: [{"title": "...", "content": "..."}]. '
                    "Each section should be 2-4 paragraphs."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt_instructions}\n\n=== COMPANY DATA ===\n{data_context}\n\n"
                    'Respond with ONLY the JSON array of {{"title": "...", "content": "..."}} objects.'
                ),
            },
        ],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    content = response.choices[0].message.content or "[]"

    try:
        # Strip code fences if present
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]

        sections_data = json.loads(content)
        return [ReportSection(title=s["title"], content=s["content"]) for s in sections_data]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Failed to parse LLM report JSON: %s", e)
        return [ReportSection(title="Report", content=content)]


def _fallback_sections(ticker: str, report_type: ReportType) -> list[ReportSection]:
    """Deterministic fallback when LLM is unavailable."""
    profile = get_profile(ticker)
    financials = get_financials(ticker)
    key_stats = get_key_stats(ticker)

    sections = [
        ReportSection(
            title="Executive Summary",
            content=(
                f"{profile.name} ({ticker.upper()}) operates in the {profile.sector or 'N/A'} sector. "
                f"Market capitalisation is ${profile.market_cap or 0:,.0f}. "
                f"Current price: ${key_stats.current_price or 0:.2f}."
            ),
        ),
    ]
    if financials.income_statements:
        latest = financials.income_statements[-1]
        sections.append(ReportSection(
            title="Financial Highlights",
            content=(
                f"FY{latest.fiscal_year} revenue was ${latest.revenue:,.0f} "
                f"with a gross margin of {latest.gross_margin:.1%} "
                f"and net margin of {latest.net_margin:.1%}."
            ),
        ))
    sections.append(ReportSection(
        title="Outlook",
        content="Enable Together AI API key for AI-generated forward-looking analysis.",
    ))
    return sections


async def generate_report(ticker: str, report_type: ReportType) -> GeneratedReport:
    """Generate a report – uses LLM if key is configured, else falls back to data-only."""
    profile = get_profile(ticker)

    if settings.together_api_key:
        data_context = _build_data_context(ticker)
        try:
            sections = await _generate_with_llm(ticker, report_type, data_context)
        except Exception as e:
            logger.error("LLM report generation failed: %s", e)
            sections = _fallback_sections(ticker, report_type)
    else:
        sections = _fallback_sections(ticker, report_type)

    return GeneratedReport(
        report_type=report_type,
        ticker=ticker.upper(),
        company_name=profile.name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        sections=sections,
    )


def create_pdf(report: GeneratedReport) -> bytes:
    """Render a GeneratedReport to PDF bytes using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, report.company_name, ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    title_text = report.report_type.value.replace("_", " ").title()
    pdf.cell(0, 8, title_text, ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated: {report.generated_at}", ln=True, align="C")
    pdf.ln(8)

    # Sections
    for section in report.sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, section.title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        # encode to latin-1 with replace to avoid fpdf encoding errors
        safe_content = section.content.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_content)
        pdf.ln(4)

    # Disclaimer
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    safe_disclaimer = report.disclaimer.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 4, safe_disclaimer)

    return bytes(pdf.output())
