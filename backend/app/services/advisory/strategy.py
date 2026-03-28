from __future__ import annotations

from app.models.company import (
    AdvisoryRecommendation,
    ConfidenceLevel,
    HistoricalFinancials,
    MacroData,
)


def analyze_capital_structure(financials: HistoricalFinancials) -> str:
    if not financials.balance_sheets:
        return "Insufficient balance sheet data to analyze capital structure."

    latest = financials.balance_sheets[-1]
    debt_ratio = latest.total_debt / latest.total_assets if latest.total_assets else 0
    leverage = "conservative" if debt_ratio < 0.25 else "moderate" if debt_ratio < 0.45 else "elevated"

    parts = [
        f"Debt-to-assets is {debt_ratio:.1%} ({leverage} leverage).",
        f"Debt-to-equity is {latest.debt_to_equity:.2f}.",
        f"Current ratio is {latest.current_ratio:.2f}.",
    ]
    if latest.cash_and_equivalents:
        parts.append(f"Cash reserves of ${latest.cash_and_equivalents:,.0f}.")
    if latest.net_debt < 0:
        parts.append("The company is in a net cash position.")
    return " ".join(parts)


def suggest_funding_options(
    financials: HistoricalFinancials,
    macro: MacroData,
) -> list[AdvisoryRecommendation]:
    recs: list[AdvisoryRecommendation] = []

    if financials.balance_sheets:
        bs = financials.balance_sheets[-1]
        debt_ratio = bs.total_debt / bs.total_assets if bs.total_assets else 0

        if debt_ratio > 0.45:
            recs.append(AdvisoryRecommendation(
                title="Prioritize debt reduction",
                rationale=(
                    f"Debt-to-assets of {debt_ratio:.1%} is above the 45% threshold. "
                    "Consider directing FCF toward deleveraging to lower interest burden."
                ),
                confidence=ConfidenceLevel.HIGH,
                sources=["Balance sheet analysis"],
            ))

        if bs.cash_and_equivalents > bs.total_debt * 0.5 and bs.total_debt > 0:
            recs.append(AdvisoryRecommendation(
                title="Optimize excess cash deployment",
                rationale=(
                    f"Cash of ${bs.cash_and_equivalents:,.0f} represents over half of total "
                    f"debt (${bs.total_debt:,.0f}). Evaluate shareholder returns vs. strategic "
                    "reinvestment."
                ),
                confidence=ConfidenceLevel.HIGH,
                sources=["Balance sheet analysis"],
            ))

    if macro.fed_funds_rate is not None:
        recs.append(AdvisoryRecommendation(
            title="Lock in fixed-rate debt",
            rationale=(
                f"With the fed funds rate at {macro.fed_funds_rate:.2%}, consider issuing "
                "long-dated fixed-rate bonds to secure predictable interest costs."
            ),
            confidence=ConfidenceLevel.MEDIUM,
            sources=["FRED macro data"],
        ))

    if financials.cash_flows:
        latest_cf = financials.cash_flows[-1]
        if latest_cf.free_cash_flow > 0:
            recs.append(AdvisoryRecommendation(
                title="Selective share buyback program",
                rationale=(
                    f"Positive FCF of ${latest_cf.free_cash_flow:,.0f} supports a disciplined "
                    "buyback program, particularly opportunistic when shares trade below "
                    "estimated intrinsic value."
                ),
                confidence=ConfidenceLevel.MEDIUM,
                sources=["Cash flow analysis", "Valuation model"],
            ))

    if not recs:
        recs.append(AdvisoryRecommendation(
            title="Maintain current capital structure",
            rationale="Current leverage and cash position appear appropriate.",
            confidence=ConfidenceLevel.MEDIUM,
            sources=["Financial statement analysis"],
        ))

    return recs


def suggest_strategic_options(
    financials: HistoricalFinancials | None = None,
) -> list[AdvisoryRecommendation]:
    recs: list[AdvisoryRecommendation] = []

    if financials and financials.income_statements:
        latest = financials.income_statements[-1]
        if latest.gross_margin > 0.50:
            recs.append(AdvisoryRecommendation(
                title="Invest in R&D for margin expansion",
                rationale=(
                    f"Strong gross margins of {latest.gross_margin:.1%} indicate pricing "
                    "power. Increased R&D could drive further differentiation."
                ),
                confidence=ConfidenceLevel.HIGH,
                sources=["Income statement analysis"],
            ))

    recs.extend([
        AdvisoryRecommendation(
            title="Bolt-on acquisition screening",
            rationale=(
                "Prioritize adjacent capabilities with clear cross-sell economics "
                "and low integration risk."
            ),
            confidence=ConfidenceLevel.MEDIUM,
            sources=["Peer strategy analysis", "M&A market conditions"],
        ),
        AdvisoryRecommendation(
            title="International market expansion",
            rationale=(
                "Pilot one high-growth geography with local partnerships "
                "before committing to full-scale rollout."
            ),
            confidence=ConfidenceLevel.SPECULATIVE,
            sources=["Management guidance", "Market sizing"],
        ),
    ])
    return recs
