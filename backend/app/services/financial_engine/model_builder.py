from __future__ import annotations

from datetime import date

from app.models.company import (
    Assumptions,
    DCFResult,
    ForecastYear,
    HistoricalFinancials,
    KeyStats,
    MacroData,
)


def compute_wacc(
    beta: float | None = None,
    risk_free_rate: float | None = None,
    market_cap: float | None = None,
    total_debt: float | None = None,
    tax_rate: float = 0.21,
    equity_risk_premium: float | None = None,
) -> float:
    """CAPM-based WACC: E/(E+D) × Re + D/(E+D) × Rd × (1−T).

    Re = risk-free + adjusted_beta × equity risk premium (CAPM).
    Beta is Bloomberg-adjusted (1/3 + 2/3 × raw) to regress toward market.
    Rd approximated as risk-free + credit spread.
    """
    from app.config import settings as _s
    erp = equity_risk_premium if equity_risk_premium is not None else _s.equity_risk_premium
    rf = risk_free_rate if risk_free_rate is not None else _s.fallback_risk_free_rate
    raw_beta = beta if beta is not None else 1.0
    # Bloomberg-adjusted beta: regress toward market beta of 1.0
    b = 1 / 3 + 2 / 3 * raw_beta

    cost_of_equity = rf + b * erp

    if market_cap and total_debt and total_debt > 0:
        cost_of_debt = rf + _s.credit_spread
        e, d = market_cap, total_debt
        v = e + d
        wacc = (e / v) * cost_of_equity + (d / v) * cost_of_debt * (1 - tax_rate)
    else:
        wacc = cost_of_equity

    return round(min(max(wacc, _s.wacc_floor), _s.wacc_cap), 4)


def derive_assumptions(
    historical_data: HistoricalFinancials,
    key_stats: KeyStats | None = None,
    macro: MacroData | None = None,
) -> dict[str, Assumptions]:
    """Derive Base / Upside / Downside assumptions from real historical financials."""
    if not historical_data.income_statements:
        base = Assumptions()
    else:
        stmts = historical_data.income_statements
        revenues = [s.revenue for s in stmts if s.revenue > 0]

        # Revenue growth: use average of YoY growth rates (more robust than
        # endpoint CAGR which can be skewed by one bad year).
        from app.config import settings as _cfg
        growth = 0.08
        if len(revenues) >= 2:
            yoy_rates = [
                (revenues[i] / revenues[i - 1]) - 1
                for i in range(1, len(revenues))
                if revenues[i - 1] > 0
            ]
            if yoy_rates:
                avg_growth = sum(yoy_rates) / len(yoy_rates)
                growth = max(min(avg_growth, 0.30), _cfg.min_growth_rate)

        # Gross margin: average of most recent 3 years
        margins = [s.gross_margin for s in stmts[-3:] if s.gross_margin > 0]
        avg_gm = sum(margins) / len(margins) if margins else 0.45

        # OpEx as pct of revenue: average of most recent 3 years
        opex_pcts = [
            s.operating_expenses / s.revenue
            for s in stmts[-3:]
            if s.revenue > 0 and s.operating_expenses > 0
        ]
        avg_opex = sum(opex_pcts) / len(opex_pcts) if opex_pcts else 0.25

        # CapEx as pct of revenue (from cash flows)
        capex_pct = 0.05
        if historical_data.cash_flows and revenues:
            recent_cf = historical_data.cash_flows[-3:]
            recent_rev = revenues[-len(recent_cf):]
            capex_pcts = [
                abs(cf.capex) / rev
                for cf, rev in zip(recent_cf, recent_rev)
                if rev > 0
            ]
            if capex_pcts:
                capex_pct = sum(capex_pcts) / len(capex_pcts)

        # D&A as pct of revenue: derive from (EBITDA − operating income) / revenue
        da_pct = 0.03
        da_ratios = [
            (s.ebitda - s.operating_income) / s.revenue
            for s in stmts[-3:]
            if s.revenue > 0 and s.ebitda > s.operating_income
        ]
        if da_ratios:
            da_pct = max(sum(da_ratios) / len(da_ratios), 0.005)

        # WACC from CAPM: beta × equity-risk-premium + risk-free rate
        wacc = 0.10
        if key_stats or macro:
            beta = getattr(key_stats, "beta", None) if key_stats else None
            rf = getattr(macro, "treasury_10y", None) if macro else None
            mcap = getattr(key_stats, "market_cap", None) if key_stats else None
            # Market cap might be on profile; fall back to price × shares
            if mcap is None and key_stats:
                price = key_stats.current_price
                shares = key_stats.shares_outstanding
                if price and shares:
                    mcap = price * shares
            total_debt_val = None
            if historical_data.balance_sheets:
                total_debt_val = historical_data.balance_sheets[-1].total_debt or None
            wacc = compute_wacc(
                beta=beta,
                risk_free_rate=rf,
                market_cap=mcap,
                total_debt=total_debt_val,
            )

        base = Assumptions(
            revenue_growth_rate=round(growth, 4),
            gross_margin=round(avg_gm, 4),
            opex_as_pct_revenue=round(avg_opex, 4),
            capex_as_pct_revenue=round(min(capex_pct, 0.20), 4),
            da_as_pct_revenue=round(min(da_pct, 0.15), 4),
            wacc=wacc,
        )

    return {
        "base": base,
        "upside": Assumptions(
            revenue_growth_rate=round(min(base.revenue_growth_rate + 0.05, 0.30), 4),
            gross_margin=round(min(base.gross_margin + 0.05, 0.95), 4),
            opex_as_pct_revenue=max(round(base.opex_as_pct_revenue - 0.03, 4), 0.03),
            capex_as_pct_revenue=max(round(base.capex_as_pct_revenue - 0.02, 4), 0.01),
            da_as_pct_revenue=base.da_as_pct_revenue,
            tax_rate=base.tax_rate,
            wacc=max(round(base.wacc - 0.02, 4), 0.04),
            terminal_growth_rate=round(base.terminal_growth_rate + 0.005, 4),
        ),
        "downside": Assumptions(
            revenue_growth_rate=max(round(base.revenue_growth_rate - 0.03, 4), 0.0),
            gross_margin=max(round(base.gross_margin - 0.03, 4), 0.05),
            opex_as_pct_revenue=round(base.opex_as_pct_revenue + 0.02, 4),
            capex_as_pct_revenue=round(base.capex_as_pct_revenue + 0.01, 4),
            da_as_pct_revenue=base.da_as_pct_revenue,
            tax_rate=base.tax_rate,
            wacc=round(min(base.wacc + 0.01, 0.15), 4),
            terminal_growth_rate=max(round(base.terminal_growth_rate - 0.005, 4), 0.0),
        ),
    }


def forecast(
    assumptions: Assumptions,
    years: int = 5,
    starting_revenue: float | None = None,
) -> list[ForecastYear]:
    from app.config import settings as _s
    out: list[ForecastYear] = []
    revenue = float(starting_revenue if starting_revenue is not None else _s.default_starting_revenue)
    current_year = date.today().year

    for idx in range(years):
        revenue *= 1 + assumptions.revenue_growth_rate
        gross_profit = revenue * assumptions.gross_margin
        opex = revenue * assumptions.opex_as_pct_revenue
        operating_income = gross_profit - opex
        da = revenue * assumptions.da_as_pct_revenue
        ebitda = operating_income + da
        pretax_income = operating_income
        net_income = pretax_income * (1 - assumptions.tax_rate)
        capex = revenue * assumptions.capex_as_pct_revenue
        free_cash_flow = net_income + da - capex

        out.append(
            ForecastYear(
                year=current_year + idx + 1,
                revenue=round(revenue, 2),
                gross_profit=round(gross_profit, 2),
                operating_income=round(operating_income, 2),
                ebitda=round(ebitda, 2),
                net_income=round(net_income, 2),
                free_cash_flow=round(free_cash_flow, 2),
            )
        )
    return out


def dcf_valuation(
	forecast: list[ForecastYear],
	wacc: float,
	terminal_growth: float,
	current_price: float | None = None,
	net_debt: float = 0.0,
	shares_outstanding: float | None = None,
	exit_ev_ebitda: float | None = None,
) -> DCFResult:
	"""Compute DCF valuation with blended terminal value.

	Terminal value uses a 50/50 blend of perpetuity growth model and
	EV/EBITDA exit multiple (capped at 20×) when exit_ev_ebitda is provided.
	This produces more realistic valuations for growth companies.
	"""
	from app.config import settings as _s
	current_price = current_price if current_price is not None else _s.default_current_price
	shares_outstanding = shares_outstanding if shares_outstanding is not None else _s.default_shares_outstanding
	if not forecast:
		return DCFResult(
			enterprise_value=0,
			equity_value=0,
			implied_share_price=0,
			current_price=current_price,
		)

	safe_wacc = max(wacc, terminal_growth + 0.005)
	pv = 0.0
	for idx, year in enumerate(forecast, start=1):
		pv += year.free_cash_flow / ((1 + safe_wacc) ** idx)

	# Terminal value: perpetuity growth model
	terminal_fcf = forecast[-1].free_cash_flow * (1 + terminal_growth)
	tv_perpetuity = terminal_fcf / (safe_wacc - terminal_growth)

	# Blend with EV/EBITDA exit multiple when available
	terminal_ebitda = forecast[-1].ebitda
	if exit_ev_ebitda and terminal_ebitda > 0:
		capped_multiple = min(exit_ev_ebitda, 20.0)
		tv_exit = terminal_ebitda * capped_multiple
		terminal_value = (tv_perpetuity + tv_exit) / 2
	else:
		terminal_value = tv_perpetuity

	terminal_pv = terminal_value / ((1 + safe_wacc) ** len(forecast))

	enterprise_value = pv + terminal_pv
	equity_value = enterprise_value - net_debt
	implied_share_price = equity_value / shares_outstanding

	return DCFResult(
		enterprise_value=round(enterprise_value, 2),
		equity_value=round(equity_value, 2),
		implied_share_price=round(implied_share_price, 2),
		current_price=round(current_price, 2),
	)
