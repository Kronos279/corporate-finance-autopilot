from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings
from app.models.company import Assumptions, Scenario, ScenarioResult
from app.services.financial_engine.model_builder import dcf_valuation, derive_assumptions, forecast
from app.services.financial_engine.sensitivity import sensitivity_table
from app.services.ingestion.fred import get_macro_snapshot
from app.services.ingestion.yahoo_finance import get_financials, get_key_stats

router = APIRouter()


class ForecastRequest(BaseModel):
    assumptions: Assumptions = Field(default_factory=Assumptions)
    years: int = 5


class SensitivityRequest(BaseModel):
    assumptions: Assumptions = Field(default_factory=Assumptions)
    param_x: str = "revenue_growth_rate"
    param_y: str = "wacc"
    range_x: list[float] = Field(default_factory=lambda: [0.04, 0.06, 0.08, 0.10])
    range_y: list[float] = Field(default_factory=lambda: [0.08, 0.10, 0.12])


class DCFRequest(BaseModel):
    assumptions: Assumptions = Field(default_factory=Assumptions)
    years: int = 5


def _starting_revenue(ticker: str) -> float:
    financials = get_financials(ticker)
    if financials.income_statements:
        return financials.income_statements[-1].revenue
    return settings.default_starting_revenue


@router.post("/{ticker}/forecast")
async def run_forecast(ticker: str, request: ForecastRequest):
    return {
        "ticker": ticker.upper(),
        "assumptions": request.assumptions,
        "forecast": forecast(
            request.assumptions,
            years=request.years,
            starting_revenue=_starting_revenue(ticker),
        ),
    }


@router.get("/{ticker}/scenarios")
async def get_scenarios(ticker: str):
    financials = get_financials(ticker)
    key_stats = get_key_stats(ticker)
    macro = get_macro_snapshot()
    starting_rev = (
        financials.income_statements[-1].revenue
        if financials.income_statements
        else settings.default_starting_revenue
    )
    assumptions = derive_assumptions(financials, key_stats=key_stats, macro=macro)
    return [
        ScenarioResult(
            scenario=Scenario(name),
            assumptions=assumption_set,
            forecast=forecast(assumption_set, starting_revenue=starting_rev),
        )
        for name, assumption_set in assumptions.items()
    ]


@router.post("/{ticker}/sensitivity")
async def run_sensitivity(ticker: str, request: SensitivityRequest):
    financials = get_financials(ticker)
    key_stats = get_key_stats(ticker)
    starting_rev = (
        financials.income_statements[-1].revenue
        if financials.income_statements
        else settings.default_starting_revenue
    )
    net_debt = financials.balance_sheets[-1].net_debt if financials.balance_sheets else 0.0
    result = sensitivity_table(
        request.assumptions,
        param_x=request.param_x,
        param_y=request.param_y,
        range_x=request.range_x,
        range_y=request.range_y,
        starting_revenue=starting_rev,
        net_debt=net_debt,
        shares_outstanding=key_stats.shares_outstanding or settings.default_shares_outstanding,
        current_price=key_stats.current_price or settings.default_current_price,
    )
    return {"ticker": ticker.upper(), "result": result}


@router.post("/{ticker}/dcf")
async def run_dcf(ticker: str, request: DCFRequest):
    financials = get_financials(ticker)
    key_stats = get_key_stats(ticker)
    starting_rev = (
        financials.income_statements[-1].revenue
        if financials.income_statements
        else settings.default_starting_revenue
    )
    fcst = forecast(request.assumptions, years=request.years, starting_revenue=starting_rev)

    net_debt = 0.0
    if financials.balance_sheets:
        net_debt = financials.balance_sheets[-1].net_debt

    value = dcf_valuation(
        fcst,
        request.assumptions.wacc,
        request.assumptions.terminal_growth_rate,
        current_price=key_stats.current_price or settings.default_current_price,
        net_debt=net_debt,
        shares_outstanding=key_stats.shares_outstanding or settings.default_shares_outstanding,
        exit_ev_ebitda=key_stats.ev_to_ebitda,
    )
    return {"ticker": ticker.upper(), "dcf": value}
