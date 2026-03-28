from pydantic import BaseModel, computed_field
from typing import Optional
from enum import Enum


# ──────────────────────────────────────────────
# Company Profile
# ───────────────────────��──────────────────────

class CompanyProfile(BaseModel):
    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    market_cap: Optional[float] = None
    employees: Optional[int] = None
    website: Optional[str] = None
    headquarters: Optional[str] = None
    ceo: Optional[str] = None
    logo_url: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = "USD"


# ──────────────────────────────────────────────
# Financial Statements
# ──────────────────────────────────────────────

class IncomeStatement(BaseModel):
    fiscal_year: int
    revenue: float
    cost_of_revenue: float = 0
    gross_profit: float = 0
    operating_expenses: float = 0
    operating_income: float = 0
    ebitda: float = 0
    net_income: float = 0
    eps: Optional[float] = None

    @computed_field
    @property
    def gross_margin(self) -> float:
        return round(self.gross_profit / self.revenue, 4) if self.revenue else 0

    @computed_field
    @property
    def operating_margin(self) -> float:
        return round(self.operating_income / self.revenue, 4) if self.revenue else 0

    @computed_field
    @property
    def net_margin(self) -> float:
        return round(self.net_income / self.revenue, 4) if self.revenue else 0


class BalanceSheet(BaseModel):
    fiscal_year: int
    total_assets: float = 0
    total_liabilities: float = 0
    total_equity: float = 0
    cash_and_equivalents: float = 0
    total_debt: float = 0
    current_assets: float = 0
    current_liabilities: float = 0

    @computed_field
    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash_and_equivalents

    @computed_field
    @property
    def debt_to_equity(self) -> float:
        return round(self.total_debt / self.total_equity, 4) if self.total_equity else 0

    @computed_field
    @property
    def current_ratio(self) -> float:
        return round(self.current_assets / self.current_liabilities, 4) if self.current_liabilities else 0


class CashFlowStatement(BaseModel):
    fiscal_year: int
    operating_cash_flow: float = 0
    capex: float = 0
    dividends_paid: float = 0
    share_buybacks: float = 0

    @computed_field
    @property
    def free_cash_flow(self) -> float:
        return self.operating_cash_flow - abs(self.capex)


class HistoricalFinancials(BaseModel):
    ticker: str
    income_statements: list[IncomeStatement] = []
    balance_sheets: list[BalanceSheet] = []
    cash_flows: list[CashFlowStatement] = []


# ──────────────────────────────────────────────
# Price Data
# ─────────────────��────────────────────────────

class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class KeyStats(BaseModel):
    ticker: str
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    price_to_book: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    current_price: Optional[float] = None
    shares_outstanding: Optional[float] = None


# ──────────────────────────────────────────────
# Macro Data
# ──────────────────────────────────────────────

class MacroData(BaseModel):
    treasury_10y: Optional[float] = None       # Risk-free rate
    fed_funds_rate: Optional[float] = None
    gdp_growth: Optional[float] = None
    cpi_inflation: Optional[float] = None


# ──────────────────────────────────────────────
# Financial Model
# ──────────────────────────────────────────────

class Scenario(str, Enum):
    BASE = "base"
    UPSIDE = "upside"
    DOWNSIDE = "downside"


class Assumptions(BaseModel):
    revenue_growth_rate: float = 0.08
    gross_margin: float = 0.45
    opex_as_pct_revenue: float = 0.25
    capex_as_pct_revenue: float = 0.05
    da_as_pct_revenue: float = 0.03
    tax_rate: float = 0.21
    wacc: float = 0.10
    terminal_growth_rate: float = 0.025


class ForecastYear(BaseModel):
    year: int
    revenue: float
    gross_profit: float
    operating_income: float
    ebitda: float
    net_income: float
    free_cash_flow: float

    @computed_field
    @property
    def gross_margin(self) -> float:
        return round(self.gross_profit / self.revenue, 4) if self.revenue else 0

    @computed_field
    @property
    def net_margin(self) -> float:
        return round(self.net_income / self.revenue, 4) if self.revenue else 0


class ScenarioResult(BaseModel):
    scenario: Scenario
    assumptions: Assumptions
    forecast: list[ForecastYear]


class DCFResult(BaseModel):
    enterprise_value: float
    equity_value: float
    implied_share_price: float
    current_price: float

    @computed_field
    @property
    def upside_pct(self) -> float:
        return round((self.implied_share_price - self.current_price) / self.current_price, 4) if self.current_price else 0


class SensitivityCell(BaseModel):
    param_x_value: float
    param_y_value: float
    output_value: float


class SensitivityResult(BaseModel):
    param_x_name: str
    param_y_name: str
    output_name: str
    cells: list[SensitivityCell]


# ──────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────

class AgentTrace(BaseModel):
    timestamp: str
    step: int
    type: str        # planning, tool_call, tool_result, complete, error
    detail: str
    data: Optional[dict] = None


class AgentResponse(BaseModel):
    response: str
    traces: list[AgentTrace]


# ──────────────────────────────────────────────
# Advisory
# ──────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    SPECULATIVE = "SPECULATIVE"


class AdvisoryRecommendation(BaseModel):
    title: str
    rationale: str
    confidence: ConfidenceLevel
    sources: list[str] = []


class AdvisoryReport(BaseModel):
    ticker: str
    capital_structure_summary: str
    funding_options: list[AdvisoryRecommendation]
    strategic_options: list[AdvisoryRecommendation]
    disclaimer: str = "This is AI-generated analysis for educational purposes only. Not investment advice."


# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────

class ReportType(str, Enum):
    INFO_MEMO = "information_memorandum"
    SHAREHOLDER_REPORT = "shareholder_report"
    EQUITY_NOTE = "equity_research_note"
    PRESENTATION = "corporate_presentation"


class ReportSection(BaseModel):
    title: str
    content: str


class GeneratedReport(BaseModel):
    report_type: ReportType
    ticker: str
    company_name: str
    generated_at: str
    sections: list[ReportSection]
    disclaimer: str = "This is AI-generated analysis for educational purposes only. Not investment advice."


# ──────────────────────────────────────────────
# Full Company Data (combined response)
# ──────────────────────────────────────────────

class FullCompanyData(BaseModel):
    profile: CompanyProfile
    financials: HistoricalFinancials
    key_stats: KeyStats
    price_history: list[PricePoint] = []
    macro: MacroData
    scenarios: list[ScenarioResult] = []