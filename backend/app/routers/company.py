from fastapi import APIRouter

from app.config import settings
from app.models.company import FullCompanyData, Scenario, ScenarioResult
from app.services.financial_engine.model_builder import derive_assumptions, forecast
from app.services.ingestion.fred import get_macro_snapshot
from app.services.ingestion.yahoo_finance import (
    get_financials,
    get_key_stats,
    get_price_history,
    get_profile,
)
from app.services.rag import ingest_company_data, search as rag_search

router = APIRouter()


@router.get("/{ticker}/profile")
async def company_profile(ticker: str):
    return get_profile(ticker)


@router.get("/{ticker}/financials")
async def company_financials(ticker: str):
    return get_financials(ticker)


@router.get("/{ticker}/price-history")
async def company_price_history(ticker: str):
    return get_price_history(ticker)


@router.get("/{ticker}/macro")
async def company_macro(ticker: str):
    return get_macro_snapshot()


@router.get("/{ticker}/full")
async def company_full(ticker: str):
    financials = get_financials(ticker)
    key_stats = get_key_stats(ticker)

    starting_rev = (
        financials.income_statements[-1].revenue
        if financials.income_statements
        else settings.default_starting_revenue
    )
    macro = get_macro_snapshot()
    scenario_inputs = derive_assumptions(financials, key_stats=key_stats, macro=macro)
    scenarios = [
        ScenarioResult(
            scenario=Scenario(name),
            assumptions=assumptions,
            forecast=forecast(assumptions, starting_revenue=starting_rev),
        )
        for name, assumptions in scenario_inputs.items()
    ]

    return FullCompanyData(
        profile=get_profile(ticker),
        financials=financials,
        key_stats=key_stats,
        price_history=get_price_history(ticker),
        macro=get_macro_snapshot(),
        scenarios=scenarios,
    )


@router.post("/{ticker}/ingest")
async def ingest_company(ticker: str):
    """Ingest company data (profile, financials, SEC filings) into the RAG vector store."""
    result = await ingest_company_data(ticker)
    return result


@router.get("/{ticker}/search")
async def search_documents(ticker: str, q: str, top_k: int = 5):
    """Semantic search over ingested documents for a company."""
    results = await rag_search(q, ticker=ticker, top_k=top_k)
    return {"ticker": ticker.upper(), "query": q, "results": results}
