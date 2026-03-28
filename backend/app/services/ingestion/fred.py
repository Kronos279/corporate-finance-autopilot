from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.models.company import MacroData
logger = logging.getLogger(__name__)

FRED_BASE = settings.fred_base_url


def _fetch_fred(series_id: str, units: str | None = None) -> float | None:
    """Low-level FRED fetch (not cached – callers are cached individually)."""
    if not settings.fred_api_key:
        logger.warning("FRED API key not configured, skipping %s", series_id)
        return None
    try:
        params: dict = {
            "series_id": series_id,
            "api_key": settings.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        if units:
            params["units"] = units
        resp = httpx.get(FRED_BASE, params=params, timeout=settings.http_timeout)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if obs and obs[0].get("value", ".") != ".":
            return float(obs[0]["value"])
    except Exception as e:
        logger.warning("FRED API error for %s: %s", series_id, e)
    return None


def get_treasury_rate() -> float | None:
    """10-Year Treasury Constant Maturity Rate (decimal, e.g. 0.043)."""
    val = _fetch_fred("GS10")
    return round(val / 100, 4) if val is not None else None


def get_gdp_growth() -> float | None:
    """Real GDP growth rate (decimal)."""
    val = _fetch_fred("A191RL1Q225SBEA")
    return round(val / 100, 4) if val is not None else None


def get_cpi() -> float | None:
    """CPI year-over-year percent change (decimal, e.g. 0.028)."""
    val = _fetch_fred("CPIAUCSL", units="pc1")
    return round(val / 100, 4) if val is not None else None


def get_fed_funds_rate() -> float | None:
    """Effective Federal Funds Rate (decimal)."""
    val = _fetch_fred("FEDFUNDS")
    return round(val / 100, 4) if val is not None else None


def get_macro_snapshot() -> MacroData:
    return MacroData(
        treasury_10y=get_treasury_rate(),
        fed_funds_rate=get_fed_funds_rate(),
        gdp_growth=get_gdp_growth(),
        cpi_inflation=get_cpi(),
    )
