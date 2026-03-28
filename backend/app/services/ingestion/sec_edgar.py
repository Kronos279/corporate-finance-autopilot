from __future__ import annotations

import logging

import httpx

from app.config import settings
logger = logging.getLogger(__name__)

SEC_HEADERS = {"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"}


def _load_tickers_map() -> dict[str, str]:
    """Download the full ticker → CIK mapping from SEC."""
    try:
        resp = httpx.get(
            f"{settings.sec_base_url}/files/company_tickers.json",
            headers=SEC_HEADERS,
            timeout=settings.http_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    except Exception as e:
        logger.warning("SEC EDGAR tickers map error: %s", e)
        return {}


def get_cik(ticker: str) -> str | None:
    """Map ticker symbol to SEC CIK number."""
    tickers_map = _load_tickers_map()
    return tickers_map.get(ticker.upper())


def get_company_facts(cik: str) -> dict:
    """Fetch XBRL-structured financial facts from SEC EDGAR."""
    try:
        resp = httpx.get(
            f"{settings.sec_data_url}/api/xbrl/companyfacts/CIK{cik}.json",
            headers=SEC_HEADERS,
            timeout=settings.http_timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("SEC company facts error for CIK %s: %s", cik, e)
        return {}


def get_filings_list(cik: str, form_type: str = "10-K") -> list[dict]:
    """Get recent filings of a given type from SEC EDGAR."""
    try:
        resp = httpx.get(
            f"{settings.sec_data_url}/submissions/CIK{cik}.json",
            headers=SEC_HEADERS,
            timeout=settings.http_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == form_type:
                results.append({
                    "form_type": form,
                    "accession_number": accessions[i] if i < len(accessions) else "",
                    "filing_date": dates[i] if i < len(dates) else "",
                    "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                })
        return results[:settings.sec_max_filings]
    except Exception as e:
        logger.warning("SEC filings error for CIK %s: %s", cik, e)
        return []


def get_filing_text(cik: str, accession_number: str) -> str:
    """Fetch the filing index page content (first 20 KB)."""
    try:
        clean_acc = accession_number.replace("-", "")
        cik_int = int(cik)
        url = (
            f"{settings.sec_base_url}/Archives/edgar/data/"
            f"{cik_int}/{clean_acc}/{accession_number}-index.htm"
        )
        resp = httpx.get(url, headers=SEC_HEADERS, timeout=settings.http_timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp.text[:settings.sec_filing_max_chars]
    except Exception as e:
        logger.warning("SEC filing text error: %s", e)
        return f"Filing text unavailable for accession {accession_number}."
