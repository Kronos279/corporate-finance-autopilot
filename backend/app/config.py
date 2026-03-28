from pathlib import Path

from pydantic_settings import BaseSettings

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    app_name: str = "Corporate Finance Autopilot"
    debug: bool = True
    port: int = 8080
    cors_origins: str = "*"

    # Together AI
    together_api_key: str = ""
    together_base_url: str = "https://api.together.xyz/v1"
    together_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4000

    # Agent
    agent_max_iterations: int = 12
    agent_tool_result_max_chars: int = 8000

    # SEC EDGAR
    sec_user_agent: str = "YourName your@email.com"
    sec_base_url: str = "https://www.sec.gov"
    sec_data_url: str = "https://data.sec.gov"
    sec_max_filings: int = 10
    sec_filing_max_chars: int = 20000

    # FRED
    fred_api_key: str = ""
    fred_base_url: str = "https://api.stlouisfed.org/fred/series/observations"

    # Logo.dev
    logo_dev_token: str = ""

    # FMP (optional)
    fmp_api_key: str = ""

    # Neon Postgres (pgvector)
    database_url: str = ""
    db_pool_min: int = 1
    db_pool_max: int = 5

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64

    # RAG chunking
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64

    # HTTP
    http_timeout: int = 30

    # Financial model defaults
    equity_risk_premium: float = 0.055
    credit_spread: float = 0.015
    fallback_risk_free_rate: float = 0.04
    wacc_floor: float = 0.04
    wacc_cap: float = 0.13
    min_growth_rate: float = 0.03
    default_starting_revenue: float = 100_000_000_000
    default_shares_outstanding: float = 1_000_000_000
    default_current_price: float = 100.0

    class Config:
        env_file = str(_ENV_FILE)


# THIS LINE IS CRITICAL — make sure it exists
settings = Settings()