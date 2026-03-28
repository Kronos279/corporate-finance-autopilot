"""RAG service: embed, store, and search company documents using Neon pgvector + HuggingFace embeddings."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

# ── Lazy-loaded embedding model ──
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Generate embedding vector for a text string."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return np.asarray(vec)


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Generate embedding vectors for a batch of texts."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=settings.embedding_batch_size)
    return [embeddings[i] for i in range(len(texts))]


# ── Chunking ──

def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping word-level chunks."""
    chunk_size = chunk_size if chunk_size is not None else settings.rag_chunk_size
    overlap = overlap if overlap is not None else settings.rag_chunk_overlap
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def _chunk_id(ticker: str, source: str, chunk_text: str) -> str:
    """Deterministic chunk ID to prevent duplicates on re-ingestion."""
    raw = f"{ticker}:{source}:{chunk_text[:200]}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── DB helpers ──

_pool: asyncpg.Pool | None = None
_db_initialized: bool = False


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL not set")
        # Ensure pgvector extension exists before pool tries register_vector
        if not _db_initialized:
            await init_db()
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            setup=_setup_connection,
        )
    return _pool


async def _setup_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def init_db() -> None:
    """Create the pgvector extension and documents table if they don't exist."""
    global _db_initialized
    if not settings.database_url:
        logger.warning("DATABASE_URL not set — RAG disabled.")
        return
    # Use a raw connection to bootstrap the extension + table
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({settings.embedding_dimension}),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_ticker ON documents (ticker);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON documents USING hnsw (embedding vector_cosine_ops);
        """)
    finally:
        await conn.close()
    _db_initialized = True
    logger.info("RAG table initialised in Neon.")


# ── Ingest ──

async def ingest_chunks(
    ticker: str,
    source: str,
    text: str,
    chunk_size: int | None = None,
) -> int:
    """Chunk text, embed, and upsert into Neon pgvector. Returns number of chunks stored."""
    chunk_size = chunk_size if chunk_size is not None else settings.rag_chunk_size
    if not settings.database_url:
        logger.warning("DATABASE_URL not set — skipping ingest.")
        return 0

    # Clean HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) < 50:
        return 0

    chunks = chunk_text(clean, chunk_size=chunk_size)
    if not chunks:
        return 0

    vectors = embed_batch(chunks)
    pool = await _get_pool()

    rows: list[tuple[str, str, str, str, Any]] = []
    for chunk, vec in zip(chunks, vectors):
        cid = _chunk_id(ticker, source, chunk)
        rows.append((cid, ticker.upper(), source, chunk, vec))

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO documents (id, ticker, source, content, embedding)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (id) DO NOTHING;
            """,
            rows,
        )

    logger.info("Ingested %d chunks for %s from %s", len(rows), ticker, source)
    return len(rows)


# ── Search ──

async def search(
    query: str,
    ticker: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Semantic search over stored documents. Returns top_k most relevant chunks."""
    if not settings.database_url:
        return []

    q_vec = embed_text(query)
    pool = await _get_pool()

    if ticker:
        sql = """
            SELECT content, source, 1 - (embedding <=> $1) AS similarity
            FROM documents
            WHERE ticker = $2
            ORDER BY embedding <=> $1
            LIMIT $3;
        """
        params = (q_vec, ticker.upper(), top_k)
    else:
        sql = """
            SELECT content, source, 1 - (embedding <=> $1) AS similarity
            FROM documents
            ORDER BY embedding <=> $1
            LIMIT $2;
        """
        params = (q_vec, top_k)

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return [
        {"content": r["content"], "source": r["source"], "similarity": float(r["similarity"])}
        for r in rows
    ]


async def ingest_company_data(ticker: str) -> dict:
    """Ingest all available data for a ticker: SEC filings, profile, financials summary."""
    from app.services.ingestion.sec_edgar import get_cik, get_filings_list, get_filing_text
    from app.services.ingestion.yahoo_finance import get_profile, get_financials

    total = 0
    ticker = ticker.upper()

    # 1. Company profile & description
    try:
        profile = get_profile(ticker)
        if profile.description:
            n = await ingest_chunks(ticker, "profile", profile.description, chunk_size=256)
            total += n
    except Exception as e:
        logger.warning("Profile ingest error for %s: %s", ticker, e)

    # 2. Financials summary text
    try:
        fin = get_financials(ticker)
        lines = []
        for stmt in fin.income_statements:
            lines.append(
                f"FY{stmt.fiscal_year}: Revenue ${stmt.revenue:,.0f}, "
                f"Gross Margin {stmt.gross_margin:.1%}, "
                f"Operating Income ${stmt.operating_income:,.0f}, "
                f"Net Income ${stmt.net_income:,.0f}"
            )
        for bs in fin.balance_sheets:
            lines.append(
                f"FY{bs.fiscal_year} Balance: Total Assets ${bs.total_assets:,.0f}, "
                f"Total Debt ${bs.total_debt:,.0f}, Net Debt ${bs.net_debt:,.0f}, "
                f"D/E {bs.debt_to_equity:.2f}"
            )
        for cf in fin.cash_flows:
            lines.append(
                f"FY{cf.fiscal_year} Cash Flow: Operating ${cf.operating_cash_flow:,.0f}, "
                f"CapEx ${cf.capex:,.0f}, FCF ${cf.free_cash_flow:,.0f}"
            )
        if lines:
            n = await ingest_chunks(ticker, "financials_summary", "\n".join(lines), chunk_size=256)
            total += n
    except Exception as e:
        logger.warning("Financials ingest error for %s: %s", ticker, e)

    # 3. SEC filings text
    try:
        cik = get_cik(ticker)
        if cik:
            for form_type in ["10-K", "10-Q"]:
                filings = get_filings_list(cik, form_type)
                for filing in filings[:3]:  # Latest 3 filings of each type
                    acc = filing.get("accession_number", "")
                    if acc:
                        text = get_filing_text(cik, acc)
                        source = f"sec_{form_type}_{filing.get('filing_date', acc)}"
                        n = await ingest_chunks(ticker, source, text, chunk_size=512)
                        total += n
    except Exception as e:
        logger.warning("SEC ingest error for %s: %s", ticker, e)

    return {"ticker": ticker, "chunks_ingested": total}
