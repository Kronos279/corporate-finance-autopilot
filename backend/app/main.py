from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import agent, analysis, company, reports


app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    if settings.database_url:
        from app.services.rag import init_db
        await init_db()


@app.get("/")
async def root() -> dict:
    return {"app": settings.app_name, "status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


app.include_router(company.router, prefix="/api/company", tags=["Company"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])