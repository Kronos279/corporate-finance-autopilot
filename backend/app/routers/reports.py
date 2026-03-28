import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.models.company import GeneratedReport, ReportType
from app.services.report import generator

router = APIRouter()
_generated_reports: dict[str, str] = {}


class GenerateReportRequest(BaseModel):
    report_type: ReportType


@router.post("/{ticker}/generate")
async def generate_report(ticker: str, request: GenerateReportRequest):
    report = await generator.generate_report(ticker, request.report_type)
    _generated_reports[ticker.upper()] = report.model_dump_json(indent=2)
    return report


@router.get("/{ticker}/download")
async def download_report(ticker: str):
    payload = _generated_reports.get(ticker.upper())
    if not payload:
        raise HTTPException(status_code=404, detail="No report found. Generate one first.")

    report = GeneratedReport(**json.loads(payload))
    pdf_bytes = generator.create_pdf(report)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{ticker.upper()}_report.pdf"'
        },
    )
