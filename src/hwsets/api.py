"""FastAPI backend.  Run:  uvicorn hwsets.api:app --reload"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

import anthropic

from . import api_key_configured, describe_api_error, output_dir
from .ingest import render_page_png
from .mock import MOCK_WARNING, MockExtractor
from .pipeline import parse_page_spec, run_pipeline
from .schemas import ExtractionResult

STORE = output_dir()
app = FastAPI(title="Hardware Set Extractor")


def _paths(doc_id: str) -> tuple[Path, Path]:
    if not doc_id.isalnum():
        raise HTTPException(400, "bad id")
    return STORE / f"{doc_id}.pdf", STORE / f"{doc_id}.json"


@app.get("/health")
def health() -> dict:
    return {"ok": True, "mock": os.getenv("HWSETS_MOCK") == "1"}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    pages: str | None = Query(None, description="e.g. '40-52,60' (skips page detection)"),
) -> dict:
    data = await file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "not a PDF")
    mock = os.getenv("HWSETS_MOCK") == "1"
    if not mock and not api_key_configured():
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set. Set it, or start with HWSETS_MOCK=1 (dev mode).")
    try:
        result = run_pipeline(
            data, name=file.filename, pages_hint=parse_page_spec(pages),
            extractor=MockExtractor() if mock else None,
        )
    except anthropic.APIError as e:
        raise HTTPException(502, describe_api_error(e))
    if mock:
        result.document += " [MOCK]"
        result.warnings.insert(0, MOCK_WARNING)
    doc_id = uuid.uuid4().hex[:12]
    pdf_path, json_path = _paths(doc_id)
    STORE.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(data)
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return {"id": doc_id, "result": result.model_dump()}


@app.get("/results/{doc_id}", response_model=ExtractionResult)
def get_result(doc_id: str) -> ExtractionResult:
    _, jp = _paths(doc_id)
    if not jp.exists():
        raise HTTPException(404)
    return ExtractionResult.model_validate_json(jp.read_text(encoding="utf-8"))


@app.put("/results/{doc_id}", response_model=ExtractionResult)
def save_corrections(doc_id: str, result: ExtractionResult) -> ExtractionResult:
    """Persist a user-corrected result (feedback UI)."""
    _, jp = _paths(doc_id)
    if not jp.exists():
        raise HTTPException(404)
    jp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


@app.get("/results/{doc_id}/page/{page}.png")
def page_image(doc_id: str, page: int, zoom: float = 1.5) -> Response:
    pp, _ = _paths(doc_id)
    if not pp.exists():
        raise HTTPException(404)
    try:
        png = render_page_png(pp, page, zoom)
    except IndexError:
        raise HTTPException(404, "page out of range")
    return Response(png, media_type="image/png")
