from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .models import Landmark, ScoreResponse
from . import image_processing as imgproc
from . import scoring
from . import storage
from . import species_classifier

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="ATC MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend assets under /static
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="UI not found")
    return FileResponse(str(index_path))


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/score", response_model=ScoreResponse)
async def api_score(
    image: UploadFile = File(...),
    species: str = Form("cattle"),
    landmarks_json: Optional[str] = Form(None),
    auto_detect: bool = Form(False),
    auto_species: bool = Form(False),
    save_record: bool = Form(True),
):
    try:
        image_bytes = await image.read()
        landmarks: Optional[List[Dict[str, Any]]] = None
        if landmarks_json:
            try:
                landmarks = json.loads(landmarks_json)
            except Exception:
                landmarks = None

        # Species detection (optional ML)
        detected_species: Optional[str] = None
        species_confidence: Optional[float] = None
        species_method: Optional[str] = None
        try:
            det_sp, det_conf, det_method = species_classifier.predict_species(image_bytes)
            detected_species = det_sp
            species_confidence = det_conf
            species_method = det_method
        except Exception:
            pass

        species_input = (species or "cattle").lower()
        # Confidence gating for auto species
        thr = 0.6
        try:
            thr_env = os.getenv("SPECIES_CONF_THRESHOLD")
            if thr_env is not None:
                thr = float(thr_env)
        except Exception:
            thr = 0.6

        if auto_species or species_input == "auto":
            if detected_species and (species_confidence or 0.0) >= thr:
                species_used = detected_species
            else:
                # Fallback to user choice if provided, else default cattle
                species_used = species_input if species_input in ("cattle", "buffalo") else "cattle"
        else:
            species_used = species_input

        features = imgproc.extract_features(image_bytes, landmarks, auto_detect)
        scores, overall = scoring.compute_scores(features, species_used)

        record_id: Optional[str] = None
        if save_record:
            rec = storage.save_record(
                species=species_used,
                features=features,
                scores=scores,
                overall_score=overall,
                landmarks=landmarks,
                image_bytes=image_bytes,
                detected_species=detected_species,
                species_confidence=species_confidence,
                species_method=species_method,
            )
            record_id = rec.get("id")

        return ScoreResponse(
            record_id=record_id,
            species=species_used,
            detected_species=detected_species,
            species_confidence=species_confidence,
            species_method=species_method,
            features=features,
            scores=scores,
            overall_score=overall,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {e}")


@app.get("/api/records")
async def api_records(limit: int = 10) -> List[Dict[str, Any]]:
    return storage.load_recent(limit)


@app.get("/api/export")
async def api_export() -> FileResponse:
    path = storage.csv_path()
    if not Path(path).exists():
        # Ensure a CSV exists even with no data
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
    return FileResponse(path, media_type="text/csv", filename="records.csv")


class BPARequest(BaseModel):
    record_id: str


@app.post("/api/send_to_bpa")
async def api_send_to_bpa(req: BPARequest) -> Dict[str, Any]:
    record = storage.load_record(req.record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    bpa_url = os.getenv("BPA_URL")
    if not bpa_url:
        # Stubbed response
        return {"sent": False, "reason": "BPA_URL not configured", "payload_example": _make_bpa_payload(record)}

    headers = {}
    api_key = os.getenv("BPA_API_KEY")
    auth_type = os.getenv("BPA_AUTH_TYPE", "Bearer")
    if api_key:
        headers["Authorization"] = f"{auth_type} {api_key}"
    headers["Content-Type"] = "application/json"

    payload = _make_bpa_payload(record)

    try:
        resp = requests.post(bpa_url, headers=headers, json=payload, timeout=10)
        return {"sent": resp.ok, "status_code": resp.status_code, "response_text": resp.text}
    except Exception as e:
        return {"sent": False, "error": str(e), "payload": payload}


def _make_bpa_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal illustrative payload mapping; adapt to BPA spec when available
    f = record.get("features", {})
    s = record.get("scores", {})
    return {
        "record_id": record.get("id"),
        "timestamp": record.get("timestamp"),
        "species": record.get("species"),
        "traits": {
            "body_length_ratio": {
                "value": f.get("body_length_ratio"),
                "score": s.get("body_length_ratio"),
            },
            "chest_depth_ratio": {
                "value": f.get("chest_depth_ratio"),
                "score": s.get("chest_depth_ratio"),
            },
            "rump_angle_deg": {
                "value": f.get("rump_angle_deg"),
                "score": s.get("rump_angle_deg"),
            },
        },
        "overall_score": record.get("overall_score"),
    }
