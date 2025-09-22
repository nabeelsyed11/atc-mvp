from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timezone
import json
import csv
import uuid

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = DATA_DIR / "records.jsonl"
CSV_PATH = DATA_DIR / "records.csv"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


def _write_image(image_bytes: Optional[bytes], record_id: str) -> Optional[str]:
    if not image_bytes:
        return None
    fname = f"{record_id}.jpg"
    out_path = IMAGES_DIR / fname
    try:
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        return str(out_path)
    except Exception:
        return None


def save_record(*, species: str, features: Dict[str, Any], scores: Dict[str, float], overall_score: Optional[float], landmarks: Optional[List[Dict[str, Any]]], image_bytes: Optional[bytes], detected_species: Optional[str] = None, species_confidence: Optional[float] = None, species_method: Optional[str] = None) -> Dict[str, Any]:
    record_id = _short_id()
    ts = _now_iso()
    image_path = _write_image(image_bytes, record_id)

    record = {
        "id": record_id,
        "timestamp": ts,
        "species": species,
        "detected_species": detected_species,
        "species_confidence": species_confidence,
        "species_method": species_method,
        "features": features,
        "scores": scores,
        "overall_score": overall_score,
        "landmarks": landmarks,
        "image_path": image_path,
    }

    # append JSONL
    with open(JSONL_PATH, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(record, ensure_ascii=False) + "\n")

    # append/update CSV summary
    _append_csv(record)

    return record


def _append_csv(record: Dict[str, Any]) -> None:
    header = [
        "id", "timestamp", "species", "detected_species", "species_confidence", "species_method", "method",
        "height_px", "body_length_px", "chest_depth_px", "rump_angle_deg",
        "body_length_ratio", "chest_depth_ratio",
        "score_body_length_ratio", "score_chest_depth_ratio", "score_rump_angle_deg",
        "overall_score",
    ]

    features = record.get("features", {})
    scores = record.get("scores", {})

    row = {
        "id": record.get("id"),
        "timestamp": record.get("timestamp"),
        "species": record.get("species"),
        "detected_species": record.get("detected_species"),
        "species_confidence": record.get("species_confidence"),
        "species_method": record.get("species_method"),
        "method": features.get("method"),
        "height_px": features.get("height_px"),
        "body_length_px": features.get("body_length_px"),
        "chest_depth_px": features.get("chest_depth_px"),
        "rump_angle_deg": features.get("rump_angle_deg"),
        "body_length_ratio": features.get("body_length_ratio"),
        "chest_depth_ratio": features.get("chest_depth_ratio"),
        "score_body_length_ratio": scores.get("body_length_ratio"),
        "score_chest_depth_ratio": scores.get("chest_depth_ratio"),
        "score_rump_angle_deg": scores.get("rump_angle_deg"),
        "overall_score": record.get("overall_score"),
    }

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_recent(n: int = 10) -> List[Dict[str, Any]]:
    if not JSONL_PATH.exists():
        return []
    lines: List[str]
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    recent = lines[-n:]
    return [json.loads(x) for x in recent]


def load_record(record_id: str) -> Optional[Dict[str, Any]]:
    if not JSONL_PATH.exists():
        return None
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("id") == record_id:
                    return obj
            except Exception:
                continue
    return None


def csv_path() -> str:
    return str(CSV_PATH)
