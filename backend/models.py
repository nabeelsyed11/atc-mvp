from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Landmark(BaseModel):
    name: str
    x: float
    y: float


class ScoreResponse(BaseModel):
    record_id: Optional[str]
    species: Optional[str] = None
    detected_species: Optional[str] = None
    species_confidence: Optional[float] = None
    species_method: Optional[str] = None
    features: Dict[str, Any]
    scores: Dict[str, Optional[float]]
    overall_score: Optional[float]
