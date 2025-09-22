from typing import Dict, Optional, Tuple


SPECIES_TARGETS: Dict[str, Dict[str, float]] = {
    "cattle": {
        # approximate placeholders; tune for policy
        "body_length_ratio_target": 1.10,
        "chest_depth_ratio_target": 0.45,
        "rump_angle_deg_target": 10.0,
    },
    "buffalo": {
        "body_length_ratio_target": 1.05,
        "chest_depth_ratio_target": 0.43,
        "rump_angle_deg_target": 10.0,
    },
}


def score_ratio(value: Optional[float], target: float) -> Optional[float]:
    if value is None:
        return None
    if target == 0:
        return 5.0
    d = abs(value - target) / abs(target)
    # simple step-wise mapping
    if d <= 0.05:
        return 9.0
    if d <= 0.10:
        return 8.0
    if d <= 0.20:
        return 6.0
    if d <= 0.30:
        return 4.0
    return 2.0


def score_angle_close(value: Optional[float], target: float = 10.0) -> Optional[float]:
    if value is None:
        return None
    d = abs(value - target)
    if d <= 2:
        return 9.0
    if d <= 5:
        return 8.0
    if d <= 10:
        return 6.0
    if d <= 15:
        return 4.0
    return 2.0


def compute_scores(features: Dict[str, float], species: str = "cattle") -> Tuple[Dict[str, float], Optional[float]]:
    species = (species or "cattle").lower()
    spec = SPECIES_TARGETS.get(species, SPECIES_TARGETS["cattle"])  # default to cattle

    blr = features.get("body_length_ratio")
    cdr = features.get("chest_depth_ratio")
    ra = features.get("rump_angle_deg")

    scores: Dict[str, Optional[float]] = {
        "body_length_ratio": score_ratio(blr, spec["body_length_ratio_target"]),
        "chest_depth_ratio": score_ratio(cdr, spec["chest_depth_ratio_target"]),
        "rump_angle_deg": score_angle_close(ra, spec["rump_angle_deg_target"]),
    }

    # compute overall as weighted average of available scores
    weights = {
        "body_length_ratio": 0.4,
        "chest_depth_ratio": 0.4,
        "rump_angle_deg": 0.2,
    }
    total_w = 0.0
    acc = 0.0
    for k, s in scores.items():
        if s is not None:
            w = weights.get(k, 0.0)
            acc += s * w
            total_w += w
    overall = (acc / total_w) if total_w > 0 else None

    # cast Optional[float] to float in final dict where possible
    return {k: (float(v) if v is not None else None) for k, v in scores.items()}, (float(overall) if overall is not None else None)
