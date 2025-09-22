from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import cv2


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def _angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return float(math.degrees(math.atan2(dy, dx)))


def extract_features_from_landmarks(landmarks: List[Dict[str, float]]) -> Dict[str, float]:
    # expected names: withers, brisket, front_hoof, shoulder, hook, pin
    pts = {lm["name"].lower(): (float(lm["x"]), float(lm["y"])) for lm in landmarks}
    required = ["withers", "brisket", "front_hoof", "shoulder", "hook", "pin"]
    if not all(name in pts for name in required):
        raise ValueError(f"Missing required landmarks. Need: {required}")

    withers = pts["withers"]
    brisket = pts["brisket"]
    front_hoof = pts["front_hoof"]
    shoulder = pts["shoulder"]
    hook = pts["hook"]
    pin = pts["pin"]

    height_px = abs(front_hoof[1] - withers[1])
    chest_depth_px = abs(withers[1] - brisket[1])
    body_length_px = _euclidean(shoulder, pin)
    rump_angle_deg = _angle_deg(hook, pin)

    features = {
        "method": "landmarks",
        "height_px": float(height_px) if height_px is not None else None,
        "chest_depth_px": float(chest_depth_px) if chest_depth_px is not None else None,
        "body_length_px": float(body_length_px) if body_length_px is not None else None,
        "rump_angle_deg": float(rump_angle_deg) if rump_angle_deg is not None else None,
    }

    if height_px and height_px > 0:
        features["body_length_ratio"] = float(body_length_px / height_px)
        features["chest_depth_ratio"] = float(chest_depth_px / height_px)
    else:
        features["body_length_ratio"] = None
        features["chest_depth_ratio"] = None

    return features


def _largest_contour_bbox(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    # returns x, y, w, h for largest contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h


def extract_features_auto(image_bytes: bytes) -> Dict[str, float]:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image")

    bbox = _largest_contour_bbox(img)
    if bbox is None:
        raise ValueError("Could not find silhouette")
    _, _, w, h = bbox

    height_px = float(h)
    body_length_px = float(w)

    features = {
        "method": "auto_bbox",
        "height_px": height_px,
        "body_length_px": body_length_px,
        "chest_depth_px": None,
        "rump_angle_deg": None,
        "body_length_ratio": float(body_length_px / height_px) if height_px > 0 else None,
        # No withers/brisket; use a conservative placeholder for MVP
        "chest_depth_ratio": 0.45,
    }
    return features


def extract_features(image_bytes: Optional[bytes], landmarks: Optional[List[Dict[str, float]]], auto_detect: bool) -> Dict[str, float]:
    if landmarks and len(landmarks) >= 6:
        return extract_features_from_landmarks(landmarks)
    if auto_detect and image_bytes is not None:
        return extract_features_auto(image_bytes)
    raise ValueError("No landmarks provided and auto_detect is False; cannot extract features")
