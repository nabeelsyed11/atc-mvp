from __future__ import annotations
from typing import Optional, Tuple, List
from pathlib import Path
import os
import io

import numpy as np
import cv2
from PIL import Image

# Optional ONNX Runtime (loaded lazily as well)
try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional
    ort = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[1]

_SESSION = None  # type: ignore
_INPUT_NAME = None
_INPUT_SIZE = 224
_LABELS: List[str] = ["cattle", "buffalo"]
_USE_CROP: bool = True
_TTA: bool = True
_USE_REMBG: bool = False
_USE_LETTERBOX: bool = True
_USE_CLAHE: bool = True


def _load_labels(path: Path) -> List[str]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
                if lines:
                    return lines
    except Exception:
        pass
    return ["cattle", "buffalo"]


def _lazy_init() -> None:
    global _SESSION, _INPUT_NAME, _INPUT_SIZE, _LABELS, _USE_CROP, _TTA, _USE_REMBG, _USE_LETTERBOX, _USE_CLAHE, ort
    if _SESSION is not None:
        return

    # Read config from env
    model_path = os.getenv("SPECIES_ONNX_PATH")
    labels_path = os.getenv("SPECIES_LABELS_PATH")
    input_size_env = os.getenv("SPECIES_INPUT_SIZE")
    use_crop_env = os.getenv("SPECIES_USE_CROP")
    tta_env = os.getenv("SPECIES_TTA")
    use_rembg_env = os.getenv("SPECIES_USE_REMBG")
    use_letterbox_env = os.getenv("SPECIES_USE_LETTERBOX")
    use_clahe_env = os.getenv("SPECIES_USE_CLAHE")

    if input_size_env:
        try:
            _INPUT_SIZE = int(input_size_env)
        except Exception:
            _INPUT_SIZE = 224
    if use_crop_env is not None:
        _USE_CROP = use_crop_env.strip().lower() in ("1", "true", "yes", "y")
    if tta_env is not None:
        _TTA = tta_env.strip().lower() in ("1", "true", "yes", "y")
    if use_rembg_env is not None:
        _USE_REMBG = use_rembg_env.strip().lower() in ("1", "true", "yes", "y")
    if use_letterbox_env is not None:
        _USE_LETTERBOX = use_letterbox_env.strip().lower() in ("1", "true", "yes", "y")
    if use_clahe_env is not None:
        _USE_CLAHE = use_clahe_env.strip().lower() in ("1", "true", "yes", "y")

    labels_file = Path(labels_path) if labels_path else (BASE_DIR / "models" / "species_labels.txt")
    _LABELS = _load_labels(labels_file)

    # If onnxruntime wasn't available at import time, try to import it now
    if ort is None:
        try:  # pragma: no cover - optional
            import importlib
            ort = importlib.import_module("onnxruntime")  # type: ignore
        except Exception:
            ort = None  # still unavailable
            return  # Use heuristic

    model_file = Path(model_path) if model_path else (BASE_DIR / "models" / "species_classifier.onnx")
    if not model_file.exists():
        # Model not present; fallback to heuristic
        return

    try:
        _SESSION = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])  # type: ignore
        _INPUT_NAME = _SESSION.get_inputs()[0].name  # type: ignore
    except Exception:
        _SESSION = None
        _INPUT_NAME = None


def _preprocess_for_onnx(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    # Optional crop around main subject
    img_bgr = _crop_main_subject(image_bgr) if _USE_CROP else image_bgr
    # Optional CLAHE for contrast enhancement
    if _USE_CLAHE:
        img_bgr = _apply_clahe(img_bgr)
    # Convert BGR to RGB and resize with optional letterbox (keep aspect)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if _USE_LETTERBOX:
        img_rgb = _letterbox(img_rgb, size)
    else:
        img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    img = img_rgb
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _onnx_predict(image_bgr: np.ndarray) -> Optional[Tuple[str, float]]:
    if _SESSION is None or _INPUT_NAME is None:
        return None
    # Test-time augmentation: average predictions with horizontal flip
    imgs = [image_bgr]
    if _TTA:
        # Horizontal flip
        imgs.append(cv2.flip(image_bgr, 1))
        # Slight contrast/brightness variants (deterministic)
        imgs.append(_adjust_contrast_brightness(image_bgr, alpha=1.1, beta=0))
        imgs.append(_adjust_contrast_brightness(image_bgr, alpha=0.9, beta=0))
    batched = [_preprocess_for_onnx(im, _INPUT_SIZE) for im in imgs]  # list of (1,C,H,W)
    inp = np.concatenate(batched, axis=0)  # (N,C,H,W)
    try:
        outputs = _SESSION.run(None, {_INPUT_NAME: inp})  # type: ignore
        logits = outputs[0]
        if logits.ndim == 1:
            logits = np.expand_dims(logits, 0)
        mean_logits = np.mean(logits, axis=0)
        probs = _softmax(mean_logits)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = _LABELS[idx] if idx < len(_LABELS) else str(idx)
        # Normalize to our domain if labels are not exact
        label = label.lower()
        if label not in ("cattle", "buffalo"):
            # simple mapping: if contains 'buff', map to buffalo else cattle
            if "buff" in label:
                label = "buffalo"
            else:
                label = "cattle"
        return label, conf
    except Exception:
        return None


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return out
    except Exception:
        return img_bgr


def _letterbox(img_rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    # place centered
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _adjust_contrast_brightness(img_bgr: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    # alpha: contrast (1.0 = no change), beta: brightness offset
    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return out


def _heuristic_predict(image_bgr: np.ndarray) -> Tuple[str, float]:
    # Very rough heuristic: darker grayscale -> buffalo
    img_bgr = _crop_main_subject(image_bgr) if _USE_CROP else image_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    # Threshold chosen arbitrarily for MVP demo
    if mean_val < 90:
        return "buffalo", 0.55
    else:
        return "cattle", 0.55


def predict_species(image_bytes: bytes) -> Tuple[str, float, str]:
    """Return (species, confidence, method).
    species in {"cattle", "buffalo"}
    method in {"onnx", "heuristic"}
    """
    _lazy_init()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image for species classification")

    onnx_res = _onnx_predict(img)
    if onnx_res is not None:
        sp, conf = onnx_res
        return sp, conf, "onnx"

    sp, conf = _heuristic_predict(img)
    return sp, conf, "heuristic"


def _largest_contour_bbox(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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


def _crop_main_subject(img_bgr: np.ndarray) -> np.ndarray:
    bbox = _crop_with_rembg_bbox(img_bgr) if _USE_REMBG else _largest_contour_bbox(img_bgr)
    if bbox is None:
        return img_bgr
    x, y, w, h = bbox
    # Pad bbox by 10%
    pad_x = int(round(0.1 * w))
    pad_y = int(round(0.1 * h))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img_bgr.shape[1], x + w + pad_x)
    y1 = min(img_bgr.shape[0], y + h + pad_y)
    crop = img_bgr[y0:y1, x0:x1]
    return crop if crop.size > 0 else img_bgr


def _crop_with_rembg_bbox(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    try:
        import importlib
        rembg = importlib.import_module("rembg")
        # Convert to PNG bytes
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bio = io.BytesIO()
        Image.fromarray(img_rgb).save(bio, format="PNG")
        data = bio.getvalue()
        out_bytes = rembg.remove(data)  # returns PNG bytes with alpha
        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        alpha = np.array(out_img)[:, :, 3]
        # Find bbox of non-zero alpha
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)
    except Exception:
        return None
