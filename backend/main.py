"""
FastAPI Backend for MarineDebrisSAM-Net.
Serves dual-stream PyTorch inference, foreground segmentation, and verified
benchmark analytics, and optionally serves the built React SPA in production.
"""

import io
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.data import CURATED_SAMPLES, DATASET_STATS, MODELS_BENCHMARK, TARGET_CLASSES
from backend.inference import (
    CHECKPOINTS_DIR,
    CV2_AVAILABLE,
    MODEL_CHECKPOINT_FILES,
    TORCH_AVAILABLE,
    ModelRegistry,
    generate_foreground_mask,
)

app = FastAPI(
    title="MarineDebrisSAM-Net API",
    description="Dual-stream PyTorch inference & ablation analytics API for underwater marine debris classification.",
    version="1.0.0",
)

cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()


@app.get("/api/health")
def health():
    weights = registry.scan_checkpoints()
    return {
        "status": "ok",
        "torch_available": TORCH_AVAILABLE,
        "opencv_available": CV2_AVAILABLE,
        "device": str(registry.device) if registry.device else "unavailable",
        "checkpoints_dir": str(CHECKPOINTS_DIR),
        "expected_checkpoints": MODEL_CHECKPOINT_FILES,
        "weights_loaded": weights,
        "any_weights_loaded": any(weights.values()),
    }


@app.get("/api/metrics")
def get_metrics():
    return {
        "classes": TARGET_CLASSES,
        "dataset": DATASET_STATS,
        "models": MODELS_BENCHMARK,
    }


@app.get("/api/samples")
def get_samples():
    return {"samples": CURATED_SAMPLES}


@app.post("/api/segment")
async def segment_image(
    image: UploadFile = File(...),
    x1: Optional[int] = Form(None),
    y1: Optional[int] = Form(None),
    x2: Optional[int] = Form(None),
    y2: Optional[int] = Form(None),
):
    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    bbox = None
    if None not in (x1, y1, x2, y2):
        bbox = (int(x1), int(y1), int(x2), int(y2))

    return generate_foreground_mask(pil_img, bbox=bbox)


@app.post("/api/predict")
async def predict_debris(
    orig_image: UploadFile = File(...),
    seg_image: Optional[UploadFile] = File(None),
    model_id: str = Form("a5_opt"),
):
    if not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PyTorch is not installed in the active backend environment. Install backend/requirements.txt to run live PyTorch inference.",
        )

    orig_bytes = await orig_image.read()
    try:
        orig_pil = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid original image: {exc}")

    if seg_image is not None:
        seg_bytes = await seg_image.read()
        try:
            seg_pil = Image.open(io.BytesIO(seg_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid segmented image: {exc}")
    else:
        seg_pil = orig_pil

    try:
        result = registry.predict(model_id=model_id, orig_img=orig_pil, seg_img=seg_pil)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result


@app.post("/api/detect_frame")
async def detect_live_frame(
    frame: UploadFile = File(...),
    model_id: str = Form("a5_opt"),
):
    """
    Single-call live video/camera frame endpoint:
    1) Isolates foreground object + bounding box
    2) Runs dual-stream PyTorch classification on (orig, segmented)
    """
    if not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PyTorch is not installed in the active backend environment.",
        )

    raw = await frame.read()
    try:
        orig_pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid frame image: {exc}")

    seg_info = generate_foreground_mask(orig_pil)
    seg_b64 = seg_info["segmented_data_url"].split(",", 1)[1]
    import base64
    seg_pil = Image.open(io.BytesIO(base64.b64decode(seg_b64))).convert("RGB")

    try:
        pred = registry.predict(model_id=model_id, orig_img=orig_pil, seg_img=seg_pil)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        **pred,
        "bbox": seg_info["bbox"],
        "segmentation_method": seg_info["method"],
        "segmented_data_url": seg_info["segmented_data_url"],
    }


# Serve built React frontend in single-container production deployments
DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
