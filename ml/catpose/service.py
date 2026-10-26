"""FastAPI service exposing face + body landmark predictors.

Endpoints:
- GET  /                  -> web dashboard (camera preview + data)
- GET  /health            -> which checkpoints are loaded
- POST /predict/face      -> multipart image (+ optional bbox), returns 48 pts
- POST /predict/body      -> multipart image (+ optional bbox), returns 20 pts
- POST /predict/all       -> both, on the same image (no detector)
- POST /predict/all/image -> both, returns PNG with overlay drawn
- POST /predict/auto      -> YOLO cat detector + face + body, returns rich JSON
- POST /predict/auto/image-> YOLO auto + returns PNG overlay (query ?data=1
                              adds JSON in a header for dashboards)

The auto endpoints handle the whole webcam use case: send a raw frame and
get back landmarks in the ORIGINAL image coordinates, along with which cat
was detected and the confidence.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .crop import face_bbox_from_body_keypoints, face_bbox_from_face_landmarks
from .draw import MIN_CONF, overlay
from .predictor import LandmarkPredictor
from .smoothing import OneEuro

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "ml" / "checkpoints"
FACE_CKPT = Path(os.environ.get("CATPOSE_FACE_CKPT", CKPT_DIR / "face.pt"))
BODY_CKPT = Path(os.environ.get("CATPOSE_BODY_CKPT", CKPT_DIR / "body.pt"))
YOLO_CKPT = Path(os.environ.get("CATPOSE_YOLO_CKPT", CKPT_DIR / "yolov8s.pt"))
DEVICE = os.environ.get("CATPOSE_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
WEBUI_DIR = Path(__file__).parent / "webui"


app = FastAPI(title="CatPose", version="0.2.0")
_face: Optional[LandmarkPredictor] = None
_body: Optional[LandmarkPredictor] = None
_detector = None  # type: ignore[assignment]


TTA = os.environ.get("CATPOSE_TTA", "1") != "0"


def _load(kind: str) -> LandmarkPredictor:
    ckpt = FACE_CKPT if kind == "face" else BODY_CKPT
    if not ckpt.exists():
        raise HTTPException(
            status_code=503,
            detail=f"checkpoint missing: {ckpt}. Train first: python -m catpose.train_hm --kind {kind}",
        )
    predictor = LandmarkPredictor(kind, ckpt, device=DEVICE)
    predictor.default_tta = TTA
    return predictor


def _load_detector():
    from .detector import CatDetector

    if not YOLO_CKPT.exists():
        raise HTTPException(
            status_code=503,
            detail=f"YOLO weights missing: {YOLO_CKPT}",
        )
    return CatDetector(weights=YOLO_CKPT, device=DEVICE)


@app.on_event("startup")
async def startup() -> None:
    global _face, _body, _detector
    if FACE_CKPT.exists():
        _face = _load("face")
    if BODY_CKPT.exists():
        _body = _load("body")
    if YOLO_CKPT.exists():
        try:
            _detector = _load_detector()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to preload YOLO detector: {exc}")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "face_loaded": _face is not None,
        "body_loaded": _body is not None,
        "detector_loaded": _detector is not None,
        "face_ckpt": str(FACE_CKPT),
        "body_ckpt": str(BODY_CKPT),
        "yolo_ckpt": str(YOLO_CKPT),
        "device": DEVICE,
    }


def _parse_bbox(raw: Optional[str]) -> Optional[tuple]:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"bbox must be JSON: {exc}") from exc
    if not (isinstance(value, list) and len(value) == 4):
        raise HTTPException(status_code=400, detail="bbox must be [x0,y0,x1,y1]")
    return tuple(float(v) for v in value)


async def _read_image(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    try:
        return Image.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc


def _prediction_to_json(pred, landmarks=None) -> Optional[dict]:
    if pred is None:
        return None
    pts = pred.landmarks if landmarks is None else landmarks
    out = {
        "kind": pred.kind,
        "bbox": [float(v) for v in pred.bbox],
        "landmarks": np.round(pts, 2).tolist(),
        "num_landmarks": int(pts.shape[0]),
    }
    if pred.confidences is not None:
        out["confidences"] = np.round(pred.confidences, 3).tolist()
        out["num_confident"] = int((pred.confidences >= MIN_CONF).sum())
    return out


@app.post("/predict/face")
async def predict_face(
    image: UploadFile = File(...),
    bbox: Optional[str] = Form(None),
) -> JSONResponse:
    global _face
    if _face is None:
        _face = _load("face")
    img = await _read_image(image)
    pred = _face.predict(img, bbox=_parse_bbox(bbox))
    return JSONResponse(_prediction_to_json(pred))


@app.post("/predict/body")
async def predict_body(
    image: UploadFile = File(...),
    bbox: Optional[str] = Form(None),
) -> JSONResponse:
    global _body
    if _body is None:
        _body = _load("body")
    img = await _read_image(image)
    pred = _body.predict(img, bbox=_parse_bbox(bbox))
    return JSONResponse(_prediction_to_json(pred))


@app.post("/predict/all")
async def predict_all(
    image: UploadFile = File(...),
    face_bbox: Optional[str] = Form(None),
    body_bbox: Optional[str] = Form(None),
) -> JSONResponse:
    global _face, _body
    if _face is None:
        _face = _load("face")
    if _body is None:
        _body = _load("body")
    img = await _read_image(image)
    return JSONResponse({
        "face": _prediction_to_json(_face.predict(img, bbox=_parse_bbox(face_bbox))),
        "body": _prediction_to_json(_body.predict(img, bbox=_parse_bbox(body_bbox))),
    })


@app.post("/predict/all/image")
async def predict_all_image(
    image: UploadFile = File(...),
    face_bbox: Optional[str] = Form(None),
    body_bbox: Optional[str] = Form(None),
) -> Response:
    global _face, _body
    if _face is None:
        _face = _load("face")
    if _body is None:
        _body = _load("body")
    img = await _read_image(image)
    face_pred = _face.predict(img, bbox=_parse_bbox(face_bbox))
    body_pred = _body.predict(img, bbox=_parse_bbox(body_bbox))
    out = overlay(
        img,
        face_pts=face_pred.landmarks,
        body_pts=body_pred.landmarks,
        face_bbox=face_pred.bbox,
        body_bbox=body_pred.bbox,
        face_conf=face_pred.confidences,
        body_conf=body_pred.confidences,
    )
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# The dashboard is a single local client streaming frames, so one pair of
# filters is enough. They reset when the stream pauses or the cat is lost.
_body_filter, _face_filter = OneEuro(), OneEuro()
_last_auto_call = 0.0


def _run_auto(img: Image.Image, draw_overlay: bool):
    """YOLO -> body -> face on a head-derived crop. Returns (payload, overlay or None)."""
    global _face, _body, _detector, _last_auto_call
    if _face is None:
        _face = _load("face")
    if _body is None:
        _body = _load("body")
    if _detector is None:
        _detector = _load_detector()

    now = time.time()
    if now - _last_auto_call > 1.0:
        _body_filter.reset()
        _face_filter.reset()
    _last_auto_call = now

    img = img.convert("RGB")
    arr = np.asarray(img)
    t0 = time.time()
    detection = _detector.best_cat(arr)
    t_det = (time.time() - t0) * 1000

    body_pred = face_pred = body_pts = face_pts = None
    detected = None
    if detection is None:
        # No cat: return no keypoints rather than noise across the background.
        _body_filter.reset()
        _face_filter.reset()
    else:
        box_scale = max(detection.bbox[2] - detection.bbox[0], detection.bbox[3] - detection.bbox[1])
        body_pred = _body.predict(arr, bbox=detection.bbox)
        body_pts = _body_filter(body_pred.landmarks, scale=box_scale)
        face_input_bbox = face_bbox_from_body_keypoints(body_pts, img.size, conf=body_pred.confidences)
        if face_input_bbox is None and body_pred.confidences is None:
            face_input_bbox = detection.bbox  # legacy model: no confidences to gate on
        if face_input_bbox is not None:
            face_pred = _face.predict(arr, bbox=face_input_bbox)
            # Second pass on a box rebuilt from pass-1 landmarks: recovers the
            # accuracy lost to a rough head-derived box (NME 1.58% -> 1.19%).
            refined = face_bbox_from_face_landmarks(face_pred.landmarks, face_pred.confidences)
            if refined is not None:
                face_pred = _face.predict(arr, bbox=refined)
            face_pts = _face_filter(face_pred.landmarks, scale=face_pred.bbox[2] - face_pred.bbox[0])
        else:
            _face_filter.reset()
        detected = {
            "class_id": detection.cls_id,
            "label": detection.label,
            "confidence": round(detection.confidence, 4),
            "bbox": [round(v, 2) for v in detection.bbox],
        }

    t_kp = (time.time() - t0) * 1000 - t_det

    payload = {
        "detected": detected,
        "face": _prediction_to_json(face_pred, face_pts),
        "body": _prediction_to_json(body_pred, body_pts),
        "timing_ms": {
            "detector": round(t_det, 1),
            "landmarks": round(t_kp, 1),
            "total": round(t_det + t_kp, 1),
        },
        "image_size": {"width": img.size[0], "height": img.size[1]},
    }

    if draw_overlay:
        rendered = overlay(
            img,
            face_pts=face_pts,
            body_pts=body_pts,
            face_bbox=face_pred.bbox if face_pred else None,
            body_bbox=tuple(detection.bbox) if detection else None,
            face_conf=face_pred.confidences if face_pred else None,
            body_conf=body_pred.confidences if body_pred else None,
        )
        return payload, rendered
    return payload, None


@app.post("/predict/auto")
async def predict_auto(image: UploadFile = File(...)) -> JSONResponse:
    img = await _read_image(image)
    payload, _ = _run_auto(img, draw_overlay=False)
    return JSONResponse(payload)


@app.post("/predict/auto/image")
async def predict_auto_image(
    image: UploadFile = File(...),
    include_data: bool = True,
) -> Response:
    img = await _read_image(image)
    payload, rendered = _run_auto(img, draw_overlay=True)

    # Return a JSON payload with the base64 PNG inside — simpler for the
    # dashboard than juggling multipart or headers.
    if include_data:
        buf = io.BytesIO()
        rendered.save(buf, format="JPEG", quality=85)
        payload["overlay_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        payload["overlay_mime"] = "image/jpeg"
        return JSONResponse(payload)

    buf = io.BytesIO()
    rendered.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------- Web dashboard ----------

if WEBUI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEBUI_DIR)), name="webui")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_html = WEBUI_DIR / "index.html"
    if not index_html.exists():
        return HTMLResponse(
            "<h1>catpose</h1><p>Dashboard file missing at ml/catpose/webui/index.html</p>",
            status_code=500,
        )
    return HTMLResponse(index_html.read_text(encoding="utf-8"))
