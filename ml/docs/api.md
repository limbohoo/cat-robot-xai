# HTTP API

```bash
.venv/bin/uvicorn catpose.service:app --port 8080
```

Base URL `http://localhost:8080`. All `POST` endpoints take multipart form data
with an `image` file.

## `GET /` — dashboard

## `GET /health`

```json
{"status": "ok", "face_loaded": true, "body_loaded": true, "detector_loaded": true,
 "face_ckpt": "...", "body_ckpt": "...", "yolo_ckpt": ".../yolov8s.pt", "device": "mps"}
```

## `POST /predict/auto` — recommended

Full pipeline: YOLO → body → two-pass face. Also applies temporal smoothing
between calls made less than 1 s apart (meant for a single streaming client).

```json
{
  "detected": {"class_id": 15, "label": "cat", "confidence": 0.91, "bbox": [7.8, 17.7, 285.5, 239.3]},
  "body": {
    "kind": "body", "bbox": [...], "num_landmarks": 20,
    "landmarks": [[x, y], ...],
    "confidences": [0.83, ...],
    "num_confident": 17
  },
  "face": { "kind": "face", "num_landmarks": 48, "landmarks": [...], "confidences": [...], "num_confident": 48, "bbox": [...] },
  "timing_ms": {"detector": 11.0, "landmarks": 27.5, "total": 38.5},
  "image_size": {"width": 640, "height": 480}
}
```

- Coordinates are pixels in the uploaded image.
- `confidences` are in [0, 1]; the dashboard hides points below 0.25.
- No cat found → `detected`, `body`, `face` are all `null`.
- Head not confidently located (e.g. cat facing away) → `face` is `null`.

## `POST /predict/auto/image?include_data=true`

Same as above plus `overlay_png_base64` (a JPEG) and `overlay_mime`.
With `include_data=false`, returns the overlay PNG directly.

## `POST /predict/face`, `POST /predict/body`

Run one model on the image. Optional form field `bbox` = `"[x0, y0, x1, y1]"`;
without it the whole image is used. Returns one object like `body` above.

## `POST /predict/all`, `POST /predict/all/image`

Both models on the same image, optional `face_bbox` / `body_bbox`, no
detector. The `/image` variant returns a PNG.

## Environment variables

`CATPOSE_DEVICE` (`mps` / `cuda` / `cpu`), `CATPOSE_TTA` (`0` to disable flip
augmentation), `CATPOSE_FACE_CKPT`, `CATPOSE_BODY_CKPT`, `CATPOSE_YOLO_CKPT`.
