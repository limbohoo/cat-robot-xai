# Quickstart

All commands run from the `ml/` folder of this repository.

## 0. Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r ../requirements.txt
```

The YOLOv8 detector weights are downloaded by Ultralytics on first use.

## 1. Browser dashboard

```bash
.venv/bin/uvicorn catpose.service:app --port 8080
```

Open http://localhost:8080/ in Safari, Chrome or Firefox and click
**Start camera**.

- Left: video with the YOLO cat box (green), body skeleton (orange) and face
  landmarks (red). Low-confidence points are hidden.
- Right: detected class, confidence, bbox, how many points passed the
  confidence threshold, per-keypoint coordinates with confidence, and
  per-stage server timing.

Server options (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `CATPOSE_DEVICE` | `mps` if available, else `cpu` | Inference device (`cuda` also works) |
| `CATPOSE_TTA` | `1` | `0` disables flip test-time augmentation (faster on CPU) |
| `CATPOSE_FACE_CKPT` / `CATPOSE_BODY_CKPT` / `CATPOSE_YOLO_CKPT` | files in `checkpoints/` | Use other weights |

Example for a CPU-only laptop:

```bash
CATPOSE_TTA=0 CATPOSE_YOLO_CKPT=checkpoints/yolov8n.pt .venv/bin/uvicorn catpose.service:app --port 8080
```

Run it in the background:

```bash
nohup .venv/bin/uvicorn catpose.service:app --port 8080 > /tmp/catpose.log 2>&1 &
echo $! > /tmp/catpose.pid
kill "$(cat /tmp/catpose.pid)"   # stop later
```

## 2. OpenCV two-window mode

```bash
.venv/bin/python -m catpose.realtime --detect
```

Opens `catpose | camera` (video + overlay) and `catpose | data` (metrics,
colour-coded keypoint confidence), and prints a status line in the terminal.
Press `q` or `Esc` to quit.

| Flag | Default | Meaning |
|---|---|---|
| `--source 1` | `0` | Camera index, or a video file path |
| `--output out.mp4` | off | Save the annotated video |
| `--no-display` | off | Headless (use with `--output`) |
| `--device cpu` | `mps` if available | Inference device |
| `--no-tta` | off | Skip flip augmentation (~2x faster on CPU) |
| `--min-conf 0.4` | `0.25` | Hide keypoints below this confidence |
| `--face-passes 1` | `2` | Face refinement passes |
| `--no-smooth` | off | Disable One-Euro temporal smoothing |
| `--yolo-ckpt checkpoints/yolov8n.pt` | `yolov8s.pt` | Smaller, faster detector |
| `--detect-every 3` | `2` | Run YOLO every N frames |
| `--bbox-hold 30` | `15` | Frames to keep the last box when YOLO misses |
| `--skip-face` / `--skip-body` | off | Run only one landmark model |
| `--no-data` / `--no-print` | off | Hide the data window / terminal line |

## 3. Single image through the API

```bash
curl -s -F "image=@cat.jpg" http://localhost:8080/predict/auto | python3 -m json.tool
curl -s -F "image=@cat.jpg" "http://localhost:8080/predict/auto/image?include_data=false" -o out.png
```

See [api.md](api.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `curl http://localhost:8080/health` returns nothing | Server not running; start it (step 1) |
| Browser: "Camera unavailable" | System Settings → Privacy & Security → Camera → allow the browser, reload |
| Very low FPS (~1.5 on CPU) | No GPU found. Use `--no-tta --yolo-ckpt checkpoints/yolov8n.pt` (~2.8 FPS), optionally `--face-passes 1` |
| "no cat detected" while a cat is visible | Cat too small or dark: move closer, add light, or try `--yolo-imgsz 800` |
| No face points drawn | Head keypoints weren't confident enough (cat facing away); this is intentional |
