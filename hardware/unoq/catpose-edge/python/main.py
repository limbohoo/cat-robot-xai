# CatPose Edge（UNO Q, Linux 侧）
# 摄像头 -> YOLO 找猫 -> 身体 20 点 + 猫脸 48 点 -> 网页实时画面（Mac 浏览器打开板子的 :7000）
# LED 点阵：有猫显示猫脸，无猫显示待机；SG90 舵机：有猫左右摇摆，无猫回中。
import threading
import time
import traceback

import cv2
import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import Camera
from arduino.app_utils import *

import catpose_onnx as co

RESOLUTION = (1280, 720)  # Brio 105 支持到 1920x1080；越大画面越清楚，但每帧更慢
MIN_CONF = 0.25
WAG_CONF = 0.4   # 猫检测置信度达到这个值才算"真的有猫"
WAG_HITS = 2     # 连续这么多次 YOLO 检测到猫才开始摇摆，防止误判让舵机乱动

_ = 0
CAT_FACE = np.array([
    [0, 0, 7, _, _, _, _, _, _, _, 7, 0, 0],
    [0, 0, 7, 7, _, _, _, _, _, 7, 7, 0, 0],
    [0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0],
    [0, 0, 7, _, 7, _, _, _, 7, _, 7, 0, 0],
    [0, 0, 7, _, _, _, 7, _, _, _, 7, 0, 0],
    [0, 0, 7, _, _, 7, _, 7, _, _, 7, 0, 0],
    [0, 0, _, 7, _, _, _, _, _, 7, _, 0, 0],
    [0, 0, _, _, 7, 7, 7, 7, 7, _, _, 0, 0],
], dtype=np.uint8)
IDLE = np.zeros((8, 13), dtype=np.uint8)
IDLE[4, 3:6] = 2
IDLE[4, 7:10] = 2  # 两只闭着的眼睛


def _led(frame: np.ndarray):
    try:
        Bridge.call("draw", Frame(frame).to_board_bytes())
    except Exception as e:
        print("LED matrix error:", e)


ui = WebUI()
pipe = co.CatPose(detect_every=3, bbox_hold=6, face_passes=1, threads=2)  # 留出 CPU 给摄像头和推流

lock = threading.Lock()
latest_frame: np.ndarray | None = None  # 最新摄像头画面（供推理用）
latest_result = co.Result()             # 最近一次识别结果（叠加到每一帧上）
latest_jpeg: bytes | None = None
status = {"cat": False}
cat_state = None
cat_hits = 0
infer_ms = 0.0


def open_camera():
    """摄像头没插时不退出，每 3 秒重试一次（方便换扩展坞时热插拔）。"""
    while True:
        try:
            cam = Camera(resolution=RESOLUTION, fps=15, codec="MJPG")  # YUYV 在 720p 只有 5fps
            cam.start()
            print("摄像头已连接", RESOLUTION)
            return cam
        except Exception as e:
            print("等待摄像头…", e)
            time.sleep(3)


def capture_loop():
    """按摄像头帧率推送画面：视频保持流畅，关键点用最近一次推理结果叠加。"""
    global latest_frame, latest_jpeg
    camera = open_camera()
    while True:
        try:
            frame = camera.capture()
        except Exception as e:  # 摄像头掉线：重新打开，而不是让推流线程悄悄退出
            print("摄像头读取失败，重新连接:", e)
            try:
                camera.stop()
            except Exception:
                pass
            camera = open_camera()
            continue
        if frame is None:
            time.sleep(0.01)
            continue
        with lock:
            latest_frame, r, ms = frame, latest_result, infer_ms
        out = co.draw(frame, r, MIN_CONF)
        cv2.putText(out, f"AI {1000 / max(ms, 1):.1f} fps  {ms:.0f} ms", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ok, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with lock:
                latest_jpeg = jpeg.tobytes()


def react(r: co.Result):
    """把识别结果变成身体动作：LED 点阵 + 舵机摇摆（只在状态切换时发送）。"""
    global cat_state, cat_hits
    if r.cat is None:
        cat_hits = 0
    elif pipe.since_hit == 1:  # 这一帧 YOLO 刚刚检测到猫
        cat_hits = cat_hits + 1 if r.cat.confidence >= WAG_CONF else 0
    has_cat = cat_hits >= WAG_HITS if not cat_state else r.cat is not None
    if has_cat != cat_state:
        cat_state = has_cat
        _led(CAT_FACE if has_cat else IDLE)
        try:
            Bridge.call("servo_wag", has_cat)
        except Exception as e:
            print("servo error:", e)
        print("看到猫 -> 摇摆" if has_cat else "猫离开 -> 回中")


def loop():
    try:
        infer_once()
    except Exception:  # 单帧出错只记日志，不让整个 App 退出
        traceback.print_exc()
        time.sleep(0.5)


def infer_once():
    global latest_result, status, infer_ms
    with lock:
        frame = latest_frame
    if frame is None:
        time.sleep(0.05)
        return
    t = time.perf_counter()
    r = pipe(frame)
    total = (time.perf_counter() - t) * 1000
    react(r)

    body = None
    if r.body is not None:
        body = [{"name": n, "x": round(float(x), 1), "y": round(float(y), 1), "conf": round(float(c), 2)}
                for n, (x, y), c in zip(co.BODY_NAMES, r.body, r.body_conf)]
    with lock:
        latest_result, infer_ms = r, total
        status = {
            "time": time.time(),
            "cat": r.cat is not None,
            "confidence": round(r.cat.confidence, 3) if r.cat else None,
            "bbox": [round(v) for v in r.cat.bbox] if r.cat else None,
            "stale": r.stale,
            "ms": {k: round(v, 1) for k, v in (r.ms or {}).items()} | {"total": round(total, 1)},
            "body": body,
            "face_points": int((r.face_conf >= MIN_CONF).sum()) if r.face is not None else 0,
            "wagging": bool(cat_state),
        }


def stream():
    from fastapi.responses import StreamingResponse

    def frames():
        last = None
        while True:
            with lock:
                jpeg = latest_jpeg
            if jpeg is None or jpeg is last:
                time.sleep(0.02)
                continue
            last = jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def get_status():
    with lock:
        return status


ui.expose_api("GET", "/stream", stream)
ui.expose_api("GET", "/status", get_status)
# 手动测试舵机：浏览器打开 /wag_on 或 /wag_off（下次识别状态变化时会被覆盖）
ui.expose_api("GET", "/wag_on", lambda: {"ok": Bridge.call("servo_wag", True) is None or True})
ui.expose_api("GET", "/wag_off", lambda: {"ok": Bridge.call("servo_wag", False) is None or True})

threading.Thread(target=capture_loop, daemon=True).start()
_led(IDLE)
Bridge.call("servo_wag", False)
print("CatPose Edge 已启动：在 Mac 浏览器打开 http://lidia.local:7000")
App.run(user_loop=loop)
