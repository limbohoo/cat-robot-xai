"""Build CatPose_Run.ipynb: the practical Colab notebook (video in -> owner report out).

Unlike the theory tutorial (build_notebook.py), this one runs the real catpose
package from catpose_run_pack.zip (code + checkpoints, built by
ml/scripts/build_share_pack.py). Outside Colab it reads CATPOSE_DATA,
CATPOSE_WORK and CATPOSE_VIDEO so the same cells can be tested locally.

    python3 ml/colab/build_run_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "colab" / "CatPose_Run.ipynb"
cells: list[dict] = []


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)})


def code(src: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(True)})


md("""
# CatPose v4 · 在 Colab 上分析视频，生成主人报告

这个笔记本运行的是**和电脑上完全相同的程序**：输入一段猫的视频，输出一份主人报告。报告包括时间占比、事件卡片（变了什么、依据、视频片段）和情绪推测。

> 系统**不识别情绪**。报告里的「看起来像……」只是推测，会标明把握程度和另一种可能；阈值和概率都还没有用标注数据验证。

**准备（只做一次）**
1. 在 Google Drive 的「我的云端硬盘」里新建文件夹 **`CatPose-XAI`**。
2. 把 **`catpose_run_pack.zip`** 放进去，不要解压。
3. 把要分析的视频（`.mp4` / `.mov`）也放进去。

**运行**
1. 菜单 **运行时 → 更改运行时类型 → T4 GPU**。没有 GPU 也能跑，但会慢很多。
2. 菜单 **运行时 → 全部运行**。第一次会请求访问 Google Drive，请允许。
3. 等第 4 步显示「完成」，第 5 步会在下面直接显示报告。

| 步骤 | 做什么 | 大约用时（T4 GPU） |
|---|---|---|
| 1 | 连接 Drive，解压代码和模型 | 30 秒 |
| 2 | 安装缺少的包，检查 GPU | 1 分钟 |
| 3 | 选择视频 | — |
| 4 | 分析视频 | 约等于视频时长 |
| 5 | 显示报告 | — |
| 6 | 把报告存回 Drive，并下载压缩包 | — |
| 7 | （可选）给事件打标签，更新推测 | — |
| 8 | （可选）多路视频 + 小车日志的完整测试 | — |
""")

md("## 1. 连接 Google Drive，准备代码和模型")
code(r'''
import os, sys, zipfile, subprocess, shutil, time, json, csv
from pathlib import Path

FOLDER = "CatPose-XAI"  # Google Drive 里的文件夹名
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA = Path("/content/drive/MyDrive") / FOLDER
    WORK = Path("/content/catpose")
else:  # 在自己电脑的 Jupyter 里运行时
    DATA = Path(os.environ.get("CATPOSE_DATA", ".")).resolve()
    WORK = Path(os.environ.get("CATPOSE_WORK", "catpose_work")).resolve()

pack = DATA / "catpose_run_pack.zip"
if not pack.exists():
    raise SystemExit(f"找不到 {pack}。请把 catpose_run_pack.zip 放进 Google Drive 的 {FOLDER} 文件夹，再重新运行。")
if not (WORK / "catpose").exists():
    WORK.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(pack).extractall(WORK)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print("代码和模型已就绪：", WORK)
print("模型：", ", ".join(sorted(p.name for p in (WORK / "checkpoints").glob("*.pt"))))
''')

md("## 2. 安装缺少的包，检查 GPU")
code(r'''
need = []
for module, package in (("torch", "torch"), ("torchvision", "torchvision"), ("cv2", "opencv-python"),
                        ("ultralytics", "ultralytics"), ("PIL", "Pillow")):
    try:
        __import__(module)
    except ImportError:
        need.append(package)
if need:
    print("安装：", " ".join(need))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("计算设备：", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
print("ffmpeg：", shutil.which("ffmpeg") or "没有（报告不检查声音线索）")
if DEVICE == "cpu":
    print("提示：没有 GPU 会很慢。可以在菜单 运行时 → 更改运行时类型 里选 T4 GPU，然后重新全部运行。")
''')

md("""
## 3. 选择视频

`VIDEO` 留空时，自动使用 Drive 文件夹里的第一个视频；Drive 里没有视频时，会弹出上传按钮。也可以填视频的完整路径。
""")
code(r'''
VIDEO = os.environ.get("CATPOSE_VIDEO", "")  # 例如 "/content/drive/MyDrive/CatPose-XAI/cat.mp4"

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
if not VIDEO:
    found = sorted(p for p in DATA.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if found:
        VIDEO = str(found[0])
    elif IN_COLAB:
        from google.colab import files
        uploaded = files.upload()
        VIDEO = str(Path(next(iter(uploaded))).resolve())
if not VIDEO or not Path(VIDEO).exists():
    raise SystemExit("没有找到视频：请把视频放进 Drive 文件夹，或在上面填 VIDEO。")
print("视频：", VIDEO)
''')

md("""
## 4. 分析视频

可以调整的设置：
- `CALIB_FRAMES`：用猫出现后的前多少帧作为它的「平常状态」。这段时间猫最好是安静的。**结果对这一项很敏感。**
- `BASELINE_IN`：如果之前保存过这只猫的基线（`.json`），填它的路径，就不再用视频开头校准。
- `INTERP_CONFIG`：第 7 步用标注更新后的推测配置。没有就留空。
- `MOVING_CAMERA`：视频是装在小车上的摄像头拍的，就改成 `True`。
""")
code(r'''
CALIB_FRAMES = 150
BASELINE_IN = ""
INTERP_CONFIG = ""
MOVING_CAMERA = False

OUT = WORK / "reports" / Path(VIDEO).stem
if OUT.exists():
    shutil.rmtree(OUT)
cmd = [sys.executable, "-m", "catpose.realtime", "--source", VIDEO, "--detect", "--no-display", "--no-print",
       "--device", DEVICE, "--calib-frames", str(CALIB_FRAMES), "--report-dir", str(OUT)]
if BASELINE_IN:
    cmd += ["--baseline-in", BASELINE_IN]
if INTERP_CONFIG:
    cmd += ["--interp-config", INTERP_CONFIG]
if MOVING_CAMERA:
    cmd += ["--moving-camera"]
if DEVICE == "cpu":  # lighter settings so a CPU run stays bearable
    cmd += ["--no-tta", "--face-passes", "1", "--yolo-ckpt", "checkpoints/yolov8n.pt"]

print("分析中，请稍候……")
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
log = []
for line in proc.stdout:
    log.append(line)
    if line.startswith(("[info]", "[warn]", "report:", "processed")):
        print(line.rstrip())
if proc.wait() != 0:
    print("".join(log[-30:]))
    raise SystemExit("分析失败，上面是最后的输出。")
print(f"完成，用时 {time.time() - t0:.0f} 秒。报告在 {OUT}")
''')

md("## 5. 查看报告")
code(r'''
events = json.load(open(OUT / "events.json", encoding="utf-8"))
summary = json.load(open(OUT / "summary.json", encoding="utf-8")) if (OUT / "summary.json").exists() else {}
print(f"时长 {summary.get('duration_s', 0):.0f} 秒，" if summary else "", f"共 {len(events)} 个事件：", sep="")
names = {"tense": "紧张", "curious": "好奇", "playful": "想玩", "attentive": "专注", "calm": "放松"}
for i, e in enumerate(events):
    g = e.get("interp") or {}
    guess = f"看起来像 {names.get(g.get('top'), '?')}（把握{g.get('band')}）" if g.get("show") else "不给推测"
    print(f"  {i}: {e.get('start_s', 0):6.1f} 秒  {e.get('reaction', ''):10s} {guess}")

if IN_COLAB:  # serve the report folder so its video clips play inside the notebook
    import threading, functools
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    from google.colab import output
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(OUT))
    for port in range(8765, 8790):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    threading.Thread(target=server.serve_forever, daemon=True).start()
    output.serve_kernel_port_as_iframe(port, path="/report.html", height=900)
else:
    print("用浏览器打开：", OUT / "report.html")
''')

md("""
## 6. 保存报告

报告会复制到 Drive 的 `CatPose-XAI/reports/` 里，同时打包成 zip 下载。转发报告时，要连同 `clips/` 文件夹一起发，否则视频片段打不开。

> 报告里有猫和家里的画面，请只在组内分享。
""")
code(r'''
dest = DATA / "reports" / OUT.name
shutil.copytree(OUT, dest, dirs_exist_ok=True)
archive = shutil.make_archive(str(WORK / OUT.name), "zip", OUT)
print("已保存到：", dest)
print("压缩包：", archive)
if IN_COLAB:
    try:
        from google.colab import files
        files.download(archive)
    except Exception as err:  # some browsers block the automatic download
        print("自动下载没有成功，请在左侧文件栏手动下载：", err)
''')

md("""
## 7. （可选）给事件打标签，更新推测

1. 看第 5 步报告里每个事件的视频片段。
2. 在下面的 `LABELS` 里填你认为的情绪，**先不要看系统的推测**。最好两个人各自独立填。
3. 运行这一格。它会把标注和原来手设的概率结合，得到新的推测配置，存到 Drive。
4. 回到第 4 步，把 `INTERP_CONFIG` 改成下面打印的路径，重新运行第 4、5 步。

只能填这几个词：`tense`（紧张）、`curious`（好奇）、`playful`（想玩）、`attentive`（专注）、`calm`（放松）、`unsure`（看不出来，不参与学习）。

几条标注只会轻微推动原来的概率，不会推翻它们。标注越多，影响越大。
""")
code(r'''
for r in csv.DictReader(open(OUT / "labels_template.csv", encoding="utf-8")):
    print(f"  事件 {r['event_id']}：{r['time']}  {r['reaction']}")

LABELS = {}  # 例如 {0: "attentive", 1: "tense", 2: "unsure"}

ALLOWED = {"tense", "curious", "playful", "attentive", "calm", "unsure"}
wrong = {k: v for k, v in LABELS.items() if v not in ALLOWED}
if wrong:
    raise SystemExit(f"这些标签不认识：{wrong}。只能填 {sorted(ALLOWED)}")
if not LABELS:
    print("还没有填 LABELS，这一步跳过。")
else:
    labels_csv = OUT / "my_labels.csv"
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "your_label"])
        w.writerows(sorted(LABELS.items()))
    new_cfg = DATA / "interp_config.json"
    args = [sys.executable, "-m", "catpose.interpret", "--events", str(OUT / "events.json"),
            "--labels", str(labels_csv), "--out", str(new_cfg)]
    if INTERP_CONFIG:  # keep learning on top of the previous update
        args += ["--config", INTERP_CONFIG]
    print(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    print(f'回到第 4 步，设置 INTERP_CONFIG = "{new_cfg}"，再运行第 4、5 步。')
''')

md("""
## 8. （可选）完整测试：多路视频 + 小车日志

正式测试时，一个测试文件夹里放几路视频、小车网桥记录的 `robot_log.csv` 和 `session.json`（写法见共享包里的 `hardware/car-bridge/README.md`）。把整个文件夹上传到 Drive 的 `CatPose-XAI/sessions/` 下，在 `SESSION` 里填文件夹名，然后运行这一格。

程序会对齐各路视频和小车日志的时间，分别分析每路视频，再合并成一份报告。
""")
code(r'''
SESSION = ""  # 例如 "test1"，对应 Drive 里的 CatPose-XAI/sessions/test1

if not SESSION:
    print("没有填 SESSION，这一步跳过。")
else:
    folder = DATA / "sessions" / SESSION
    local = WORK / "sessions" / SESSION  # work on a local copy: Drive is slow for video reads
    shutil.copytree(folder, local, dirs_exist_ok=True)
    proc = subprocess.run([sys.executable, "-m", "catpose.session", str(local)], capture_output=True, text=True)
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        raise SystemExit("完整测试分析失败，上面是输出。")
    shutil.copytree(local / "report", folder / "report", dirs_exist_ok=True)
    print("报告已存到：", folder / "report")
    if IN_COLAB:
        OUT = local / "report"
        print("重新运行第 5 步即可在这里查看这份报告。")
''')

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"{OUT}  ({len(cells)} cells)")
