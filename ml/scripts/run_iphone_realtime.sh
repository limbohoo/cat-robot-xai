#!/usr/bin/env bash
# 用 iPhone（连续互通相机）启动 CatPose 实时识别，并保存录像和报告。
# 每次运行都会新建一个带时间戳的目录，不会覆盖以前的测试:
#   sessions/test1/iphone_YYYYMMDD_HHMMSS/{raw.mp4, annotated.mp4, report.html, ...}
# 用法: ./ml/scripts/run_iphone_realtime.sh              自动找 iPhone
#       ./ml/scripts/run_iphone_realtime.sh 2            手动指定摄像头编号
#       SESSION=sessions/test2 ./ml/scripts/run_iphone_realtime.sh   换存放的 session 目录
# 窗口内: q / Esc 退出（退出时才会写报告）, b 重新校准基线
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=ml
PY=ml/.venv/bin/python

LIST=$($PY -m catpose.realtime --list-cameras 2>&1 | grep -- '--source' || true)
echo "可用摄像头:"; echo "$LIST"

if [ $# -ge 1 ]; then
  SRC=$1
else
  SRC=$(echo "$LIST" | grep -v -i -e 'Desk View' -e 'MacBook' -e 'Capture screen' \
        | grep -i -e 'iPhone' | head -1 | awk '{print $2}')
fi

if [ -z "${SRC:-}" ]; then
  echo "没有找到 iPhone 摄像头：检查同一 Apple ID、Wi-Fi/蓝牙、iPhone 的\"连续互通相机\"已开且锁屏对准场景。"
  echo "也可手动指定编号: $0 <编号>"
  exit 1
fi

SESSION=${SESSION:-sessions/test1}
D="$SESSION/iphone_$(date +%Y%m%d_%H%M%S)"
# 同一秒内重复启动时再加后缀，保证绝不覆盖已有目录
N=1; BASE=$D
while [ -e "$D" ]; do D="${BASE}_$N"; N=$((N+1)); done
mkdir -p "$D"

echo "使用摄像头 --source $SRC"
echo "本次输出目录: $D"
$PY -m catpose.realtime --source "$SRC" --detect \
  --report-dir "$D" --output "$D/annotated.mp4"

echo
echo "报告: $D/report.html"
echo "原始录像: $D/raw.mp4"
open "$D/report.html" 2>/dev/null || true
