# SG90 舵机 + PIR 人体感应 硬件测试（UNO Q, Linux 侧）
# 启动先做一次舵机扫动自检，之后：有人 -> 舵机转到 150°，无人 -> 回到 30°。
import time

from arduino.app_utils import *

REST_ANGLE = 30
ALERT_ANGLE = 150


def on_pir(state: int):
    if state:
        print("PIR: 检测到人体 -> 舵机转向", ALERT_ANGLE)
        Bridge.call("servo_write", ALERT_ANGLE)
    else:
        print("PIR: 无人 -> 舵机回位", REST_ANGLE)
        Bridge.call("servo_write", REST_ANGLE)


Bridge.provide("on_pir", on_pir)

print("舵机自检：0 -> 90 -> 180 -> 90")
for angle in (0, 90, 180, 90):
    Bridge.call("servo_write", angle)
    time.sleep(1)
Bridge.call("servo_write", REST_ANGLE)
print("自检完成。PIR 上电需预热约 30-60 秒，当前读数:", Bridge.call("pir_read"))

App.run()
