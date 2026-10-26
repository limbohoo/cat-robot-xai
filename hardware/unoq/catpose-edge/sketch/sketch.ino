// CatPose Edge（UNO Q, MCU 侧）
//   LED 点阵 8x13：Python 通过 draw() 推送图案（有猫 / 无猫）
//   SG90：橙线(信号) -> D9, 红线 -> 5V, 棕线 -> GND；Python 通过 servo_wag(1/0) 开关摇摆
// Zephyr 核心没有 Servo 库，这里用 50Hz 软件脉冲驱动舵机。

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

const uint8_t SERVO_PIN = 9;
const uint16_t PULSE_MIN_US = 500;   // 0°
const uint16_t PULSE_MAX_US = 2400;  // 180°
const uint8_t FRAME_SIZE = 8 * 13;

Arduino_LED_Matrix matrix;
uint8_t frame[FRAME_SIZE] = {0};
volatile bool frameDirty = true;
const int WAG_CENTER = 90;
const int WAG_AMPLITUDE = 35;        // 左右各摆 35°
const unsigned long WAG_PERIOD_MS = 700;

volatile uint16_t pulseUs = 1450;    // ~90°
volatile bool wagging = false;
volatile unsigned long holdUntil = 0;  // 回中后只再发一会儿脉冲，之后停发，舵机完全静止不抖
unsigned long lastPulse = 0;

void draw(std::vector<uint8_t> newFrame) {
  memcpy(frame, newFrame.data(), min(newFrame.size(), (size_t)FRAME_SIZE));
  frameDirty = true;
}

void servo_write(int angle) {
  holdUntil = millis() + 600;
  pulseUs = map(constrain(angle, 0, 180), 0, 180, PULSE_MIN_US, PULSE_MAX_US);
}

void servo_wag(bool on) {
  wagging = on;
  if (!on) {
    servo_write(WAG_CENTER);
    holdUntil = millis() + 600;      // 给舵机 0.6 秒转回中间
  }
}

void setup() {
  pinMode(SERVO_PIN, OUTPUT);
  matrix.begin();
  matrix.setGrayscaleBits(3);        // 亮度 0-7
  matrix.clear();

  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.provide("servo_write", servo_write);
  Bridge.provide("servo_wag", servo_wag);
}

void loop() {
  unsigned long now = micros();
  if (now - lastPulse >= 20000 && (wagging || (long)(millis() - holdUntil) < 0)) {  // 50Hz，空闲时不发脉冲
    lastPulse = now;
    if (wagging) {                   // 正弦摇摆，像猫尾巴一样
      float phase = (millis() % WAG_PERIOD_MS) * 2.0f * PI / WAG_PERIOD_MS;
      servo_write(WAG_CENTER + WAG_AMPLITUDE * sin(phase));
    }
    // 不能关中断：关中断会让 MCU 丢失 Bridge 串口数据，导致 RPC 超时
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(pulseUs);
    digitalWrite(SERVO_PIN, LOW);
  }
  if (frameDirty) {
    frameDirty = false;
    matrix.draw(frame);
  }
}
