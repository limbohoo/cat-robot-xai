// SG90 舵机 + PIR 人体感应 硬件测试（UNO Q, MCU 侧）
//   SG90: 橙线(信号) -> D9, 红线 -> 5V, 棕线 -> GND
//   PIR : OUT -> D2, VCC -> 5V, GND -> GND
// Zephyr 核心没有 Servo 库，这里用 50Hz 软件脉冲驱动舵机。

#include <Arduino_RouterBridge.h>

const uint8_t SERVO_PIN = 9;
const uint8_t PIR_PIN = 2;
const uint16_t PULSE_MIN_US = 500;   // 0°
const uint16_t PULSE_MAX_US = 2400;  // 180°

volatile uint16_t pulseUs = 1450;    // ~90°
int lastPir = -1;
unsigned long lastFrame = 0;

void servo_write(int angle) {
  angle = constrain(angle, 0, 180);
  pulseUs = map(angle, 0, 180, PULSE_MIN_US, PULSE_MAX_US);
  Monitor.print("servo -> ");
  Monitor.println(angle);
}

int pir_read() {
  return digitalRead(PIR_PIN);
}

void setup() {
  pinMode(SERVO_PIN, OUTPUT);
  pinMode(PIR_PIN, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);   // 板载 RGB 灯低电平点亮

  Bridge.begin();
  Monitor.begin();
  Bridge.provide("servo_write", servo_write);
  Bridge.provide("pir_read", pir_read);
}

void loop() {
  unsigned long now = micros();
  if (now - lastFrame >= 20000) {    // 50Hz
    lastFrame = now;
    // 不能关中断：关中断会让 MCU 丢失 Bridge 串口数据，导致 RPC 超时
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(pulseUs);
    digitalWrite(SERVO_PIN, LOW);

    int pir = digitalRead(PIR_PIN);
    if (pir != lastPir) {
      lastPir = pir;
      digitalWrite(LED_BUILTIN, pir ? LOW : HIGH);
      Bridge.notify("on_pir", pir);
    }
  }
}
