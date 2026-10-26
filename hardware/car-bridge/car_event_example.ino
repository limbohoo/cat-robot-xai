// Example for the car's Wi-Fi board (ESP32 / Arduino with WiFi): report every executed
// command to the laptop running `python -m catpose.robot_bridge SESSION`, and obey its STOP.
// Copy reportEvent() and checkStop() into the team's firmware; keep the existing remote control.

#include <WiFi.h>
#include <HTTPClient.h>

const char* WIFI_SSID = "your-hotspot";
const char* WIFI_PASS = "your-password";
const char* BRIDGE = "http://192.168.1.20:8800";  // the address the bridge prints at start

void stopMotors() {
  // TEAM: put your motor stop code here
}

// Call right after the car starts an action, e.g. reportEvent("slow_prey", "round1").
// Use the same names as the bridge page: slow_prey, light, sound, approach, retreat, round_end.
void reportEvent(const char* event, const char* detail) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String(BRIDGE) + "/event?src=car&e=" + event + "&d=" + detail;
  http.setTimeout(500);  // never block driving for long
  http.begin(url);
  http.GET();
  http.end();
}

// Call about every 300 ms in loop(): stops the motors when someone pressed STOP on the bridge page.
unsigned long lastPoll = 0;
void checkStop() {
  if (millis() - lastPoll < 300 || WiFi.status() != WL_CONNECTED) return;
  lastPoll = millis();
  HTTPClient http;
  http.setTimeout(300);
  http.begin(String(BRIDGE) + "/status");
  if (http.GET() == 200 && http.getString().indexOf("\"stop\": true") >= 0) {
    stopMotors();
  }
  http.end();
}

void setup() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void loop() {
  checkStop();
  // ... the team's existing remote-control handling, calling reportEvent(...) per command
}
