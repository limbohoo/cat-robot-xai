// Option A: the car's Wi-Fi board (ESP32) receives commands from catpose.robot_bridge.
// The bridge calls  http://<car-ip>/cmd?c=<command>  for every button on its page:
//   ping, slow_prey, light, sound, approach, retreat, round_end, stop
// Start the bridge with:  --car-url "http://<car-ip>/cmd?c={cmd}"
//
// Safety: stop always works, every action ends by itself after ACTION_MAX_MS,
// and the car stops if it loses Wi-Fi.

#include <WiFi.h>
#include <WebServer.h>

const char* WIFI_SSID = "your-hotspot";
const char* WIFI_PASS = "your-password";
const unsigned long ACTION_MAX_MS = 8000;  // longest any single action may run

WebServer server(80);
unsigned long actionStarted = 0;
bool busy = false;

// ---- TEAM: replace these with your motor / LED / buzzer code ----
void stopMotors()  { /* all motors off */ }
void lightOff()    { /* LED off */ }
void doSlowPrey()  { /* slow zig-zag away from the cat */ }
void doLight()     { /* LED pattern on the floor */ }
void doSound()     { /* short quiet sound */ }
void doApproach()  { /* drive slowly forward */ }
void doRetreat()   { /* drive slowly backward */ }
// ------------------------------------------------------------------

void stopAll() {
  stopMotors();
  lightOff();
  busy = false;
}

void start(void (*action)()) {
  stopAll();  // never mix two actions
  action();
  busy = true;
  actionStarted = millis();
}

void handleCmd() {
  String c = server.arg("c");
  if (c == "stop" || c == "safety_stop" || c == "round_end") stopAll();
  else if (c == "slow_prey") start(doSlowPrey);
  else if (c == "light")     start(doLight);
  else if (c == "sound")     start(doSound);
  else if (c == "approach")  start(doApproach);
  else if (c == "retreat")   start(doRetreat);
  else if (c != "ping") { server.send(400, "text/plain", "unknown " + c); return; }
  server.send(200, "text/plain", "ok " + c);
}

void setup() {
  Serial.begin(115200);
  stopAll();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(200);
  Serial.print("car ip: ");
  Serial.println(WiFi.localIP());  // put this address in --car-url
  server.on("/cmd", handleCmd);
  server.begin();
}

void loop() {
  server.handleClient();
  if (busy && millis() - actionStarted > ACTION_MAX_MS) stopAll();
  if (busy && WiFi.status() != WL_CONNECTED) stopAll();
  // TEAM: if your actions need continuous updates (e.g. zig-zag steps), drive them here while busy
}
