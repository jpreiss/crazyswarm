void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);
}

void  loop() {
  while (!Serial.available());
  int x = Serial.read();
  if (x == '1') {
    digitalWrite(LED_BUILTIN, HIGH);
  }
  else if (x == '0') {
    digitalWrite(LED_BUILTIN, LOW);
  }
}