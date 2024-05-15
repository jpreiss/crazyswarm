int constexpr TIMEOUT = 1000;

static unsigned long last_data = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(8, OUTPUT);
}

void  loop() {
  unsigned long m =  millis();
  while (!Serial.available()) {
    m = millis();
    // TODO: Handle overflow.
    if (last_data > 0 && m - last_data > TIMEOUT) {
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(8, LOW);
    }
  }
  last_data = m;
  int x = Serial.read();
  if (x == '1') {
    digitalWrite(LED_BUILTIN, HIGH);
    digitalWrite(8, HIGH);
  }
  else if (x == '0') {
    digitalWrite(LED_BUILTIN, LOW);
    digitalWrite(8, LOW);
  }
}