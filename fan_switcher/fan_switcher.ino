#include <Adafruit_NeoPixel.h>

// if we go this long without hearing anything over serial, stop.
int constexpr TIMEOUT = 1000;

// neopixels config.
int constexpr PIN_PIXELS = 4;
int constexpr NUM_PIXELS = 115;

// fan config.
int constexpr PIN_FAN = 8;

// state machine.
enum State
{
  Idle,
  Fan,
  White,
  Red,
  Green,
  Blue
};

Adafruit_NeoPixel pixels(NUM_PIXELS, PIN_PIXELS, NEO_GRB + NEO_KHZ800);

static unsigned long last_data = 0;
static State state = State::Idle;
static uint32_t color = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);

  pixels.begin();
  // pixels have their own memory, but we want resetting the Arduino to reset them.
  pixels.clear();
  pixels.show();

  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(PIN_FAN, OUTPUT);

  Serial.println("Arduino is ready.");
}

// protocol:
// if character 'f':
//     next byte is '1' or '0' to change state
// if character 'l':
//     next 4 bytes are w, r, g, b values


void  loop() {
  unsigned long m =  millis();
  while (!Serial.available()) {
    m = millis();
    // Timeout. TODO: Handle overflow.
    if (last_data > 0 && m - last_data > TIMEOUT) {
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(8, LOW);
      pixels.clear();
      state = State::Idle;
    }
  }
  last_data = m;
  int x = Serial.read();
  switch (state) {
    case Idle:
      if (x == 'f') {
        // // Serial.println("fan");
        state = Fan;
      }
      if (x == 'l') {
        // Serial.println("light");
        state = White;
      }
      // otherwise, do not understand
      break;
    case Fan:
      if (x == '1') {
        // Serial.println("on");
        digitalWrite(LED_BUILTIN, HIGH);
        digitalWrite(PIN_FAN, HIGH);
      }
      else if (x == '0') {
        // Serial.println("off");
        digitalWrite(LED_BUILTIN, LOW);
        digitalWrite(PIN_FAN, LOW);
      }
      state = Idle;
      break;
    case White:
    case Red:
    case Green:
      color = (color << 8) | x;
      state = (int)state + 1;
      break;
    case Blue:
      // Serial.println("b");
      color = (color << 8) | x;
      pixels.fill(color);
      pixels.show();
      state = State::Idle;
      break;
  }
}