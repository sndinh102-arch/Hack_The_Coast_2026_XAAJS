// ===================== NORTH/SOUTH LED PINS =====================
int NS_G = 2;
int NS_Y = 3;
int NS_R = 4;
int NS_B = 5;

// ===================== EAST/WEST LED PINS =====================
int EW_G = 6;
int EW_Y = 7;
int EW_R = 8;
int EW_B = 9;

// ===================== TIMING (ms) =====================
const int BLUE_BLINK_TIME = 3000;
const int BLUE_BLINK_INTERVAL = 250;
const int LEFT_TO_GREEN_DELAY = 1000;
const int YELLOW_TIME = 3000;
const int ALL_RED_TIME = 1000;

// Low traffic timings
const int LEFT_LOW_TIME = 5000;
const int STRAIGHT_LOW_TIME = 10000;

// High traffic timings
const int LEFT_HIGH_TIME = 10000;
const int STRAIGHT_HIGH_TIME = 25000;

// ===================== STATES =====================
// 0 = low left + low straight
// 1 = low left + high straight
// 2 = high left + high straight
const int STATE_0 = 0;
const int STATE_1 = 1;
const int STATE_2 = 2;

int testState = STATE_0;

// ===================== SETUP =====================
void setup() {
  Serial.begin(9600);

  pinMode(NS_G, OUTPUT);
  pinMode(NS_Y, OUTPUT);
  pinMode(NS_R, OUTPUT);
  pinMode(NS_B, OUTPUT);

  pinMode(EW_G, OUTPUT);
  pinMode(EW_Y, OUTPUT);
  pinMode(EW_R, OUTPUT);
  pinMode(EW_B, OUTPUT);

  setAllRed();

  Serial.println("Traffic light test ready");
  Serial.println("Send 0, 1, or 2");
  Serial.println("0 = low left + low straight");
  Serial.println("1 = low left + high straight");
  Serial.println("2 = high left + high straight");
}

// ===================== LOOP =====================
void loop() {
  readTestState();

  // Run North/South using selected state
  runNSPhase(testState);

  readTestState();

  // Run East/West using selected state
  runEWPhase(testState);
}

// ===================== PHASES =====================
void runNSPhase(int state) {
  int leftTime = getLeftTime(state);
  int straightTime = getStraightTime(state);

  // Keep EW solid red the entire time
  digitalWrite(EW_G, LOW);
  digitalWrite(EW_Y, LOW);
  digitalWrite(EW_B, LOW);
  digitalWrite(EW_R, HIGH);

  // NS starts red off because left-turn is active
  digitalWrite(NS_G, LOW);
  digitalWrite(NS_Y, LOW);
  digitalWrite(NS_R, LOW);
  digitalWrite(NS_B, HIGH);

  // solid blue
  smartDelay(leftTime);

  // blinking blue
  unsigned long startBlink = millis();
  while (millis() - startBlink < BLUE_BLINK_TIME) {
    digitalWrite(NS_B, HIGH);
    smartDelay(BLUE_BLINK_INTERVAL);

    digitalWrite(NS_B, LOW);
    smartDelay(BLUE_BLINK_INTERVAL);
  }

  // 1-second no-left-turn buffer
  digitalWrite(NS_B, LOW);
  smartDelay(LEFT_TO_GREEN_DELAY);

  // straight green
  digitalWrite(NS_G, HIGH);
  smartDelay(straightTime);

  // yellow
  digitalWrite(NS_G, LOW);
  digitalWrite(NS_Y, HIGH);
  smartDelay(YELLOW_TIME);

  // back to red
  digitalWrite(NS_Y, LOW);
  digitalWrite(NS_R, HIGH);

  // both red briefly before switching
  digitalWrite(EW_R, HIGH);
  smartDelay(ALL_RED_TIME);
}

void runEWPhase(int state) {
  int leftTime = getLeftTime(state);
  int straightTime = getStraightTime(state);

  // Keep NS solid red the entire time
  digitalWrite(NS_G, LOW);
  digitalWrite(NS_Y, LOW);
  digitalWrite(NS_B, LOW);
  digitalWrite(NS_R, HIGH);

  // EW starts red off because left-turn is active
  digitalWrite(EW_G, LOW);
  digitalWrite(EW_Y, LOW);
  digitalWrite(EW_R, LOW);
  digitalWrite(EW_B, HIGH);

  // solid blue
  smartDelay(leftTime);

  // blinking blue
  unsigned long startBlink = millis();
  while (millis() - startBlink < BLUE_BLINK_TIME) {
    digitalWrite(EW_B, HIGH);
    smartDelay(BLUE_BLINK_INTERVAL);

    digitalWrite(EW_B, LOW);
    smartDelay(BLUE_BLINK_INTERVAL);
  }

  // 1-second no-left-turn buffer
  digitalWrite(EW_B, LOW);
  smartDelay(LEFT_TO_GREEN_DELAY);

  // straight green
  digitalWrite(EW_G, HIGH);
  smartDelay(straightTime);

  // yellow
  digitalWrite(EW_G, LOW);
  digitalWrite(EW_Y, HIGH);
  smartDelay(YELLOW_TIME);

  // back to red
  digitalWrite(EW_Y, LOW);
  digitalWrite(EW_R, HIGH);

  // both red briefly before switching
  digitalWrite(NS_R, HIGH);
  smartDelay(ALL_RED_TIME);
}

// ===================== TIMING LOOKUP =====================
int getLeftTime(int state) {
  switch (state) {
    case STATE_0:
      return LEFT_LOW_TIME;
    case STATE_1:
      return LEFT_LOW_TIME;
    case STATE_2:
      return LEFT_HIGH_TIME;
    default:
      return LEFT_LOW_TIME;
  }
}

int getStraightTime(int state) {
  switch (state) {
    case STATE_0:
      return STRAIGHT_LOW_TIME;
    case STATE_1:
      return STRAIGHT_HIGH_TIME;
    case STATE_2:
      return STRAIGHT_HIGH_TIME;
    default:
      return STRAIGHT_LOW_TIME;
  }
}

// ===================== SERIAL INPUT =====================
void readTestState() {
  if (Serial.available()) {
    char input = Serial.read();
    if (input >= '0' && input <= '2') {
      testState = input - '0';
      Serial.print("State set to: ");
      Serial.println(testState);
    }
  }
}

// ===================== HELPERS =====================
void setAllRed() {
  digitalWrite(NS_G, LOW);
  digitalWrite(NS_Y, LOW);
  digitalWrite(NS_B, LOW);
  digitalWrite(NS_R, HIGH);

  digitalWrite(EW_G, LOW);
  digitalWrite(EW_Y, LOW);
  digitalWrite(EW_B, LOW);
  digitalWrite(EW_R, HIGH);
}

void smartDelay(unsigned long duration) {
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    if (Serial.available()) {
      readTestState();
    }
    delay(10);
  }
}
