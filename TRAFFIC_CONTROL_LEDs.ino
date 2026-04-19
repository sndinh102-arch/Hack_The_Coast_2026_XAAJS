// ===================== LED PINS =====================
int Gled = 2;
int Yled = 3;
int Rled = 4;
int Bled = 5; // protected left turn (blue)

// ===================== SENSOR PINS =====================
int lowTrafficSensor = 6;
int highTrafficSensor = 7;
int leftTurnRequest = 8;
int accidentSensor = 9;

// ===================== TIMING =====================
int yellowTime = 3000;

// base cycles
int lowCycle = 10000;
int highCycle = 30000;

// left turn cycles
int leftLow = 5000;
int leftHigh = 15000;

// ===================== SETUP =====================
void setup() {
  Serial.begin(9600);

  pinMode(Gled, OUTPUT);
  pinMode(Yled, OUTPUT);
  pinMode(Rled, OUTPUT);
  pinMode(Bled, OUTPUT);

  pinMode(lowTrafficSensor, INPUT);
  pinMode(highTrafficSensor, INPUT);
  pinMode(leftTurnRequest, INPUT);
  pinMode(accidentSensor, INPUT);
}

// ===================== MAIN LOOP =====================
void loop() {

  // ===== FAILSAFE MODE (ACCIDENT DETECTED) =====
  if (digitalRead(accidentSensor) == HIGH) {

  // turn everything off except red blinking
  digitalWrite(Gled, LOW);
  digitalWrite(Yled, LOW);
  digitalWrite(Bled, LOW);

  while (digitalRead(accidentSensor) == HIGH) {

    digitalWrite(Rled, HIGH);
    delay(500);
    digitalWrite(Rled, LOW);
    delay(500);
  }
}

  bool lowTraffic = digitalRead(lowTrafficSensor);
  bool highTraffic = digitalRead(highTrafficSensor);
  bool leftTurn = digitalRead(leftTurnRequest);

  // reset all lights before decision
  digitalWrite(Gled, LOW);
  digitalWrite(Yled, LOW);
  digitalWrite(Rled, LOW);
  digitalWrite(Bled, LOW);

  // ===================== HIGH TRAFFIC DOMINANT =====================
  if (highTraffic && !lowTraffic) {

    if (leftTurn) {

      // LEFT TURN PHASE (HIGH TRAFFIC)
      digitalWrite(Bled, HIGH);
      delay(leftHigh);
      digitalWrite(Bled, LOW);

      // NORMAL GREEN PHASE
      digitalWrite(Gled, HIGH);
      delay(highCycle);

      // YELLOW TRANSITION
      digitalWrite(Gled, LOW);
      digitalWrite(Yled, HIGH);
      delay(yellowTime);

      digitalWrite(Yled, LOW);
      digitalWrite(Rled, HIGH);
      delay(1000);
      digitalWrite(Rled, LOW);
    }

    else {
      digitalWrite(Gled, HIGH);
      delay(highCycle);

      digitalWrite(Gled, LOW);
      digitalWrite(Yled, HIGH);
      delay(yellowTime);

      digitalWrite(Yled, LOW);
      digitalWrite(Rled, HIGH);
      delay(1000);
      digitalWrite(Rled, LOW);
    }
  }

  // ===================== LOW TRAFFIC DOMINANT =====================
  else if (lowTraffic && !highTraffic) {

    if (leftTurn) {

      // LEFT TURN PHASE (LOW TRAFFIC)
      digitalWrite(Bled, HIGH);
      delay(leftLow);
      digitalWrite(Bled, LOW);

      // NORMAL GREEN PHASE
      digitalWrite(Gled, HIGH);
      delay(lowCycle);

      digitalWrite(Gled, LOW);
      digitalWrite(Yled, HIGH);
      delay(yellowTime);

      digitalWrite(Yled, LOW);
      digitalWrite(Rled, HIGH);
      delay(1000);
      digitalWrite(Rled, LOW);
    }

    else {
      digitalWrite(Gled, HIGH);
      delay(lowCycle);

      digitalWrite(Gled, LOW);
      digitalWrite(Yled, HIGH);
      delay(yellowTime);

      digitalWrite(Yled, LOW);
      digitalWrite(Rled, HIGH);
      delay(1000);
      digitalWrite(Rled, LOW);
    }
  }

  // ===================== NO TRAFFIC =====================
  else {

    // idle blinking yellow mode
    digitalWrite(Gled, LOW);
    digitalWrite(Rled, LOW);
    digitalWrite(Bled, LOW);

    for (int i = 0; i < 5; i++) {
      digitalWrite(Yled, HIGH);
      delay(400);
      digitalWrite(Yled, LOW);
      delay(400);
    }
  }
}