// esp32_validation.ino
//
// Validation harness for the solar EPS simulation.
//
// Reads synthetic telemetry lines from Serial (115200 baud) in the format:
//
//   timestamp,harvest_w,load_critical_w,load_essential_w,load_deferrable_w
//
// Runs a simplified SoC integrator + 5-state FSM (no predictive forecast),
// then prints one CSV result line per input line:
//
//   timestamp,calculated_soc,fsm_state,load_deferrable_on
//
// Feed it the same scenario CSV the Python simulation uses and diff the two
// outputs to confirm the firmware plumbing matches the reference model.

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static const unsigned long SERIAL_BAUD = 115200;

static const float BATTERY_CAPACITY_WH = 500.0f;  // [Wh] battery pack capacity
static const float SOC0_WH             = 350.0f;  // [Wh] initial state of charge

static const float SOC_CRITICAL_WH     = 50.0f;   // [Wh] below this -> CRITICAL
static const float SOC_DEGRADED_WH     = 100.0f;  // [Wh] below this -> DEGRADED
static const float SOC_SAFE_MODE_WH    = 25.0f;   // [Wh] deep-discharge floor -> SAFE_MODE
static const float SOC_RECOVERY_EXIT_WH = 100.0f; // [Wh] RECOVERY -> NOMINAL threshold

static const float DT_HOURS = 1.0f / 60.0f;       // one simulation step = 1 minute

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

enum FsmState {
  STATE_NOMINAL,    // all loads on
  STATE_DEGRADED,   // shed deferrable
  STATE_CRITICAL,   // shed deferrable (margin tighter than DEGRADED)
  STATE_SAFE_MODE,  // shed everything non-essential (deep-discharge guard)
  STATE_RECOVERY    // staged return toward NOMINAL
};

static float     g_soc_wh   = SOC0_WH;            // current battery SoC [Wh]
static FsmState  g_state    = STATE_NOMINAL;      // current FSM state
static char      g_line[128];                     // Serial input buffer

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Map the FSM state to its log label (matches the Python sim's FSM_STATES).
static const char* stateName(FsmState s) {
  switch (s) {
    case STATE_NOMINAL:   return "NOMINAL";
    case STATE_DEGRADED:  return "DEGRADED";
    case STATE_CRITICAL:  return "CRITICAL";
    case STATE_SAFE_MODE: return "SAFE_MODE";
    case STATE_RECOVERY:  return "RECOVERY";
    default:              return "UNKNOWN";
  }
}

// Evaluate the simplified FSM against the current SoC.
//
// Transitions (hysteresis-free, forecast-free - plumbing test only):
//   SoC <  SOC_SAFE_MODE_WH  -> SAFE_MODE  (deep-discharge guard wins)
//   SoC <  SOC_CRITICAL_WH   -> CRITICAL
//   SoC <  SOC_DEGRADED_WH   -> DEGRADED
//   SAFE_MODE + SoC >= 50    -> RECOVERY
//   RECOVERY  + SoC >= 100   -> NOMINAL (else stays RECOVERY)
//   otherwise                -> NOMINAL
static FsmState evaluateFsm(float soc_wh) {
  if (soc_wh < SOC_SAFE_MODE_WH) {
    return STATE_SAFE_MODE;
  }
  if (g_state == STATE_SAFE_MODE) {
    // Recovering out of SAFE_MODE: go through RECOVERY, not straight to NOMINAL.
    return STATE_RECOVERY;
  }
  if (soc_wh < SOC_CRITICAL_WH) {
    return STATE_CRITICAL;
  }
  if (soc_wh < SOC_DEGRADED_WH) {
    return STATE_DEGRADED;
  }
  if (g_state == STATE_RECOVERY && soc_wh < SOC_RECOVERY_EXIT_WH) {
    // Hold RECOVERY until the battery has climbed back past the exit threshold.
    return STATE_RECOVERY;
  }
  return STATE_NOMINAL;
}

// In this simplified harness, only the Deferrable load is switched.
// It is off in every state except NOMINAL (DEGRADED/CRITICAL shed it per spec).
static int deferrableOn(FsmState s) {
  return (s == STATE_NOMINAL) ? 1 : 0;
}

// Process one telemetry line: integrate SoC, update the FSM, print the result.
static void processLine(char* line) {
  // Parse: timestamp,harvest_w,critical_w,essential_w,deferrable_w
  // -- simple CSV field extraction (in-place, splits on commas) --------------
  char* fields[5];
  int   nfields = 0;
  char* p = line;
  fields[nfields++] = p;
  while (*p && nfields < 5) {
    if (*p == ',') {
      *p = '\0';
      fields[nfields++] = p + 1;
    }
    p++;
  }
  if (nfields < 5) {
    Serial.println("ERR,bad_line");
    return;
  }

  float ts        = atof(fields[0]);
  float harvest_w = atof(fields[1]);
  float load_c    = atof(fields[2]);
  float load_e    = atof(fields[3]);
  float load_d    = atof(fields[4]);

  // -- SoC integration ------------------------------------------------------
  // The Deferrable load is only physically connected when the FSM keeps it on,
  // so compute the *served* load rather than the requested load.
  float total_load_w = load_c + load_e + (deferrableOn(g_state) ? load_d : 0.0f);

  g_soc_wh += (harvest_w - total_load_w) * DT_HOURS;

  // Clip to the physical battery window [0, capacity].
  if (g_soc_wh < 0.0f)          g_soc_wh = 0.0f;
  if (g_soc_wh > BATTERY_CAPACITY_WH) g_soc_wh = BATTERY_CAPACITY_WH;

  // -- FSM update (evaluated against the new SoC) ---------------------------
  g_state = evaluateFsm(g_soc_wh);

  // -- Output CSV -----------------------------------------------------------
  Serial.print(ts, 2);
  Serial.print(',');
  Serial.print(g_soc_wh, 3);
  Serial.print(',');
  Serial.print(stateName(g_state));
  Serial.print(',');
  Serial.println(deferrableOn(g_state));
}

// ---------------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(100);  // give the USB CDC time to enumerate on ESP32-S2/S3/C3

  // Signal readiness so the host-side feeder script knows it can start.
  Serial.print("READY,soc_wh=");
  Serial.print(g_soc_wh, 1);
  Serial.print(",capacity_wh=");
  Serial.println(BATTERY_CAPACITY_WH, 1);
}

void loop() {
  // Read a full line (terminated by '\n'), then process it.
  size_t len = 0;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) {
        g_line[len] = '\0';
        processLine(g_line);
        len = 0;
      }
      // blank lines (from \r\n) are ignored
    } else if (len < sizeof(g_line) - 1) {
      g_line[len++] = c;
    }
  }
}
