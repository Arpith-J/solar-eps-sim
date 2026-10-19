// esp32_full.ino
// ==============
//
// Full port of the Python energy manager (study3_energy_budgeting.py,
// PREDICT_5 rung) to the ESP32.  This sketch implements the same seasonal
// forecaster, margin-based decision and five-state FSM as the Python
// simulation so the board can be validated against the full Python model
// using the same comparison harness (compare_esp32.py).
//
// ORDERING (requirement 9 -- keep in sync with the Python reference):
//   FSM state is decided from the PREVIOUS step's SoC and margin.
//   The logged SoC is AFTER the update.  This matches the Python reference
//   ordering used in reference_trace.csv.  Concretely (mirrors
//   simulate_trial, study3 lines 1662-1673):
//     1. update the forecaster with this minute's measured harvest
//     2. compute the conservative margin from the PREVIOUS SoC
//     3. decide + debounce the FSM state
//     4. map state -> load set (payload gate may still switch D off)
//     5. integrate the SoC with the served load
//     6. log (timestamp, SoC AFTER update, state, payload decision)
//
// SERIAL PROTOCOL (identical to esp32_validation.ino):
//   input : "{timestamp},{harvest_w},{load_c},{load_e},{load_d}\n"
//   output: "{timestamp},{soc_wh},{fsm_state},{payload_on}\n"
//   boot  : "READY,soc_wh=350.0,capacity_wh=500.0"
//
// PYTHON ORIGIN MAP (line numbers = study3_energy_budgeting.py):
//   constants ............ lines 109-239
//   clear_sky_shape ...... lines 255-272   -> shapeAt()
//   Geometry.dark_hours .. lines 498-512   -> darkHoursAt()
//   SeasonalForecaster ... lines 838-1075  -> struct Forecaster
//   PredictiveFSM ........ lines 1242-1511 -> stepManager() + helpers
//     _goto .............. lines 1320-1329
//     _debounced ......... lines 1331-1354
//     step ............... lines 1357-1460
//     _recovery_target ... lines 1463-1489
//     _payload_gate ...... lines 1492-1511
//
// SIMPLIFICATIONS vs the Python study (per the firmware spec):
//   * planning horizon fixed at 90 min (Python: min(480, minutes to eclipse
//     end), lines 1366 / Geometry.horizon 482-496)
//   * the ephemeris eclipse window is FIXED (see ECLIPSE_START_MIN) because
//     the serial protocol does not carry the scenario's randomised offset
//   * RECOVERY staging per spec: C -> C+E -> C+E+D, 15 min per stage,
//     payload SoC threshold 55 % (Python: 10/15 min holds, payload 35 %)
//   * the e_shed_latched term (Python 1372, 1441-1446) is omitted; with this
//     load mapping it never changes the enable decision, only the margin's
//     base-load booking
//   * payload booked over the whole horizon in f_load (Python books only the
//     lit minutes inside the horizon, lines 1383-1392)
//   * plot-only traces (trace_pred/htot/conf, MAE accumulators) omitted
//
// MEMORY: ~7.6 KB static (pending rings dominate at 6.7 KB), well under the
// 30 KB budget.  Forecaster state is double (ESP32 software FP); FSM, SoC
// and protocol paths are float32.

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Forecaster types -- MUST precede all function definitions.
//
// Arduino's preprocessor auto-generates function prototypes near the top of
// the sketch; a custom struct used in a prototype must already be declared
// there or the build fails with "'Forecaster' was not declared in this
// scope" at fcInit / fcPredict / fcForecastWh / fcUpdate.  The tracked-lead
// count, the BandLevel enum, the pending-ring sizes and the complete
// struct Forecaster definition are therefore placed here, before any
// constants, globals or functions.  (Moved verbatim -- no logic changes.)
// ---------------------------------------------------------------------------

// -- tracked leads (study3 line 232) ---------------------------------------
static const int N_LEADS = 8;

enum BandLevel { BAND_HIGH = 0, BAND_MEDIUM, BAND_LOW };

// Pending-forecast ring buffers: lead k needs k+1 slots; 839 = sum(lead + 1)
// over the 8 tracked leads (study3 line 905 holds the same data in a dict).
static const int PEND_SLOTS = 839;

struct Forecaster {
  // -- EWMAs (lines 893-896) -----------------------------------------------
  double level;        // seasonal (orbit-averaged) clearness, init 1.0
  double fast;         // fast clearness EWMA, tau = 20 min
  double slow;         // medium clearness EWMA, tau = 90 min
  double dev;          // band-pass anomaly (fast - slow), persisted for corr

  // -- per-lead learned tables (lines 898-903) -----------------------------
  double gain[N_LEADS];     // residual persistence, clipped to [0, 1]
  double bias[N_LEADS];     // systematic error-feedback correction
  double cov[N_LEADS];      // EWMA of cov(dev_issue, dev_now)
  double var[N_LEADS];      // EWMA of var(dev_issue)
  double errEwma[N_LEADS];  // EWMA of signed error
  double absEwma[N_LEADS];  // EWMA of |error| -> sigma at each lead

  // -- confidence (lines 908-910) ------------------------------------------
  bool seen;                 // any sunlit learning sample yet?
  BandLevel band;

  // -- EWMA gains, alpha = 1 - exp(-dt / tau) (lines 888-892) ---------------
  double aLevel, aDev, aSlow, aSys, aReg;

  // -- pending forecasts: (fc in W, dev_issue in clearness units) ----------
  float pendFc[PEND_SLOTS];
  float pendDev[PEND_SLOTS];
};

// ---------------------------------------------------------------------------
// Serial protocol (requirement 1 / 10)
// ---------------------------------------------------------------------------

static const unsigned long SERIAL_BAUD = 115200;
static char g_line[128];

// ---------------------------------------------------------------------------
// Constants -- ported from study3_energy_budgeting.py (requirement 2)
// ---------------------------------------------------------------------------

// -- battery / bus (study3 lines 164-173) ----------------------------------
static const float BAT_CAPACITY_WH  = 500.0f;  // line 168
static const float BAT_SOC0         = 350.0f;  // spec: initial SoC (validation parity)
static const float ETA_CHG          = 0.95f;   // line 170
static const float ETA_DIS          = 0.95f;   // line 171
static const float DEEP_DISCH_SOC   = 0.10f;   // line 173 -> hard SoC guard

// -- time base (lines 109-110) ---------------------------------------------
static const float DT_MIN = 1.0f;              // line 109, [min]
static const float DT_H   = DT_MIN / 60.0f;    // line 110, [h]

// -- load tiers (lines 176-180) --------------------------------------------
static const float W_CRITICAL   = 20.0f;       // line 176, never shed
static const float W_ESSENTIAL  = 15.0f;       // line 177
static const float W_DEFERRABLE = 30.0f;       // line 178, payload

// -- reserve (lines 185-186 and RESERVE_FRAC_DEGRADED line 187) -------------
static const float RESERVE_BASE_FRAC = 0.20f;  // line 185
static const float RESERVE_LOW_FRAC  = 0.30f;  // line 186 (LOW confidence + DEGRADED)

// -- margin thresholds (lines 188-190, requirement 2) ----------------------
static const float MARGIN_SHED_DEFER_WH = 0.0f;     // line 188 -> DEGRADED
static const float MARGIN_SHED_ESS_WH   = -50.0f;   // line 189 -> CRITICAL
static const float MARGIN_CRITICAL_WH   = -100.0f;  // line 190 -> SAFE_MODE

// -- payload gate (lines 196-197) ------------------------------------------
static const float PAYLOAD_GATE_LEAD_MIN = 30.0f;   // line 196
static const float PAYLOAD_GATE_HOLD_MIN = 45.0f;   // line 197

// -- FSM debounce / dwell (lines 200-201) ----------------------------------
static const int   FSM_DEBOUNCE_MIN  = 3;           // line 200, relaxation debounce
static const int   FSM_MIN_DWELL_MIN = 10;          // line 201, min dwell before relax

// -- recovery (spec requirement 2; Python values were lines 203-207) -------
static const float FSM_SAFE_SOC_CLEAR   = 0.30f;    // line 205 SAFE->RECOVERY SoC
static const float REC_SOC_ESS          = 0.30f;    // spec: SoC > 30 % for essential
static const float REC_SOC_PAYLOAD      = 0.55f;    // spec: SoC > 55 % for payload
static const float REC_HOLD_MIN         = 15.0f;    // spec: 15 min hold per stage

// -- forecaster (lines 165, 232-239) ---------------------------------------
static const float PANEL_PEAK_W = 155.0f;           // line 165
static const float RESID_TAU_MIN      = 20.0f;      // line 233, fast residual EWMA
static const float RESID_SLOW_TAU_MIN = 90.0f;      // line 234, slow residual EWMA
static const float RESID_SYS_TAU_MIN  = 120.0f;     // line 235, systematic bias EWMA
static const float RESID_REG_TAU_MIN  = 180.0f;     // line 236, gain regression EWMA
static const float ORBIT_PERIOD_MIN   = 1440.0f;    // line 119, seasonal EWMA tau
static const float CONF_HIGH_MAE = 0.12f;           // line 237
static const float CONF_MED_MAE  = 0.28f;           // line 238

// -- fixed planning horizon (spec requirement 4) ---------------------------
static const int HORIZON_MIN = 90;                  // replaces Python's dynamic
                                                    // horizon (lines 1366, 482-496)

// -- ephemeris (lines 119-124) ---------------------------------------------
// clear_sky_shape parameters: the window opens at sunrise, an 18 h daylight
// arc with a 360 min eclipse inside it, then orbital night.  The Python
// scenario randomises the eclipse start over [0, 720] min (line 124); the
// serial protocol cannot convey it, so the firmware uses the midpoint of
// that range as its *believed* ephemeris.  The host-side reference must
// mirror this constant.
static const float DAYLIGHT_MIN       = 1080.0f;    // line 120, 18 h sunlit arc
static const float ECLIPSE_LEN_MIN    = 360.0f;     // line 121, 6 h eclipse
static const float ECLIPSE_START_MIN  = 360.0f;     // midpoint of line 124 range
static const float ORBIT_WRAP_MIN     = 1440.0f;    // synthetic orbit == 1 day
static const float LIT_GEOM_EPS       = 0.0f;       // Geometry lit test (line 413)
static const float LIT_FC_EPS         = 0.05f;      // forecaster lit test (line 999)

// -- tracked leads / bands (lines 232, 239) --------------------------------
static const int   LEADS[N_LEADS] = {1, 5, 15, 30, 60, 120, 240, 360};  // line 232
static const float Z_BY_BAND[3]    = {0.50f, 1.00f, 1.50f};             // line 239

// USE_RESIDUAL = true -> PREDICT_5 (rung 4: EWMA + residual correction,
// Python line 1576).  false would give PREDICT_4 (line 1575).
static const bool USE_RESIDUAL = true;

// ---------------------------------------------------------------------------
// FSM state encoding (lines 1236-1239)
// ---------------------------------------------------------------------------

enum FsmState {
  ST_NOMINAL = 0,
  ST_DEGRADED,
  ST_CRITICAL,
  ST_SAFE_MODE,
  ST_RECOVERY
};

// Escalation order: protection is instantaneous, recovery is debounced
// (line 1239).  Indexed by FsmState.
static const int FSM_SEVERITY[5] = {0, 2, 3, 4, 1};

static const char* stateName(FsmState s) {
  switch (s) {
    case ST_NOMINAL:   return "NOMINAL";
    case ST_DEGRADED:  return "DEGRADED";
    case ST_CRITICAL:  return "CRITICAL";
    case ST_SAFE_MODE: return "SAFE_MODE";
    case ST_RECOVERY:  return "RECOVERY";
    default:           return "UNKNOWN";
  }
}

// ---------------------------------------------------------------------------
// Ephemeris -- port of clear_sky_shape (study3 lines 255-272)
// ---------------------------------------------------------------------------

// Normalised clear-sky geometry in [0, 1] at an absolute minute index.  The
// Python ephemeris is orbit-periodic (the tiled shape_ext, line 585), so the
// firmware wraps the minute modulo 1440; this is bit-equivalent to Python's
// tiled array for every index the 1440-step window plus the longest lead can
// reach (max 1439 + 360 < 1921).
static float shapeAt(long minute) {
  long tt = minute % (long)ORBIT_WRAP_MIN;
  if (tt < 0) tt += (long)ORBIT_WRAP_MIN;
  float x = (float)tt / DAYLIGHT_MIN;                       // sunrise = 0
  float s = (x >= 0.0f && x <= 1.0f) ? sinf(PI * x) : 0.0f; // line 269
  if ((float)tt >= ECLIPSE_START_MIN &&
      (float)tt < ECLIPSE_START_MIN + ECLIPSE_LEN_MIN) {
    s = 0.0f;                                               // line 271
  }
  if (s < 0.0f) s = 0.0f;
  if (s > 1.0f) s = 1.0f;                                   // line 272
  return s;
}

// Dark-run bookkeeping uses shape > 0.0 (Geometry.__init__, line 413); the
// forecaster uses shape > 0.05 (line 999).  Both thresholds are reproduced.
static bool geomLitAt(long minute) { return shapeAt(minute) > LIT_GEOM_EPS; }

// Port of Geometry.dark_hours (study3 lines 498-512): hours of continuous
// eclipse/night immediately ahead of minute i, scanning the periodic
// ephemeris.  hmax = PLAN_HORIZON_MIN_MAX / DT_MIN = 480 (line 455).
static float darkHoursAt(long i) {
  const long HMAX = 480;
  long k;
  if (geomLitAt(i)) {
    long s = 0;                                    // sun_run[i] (lines 506-508)
    while (s < HMAX && geomLitAt(i + s)) s++;
    if (s >= HMAX) return 0.0f;                    // line 508-509: dark too far
    k = i + s;                                     // first dark minute
  } else {
    k = i;                                         // line 510-511
  }
  long dark = 0;                                   // dark_run[k] (line 512)
  while (!geomLitAt(k + dark)) dark++;
  return (float)dark * DT_H;
}

// ---------------------------------------------------------------------------
// SeasonalForecaster -- port of study3 lines 838-1075
//
// Seasonal (orbit-matched) EWMA of the clearness plus a band-pass residual
// pair, per-lead least-squares persistence gains, a systematic error-feedback
// bias and a confidence band used for the conservative margin bound:
//     hhat = PEAK * max(0, level - z(conf)*sigma(k) + gain(k)*dev + bias(k))
//            * shape(t+k)                                 (lines 854-855, 967)
//
// Pending forecasts are stored in per-lead ring buffers (lead k needs k+1
// slots): a forecast issued at minute s with lead k is realised at s+k.
// Python uses a dict keyed by (lead, target step) (line 905); the rings hold
// the same information in 6.7 KB.
//
// The BandLevel enum, N_LEADS, PEND_SLOTS and struct Forecaster itself live
// at the top of this file so Arduino's auto-prototyper sees them before it
// emits the fcInit / fcPredict / fcForecastWh / fcUpdate prototypes.
// ---------------------------------------------------------------------------

static int PEND_OFF[N_LEADS + 1];    // ring base offsets per lead

static Forecaster g_fc;

// Linear interpolation across the tracked leads with endpoint clamping --
// equivalent of np.interp (used by _corr_at/_bias_at/_sigma_at, lines 929-943).
static double interpLead(const double* v, double lead) {
  if (lead <= (double)LEADS[0]) return v[0];
  if (lead >= (double)LEADS[N_LEADS - 1]) return v[N_LEADS - 1];
  for (int l = 1; l < N_LEADS; l++) {
    if (lead <= (double)LEADS[l]) {
      double f = (lead - (double)LEADS[l - 1]) /
                 (double)(LEADS[l] - LEADS[l - 1]);
      return v[l - 1] + f * (v[l] - v[l - 1]);
    }
  }
  return v[N_LEADS - 1];
}

// Error statistics adapt faster at short leads (lines 1069-1071).
static double alphaFastFor(int lead) {
  return 1.0 - exp(-1.0 / ((double)RESID_TAU_MIN * (1.0 + 0.5 * lead / 60.0)));
}

static void fcInit(Forecaster* f) {
  f->level = 1.0;                       // line 893
  f->fast  = 1.0;                       // line 894
  f->slow  = 1.0;                       // line 895
  f->dev   = 0.0;                       // line 896
  for (int l = 0; l < N_LEADS; l++) {
    f->gain[l] = f->bias[l] = f->cov[l] = f->var[l] = 0.0;   // lines 898-901
    f->errEwma[l] = f->absEwma[l] = 0.0;                     // lines 902-903
  }
  f->seen = false;                      // line 908
  f->band = BAND_LOW;                   // line 910
  // alpha = 1 - exp(-dt / tau) (lines 888-892)
  f->aLevel = 1.0 - exp(-1.0 / (double)ORBIT_PERIOD_MIN);
  f->aDev   = 1.0 - exp(-1.0 / (double)RESID_TAU_MIN);
  f->aSlow  = 1.0 - exp(-1.0 / (double)RESID_SLOW_TAU_MIN);
  f->aSys   = 1.0 - exp(-1.0 / (double)RESID_SYS_TAU_MIN);
  f->aReg   = 1.0 - exp(-1.0 / (double)RESID_REG_TAU_MIN);
  for (int s = 0; s < PEND_SLOTS; s++) {
    f->pendFc[s] = NAN;                // empty ring slots
    f->pendDev[s] = 0.0;
  }
}

// Point forecast of array power at absolute minute t + lead (W) -- port of
// predict(), lines 953-968.  conservative=true deflates the level by
// z(conf) * sigma(lead) (lines 965-966); the residual gain and bias are
// always present when USE_RESIDUAL (lines 967, 929-939).  sigma is never
// gated by USE_RESIDUAL (line 941-943).
static double fcPredict(const Forecaster* f, long t, double leadMin, bool conservative) {
  long j = t + (long)(leadMin / DT_MIN + 0.5);   // line 960
  double shapeF = shapeAt(j);                    // wraps; see shapeAt() note
  if (shapeF <= 0.0) return 0.0;                 // lines 962-963
  double lvl = f->level;                         // line 964
  if (conservative) {
    lvl -= (double)Z_BY_BAND[f->band] * interpLead(f->absEwma, leadMin);
  }
  double corr = 0.0, biasC = 0.0;
  if (USE_RESIDUAL) {
    corr  = f->dev * interpLead(f->gain, leadMin);   // _corr_at, lines 931-933
    biasC = interpLead(f->bias, leadMin);            // _bias_at, lines 937-939
  }
  double val = (double)PANEL_PEAK_W * (lvl + corr + biasC) * shapeF;  // line 967
  return val > 0.0 ? val : 0.0;                  // line 968
}

// Forecast energy over minutes t+1 .. t+kmax (Wh) -- the per-step loop form
// of predict_block().sum() * DT_H (lines 970-987, 1368).
static double fcForecastWh(const Forecaster* f, long t, int kmax, bool conservative) {
  double wh = 0.0;
  for (int k = 1; k <= kmax; k++) {
    wh += fcPredict(f, t, (double)k, conservative) * DT_H;
  }
  return wh;
}

// Online learning step -- port of update(), lines 990-1067.  All learning is
// sunlit-only (line 999) so the confidence band is not washed out by the
// eclipse's zero harvest (comment lines 994-997).
static void fcUpdate(Forecaster* f, long t, double shapeNow, float harvestW) {
  bool lit = shapeNow > LIT_FC_EPS;                       // line 999
  double clr = harvestW / (double)PANEL_PEAK_W;           // line 1000

  // -- seasonal level + band-pass pair (lines 1009-1013) -------------------
  if (lit) {
    f->level += f->aLevel * (clr - f->level);
    if (f->level < 0.0) f->level = 0.0;
    if (f->level > 1.5) f->level = 1.5;                   // line 1011
    f->fast  += f->aDev * (clr - f->fast);
    f->slow  += f->aSlow * (clr - f->slow);
  }
  double devNow = lit ? (f->fast - f->slow) : 0.0;        // line 1014

  // -- realise pending forecasts that fall due now (lines 1022-1046) -------
  for (int l = 0; l < N_LEADS; l++) {
    int k = LEADS[l];
    if (t < k) continue;                                  // not yet issued
    int slot = PEND_OFF[l] + (int)((t - k) % (k + 1));
    float fc = f->pendFc[slot];
    if (isnan(fc)) continue;                              // empty slot
    double devIssue = f->pendDev[slot];
    f->pendFc[slot] = NAN;                                // Python dict.pop
    if (!lit) continue;                                   // lines 1028-1029
    // err is only consumed on sunlit minutes (line 1027 computes it before
    // the dark check; the value is unused there, so the reorder is exact)
    double err = clr - (double)fc / (double)PANEL_PEAK_W; // line 1027
    f->dev = devNow;                                      // line 1030

    // (1) least-squares persistence gain (lines 1032-1036)
    if (devIssue != 0.0) {
      f->cov[l] += f->aReg * (devIssue * devNow - f->cov[l]);
      f->var[l] += f->aReg * (devIssue * devIssue - f->var[l]);
      if (f->var[l] > 1e-6) {
        double g = f->cov[l] / f->var[l];
        f->gain[l] = g < 0.0 ? 0.0 : (g > 1.0 ? 1.0 : g);
      }
    }
    // (2) error statistics -> confidence band (lines 1038-1040)
    double ae = alphaFastFor(k);
    f->errEwma[l] += ae * (err - f->errEwma[l]);
    f->absEwma[l] += ae * (fabs(err) - f->absEwma[l]);
    // (3) systematic error-feedback bias (lines 1042-1043)
    if (USE_RESIDUAL) {
      f->bias[l] += f->aSys * err;
    }
    f->seen = true;                                       // line 1044
    // MAE accumulators (lines 1045-1046) are plot-only and omitted.
  }

  // -- confidence band (lines 1050-1059) -----------------------------------
  if (f->seen) {
    double ebar = 0.0;
    for (int l = 0; l < N_LEADS; l++) {
      if (f->absEwma[l] > ebar) ebar = f->absEwma[l];     // line 1051
    }
    if (ebar < CONF_HIGH_MAE)      f->band = BAND_HIGH;   // line 1052
    else if (ebar < CONF_MED_MAE)  f->band = BAND_MEDIUM; // line 1054
    else                           f->band = BAND_LOW;    // line 1056
  } else {
    f->band = BAND_LOW;                                   // line 1059
  }

  // -- register forecasts now in flight (lines 1062-1063) ------------------
  for (int l = 0; l < N_LEADS; l++) {
    int k = LEADS[l];
    int slot = PEND_OFF[l] + (int)(t % (k + 1));
    f->pendFc[slot]  = (float)fcPredict(f, t, (double)k, false);
    f->pendDev[slot] = (float)devNow;
  }
  // trace_pred / trace_conf (lines 1066-1067) are plot-only and omitted.
}

// ---------------------------------------------------------------------------
// Manager state -- port of PredictiveFSM.reset (lines 1281-1309)
// ---------------------------------------------------------------------------

static FsmState g_state   = ST_NOMINAL;   // line 1292
static int      g_stateT  = 0;            // minutes in current state (1293)
static FsmState g_cand    = ST_NOMINAL;   // debounce candidate (1294)
static int      g_candT   = 0;            // candidate age (1295)
static int      g_recStage = 0;           // RECOVERY stage (1296)
static float    g_recEssHold = 0.0f;      // stage 0->1 hold timer (1297)
static float    g_recDefHold = 0.0f;      // stage 1->2 hold timer (1298)
static int      g_recBad  = 0;            // consecutive negative-margin min (1299)
static bool     g_prevDOn = false;        // payload commitment for margin (1301)
static float    g_soc     = BAT_SOC0;     // battery SoC [Wh], AFTER update

// Hardware trip input (requirement 5): not exercised by this validation
// harness.  A future fault-detection module (study4_fault_response) sets the
// flag from its trip detector / ISR; while set, the FSM is forced to
// SAFE_MODE.
static volatile bool g_hwTrip = false;

void setHardwareTrip(bool tripped) { g_hwTrip = tripped; }

// ---------------------------------------------------------------------------
// FSM plumbing -- _goto (lines 1320-1329) and _debounced (lines 1331-1354)
// ---------------------------------------------------------------------------

static void fsmGoto(FsmState newS) {
  if (newS != g_state) {                    // lines 1321-1325
    g_state  = newS;
    g_stateT = 0;
    g_cand   = newS;
    g_candT  = 0;
  }
  if (newS != ST_RECOVERY) {                // lines 1326-1329
    g_recStage    = 0;
    g_recEssHold  = 0.0f;
    g_recDefHold  = 0.0f;
  }
}

// Escalation is immediate (protection never waits); relaxation needs the
// condition to persist FSM_DEBOUNCE_MIN minutes AND FSM_MIN_DWELL_MIN
// minutes of dwell in the current state.
static void fsmDebounced(FsmState target) {
  if (target == g_state) {                  // lines 1341-1343
    g_cand  = target;
    g_candT = 0;
    return;
  }
  if (FSM_SEVERITY[target] > FSM_SEVERITY[g_state]) {  // lines 1344-1346
    fsmGoto(target);
    return;
  }
  if (target == g_cand) {                   // lines 1347-1350
    g_candT += 1;
  } else {
    g_cand  = target;
    g_candT = 1;
  }
  if (g_candT >= FSM_DEBOUNCE_MIN && g_stateT >= FSM_MIN_DWELL_MIN) {  // 1351
    fsmGoto(target);
  }
}

// ---------------------------------------------------------------------------
// RECOVERY -- staged return, port of _recovery_target (lines 1463-1489) with
// the spec's parameters (requirement 2): essential needs SoC > 30 %, payload
// SoC > 55 %, REC_HOLD_MIN per stage.  Stages map to the load set as
// C (stage 0) -> C+E (stage 1) -> C+E+D (stage 2) (requirement 6).
//
// Note on hold-timer resets: Python only clears rec_ess_hold / rec_def_hold
// in _goto when LEAVING recovery (lines 1326-1329), not when the stage
// counter is reset by a bad minute (line 1474).  Reproduced verbatim.
// ---------------------------------------------------------------------------

static FsmState recoveryTarget(float socf, float margin) {
  if (margin < 0.0f) {                      // lines 1468-1471
    g_recBad += 1;
  } else {
    g_recBad = 0;
  }
  bool sustainedBad = (g_recBad >= FSM_DEBOUNCE_MIN);   // line 1472

  if (socf < REC_SOC_ESS || sustainedBad) { // lines 1473-1475
    g_recStage = 0;
    return ST_RECOVERY;
  }
  if (g_recStage == 0) {                    // lines 1476-1479, 15 min per spec
    g_recEssHold += DT_MIN;
    if (g_recEssHold >= REC_HOLD_MIN) {
      g_recStage = 1;
    }
  }
  if (g_recStage >= 1) {                    // lines 1483-1488
    if (socf < REC_SOC_PAYLOAD || margin < 0.0f) {
      return ST_RECOVERY;
    }
    g_recDefHold += DT_MIN;
    if (g_recDefHold >= REC_HOLD_MIN) {
      g_recStage = 2;
    }
  }
  return (g_recStage >= 2) ? ST_NOMINAL : ST_RECOVERY;  // line 1489
}

// ---------------------------------------------------------------------------
// Payload gate -- port of _payload_gate (lines 1492-1511)
//
// (a) hard safety: SoC must already cover Critical + Essential through the
//     coming dark window, plus the reserve;
// (b) opportunism: the conservative 30-min forecast surplus pays for the
//     payload, or the battery holds PAYLOAD_GATE_HOLD_MIN of extra payload
//     energy above (a).
// ---------------------------------------------------------------------------

static bool payloadGate(long t, float soc, float reserve) {
  float baseW = W_CRITICAL + W_ESSENTIAL;                       // line 1503
  float darkReq = baseW * darkHoursAt(t) + reserve;             // lines 1502-1504
  if (soc < darkReq) return false;                              // lines 1505-1506

  int k = (int)(PAYLOAD_GATE_LEAD_MIN / DT_MIN);                // line 1507, = 30
  double surplus = fcForecastWh(&g_fc, t, k, true)              // pred_block[:k]
                   - (double)(baseW + W_DEFERRABLE) * (k * DT_H);  // line 1508
  if (surplus >= 0.0) return true;                              // lines 1509-1510
  return soc >= darkReq + W_DEFERRABLE * PAYLOAD_GATE_HOLD_MIN / 60.0f;  // 1511
}

// ---------------------------------------------------------------------------
// One control step -- port of PredictiveFSM.step (lines 1357-1460) plus the
// SoC integration (spec requirement 7; Python bus_step lines 781-831).
//
// Called once per received telemetry line.  soc_prev is the SoC AFTER the
// previous step's update; the decision phase below never sees the new SoC.
// ---------------------------------------------------------------------------

static void stepManager(long t, float harvestW) {
  float   socPrev = g_soc;                                // ctx.soc_wh (1359)
  double  socf    = (double)socPrev / (double)BAT_CAPACITY_WH;
  double  shapeNow = shapeAt(t);

  // --- forecast update: learn from this minute's measurement (line 1363) --
  fcUpdate(&g_fc, t, shapeNow, harvestW);

  // --- planning horizon + forecasts (lines 1366-1368; H fixed per spec) ---
  double fHarvestWh = fcForecastWh(&g_fc, t, HORIZON_MIN, true);
  // Load forecast per spec requirement 4: the scheduled base load runs for
  // the whole horizon and the payload is booked only if it was commanded at
  // the previous step (g_prevDOn, Python self._d_on lines 1301/1391/1452).
  // Python additionally books the payload over the lit minutes only and
  // drops Essential while its shed-latch is set (lines 1372-1392); the spec
  // fixes the formula to C + E + D_if_committed, reproduced here.
  float fLoadWh = (W_CRITICAL + W_ESSENTIAL +
                   (g_prevDOn ? W_DEFERRABLE : 0.0f)) *
                  ((float)HORIZON_MIN / 60.0f);

  // --- reserve + margin (lines 1397-1402) ---------------------------------
  float reserve = BAT_CAPACITY_WH *
                  ((g_fc.band == BAND_LOW) ? RESERVE_LOW_FRAC
                                            : RESERVE_BASE_FRAC);   // 1399
  if (g_state == ST_DEGRADED) {                             // lines 1400-1401
    float degRes = BAT_CAPACITY_WH * RESERVE_LOW_FRAC;
    if (degRes > reserve) reserve = degRes;
  }
  double margin = (double)socPrev + fHarvestWh - (double)fLoadWh - (double)reserve;

  // --- margin-driven target state (lines 1405-1419) ----------------------
  FsmState target = g_state;
  if (g_state == ST_SAFE_MODE) {                            // lines 1407-1408
    target = (socf >= FSM_SAFE_SOC_CLEAR && margin >= 0.0) ? ST_RECOVERY
                                                           : ST_SAFE_MODE;
  } else if (g_state == ST_RECOVERY) {                      // lines 1409-1410
    target = recoveryTarget((float)socf, (float)margin);
  } else if (margin < MARGIN_CRITICAL_WH) {                 // lines 1412-1413
    target = ST_SAFE_MODE;
  } else if (margin < MARGIN_SHED_ESS_WH) {                 // lines 1414-1415
    target = ST_CRITICAL;
  } else if (margin < MARGIN_SHED_DEFER_WH) {               // lines 1416-1417
    target = ST_DEGRADED;
  } else {                                                  // lines 1418-1419
    target = ST_NOMINAL;
  }

  // Hard SoC floor -> SAFE_MODE (deep-discharge guard, lines 1421-1423)
  if (socf < DEEP_DISCH_SOC && g_state != ST_SAFE_MODE) {
    target = ST_SAFE_MODE;
  }
  // Hardware trip (requirement 5): protection overrides everything and, since
  // SAFE_MODE has the highest severity, the escalation is immediate.
  if (g_hwTrip) {
    target = ST_SAFE_MODE;
  }
  fsmDebounced(target);                                     // line 1424

  // --- load set implied by the state (lines 1426-1441) -------------------
  // Spec requirement 6:
  //   NOMINAL   : C + E + D
  //   DEGRADED  : C + E
  //   CRITICAL / SAFE_MODE : C only
  //   RECOVERY  : staged C -> C+E -> C+E+D (15 min per stage)
  bool eOn = true, dOn = false;
  switch (g_state) {
    case ST_NOMINAL:
      eOn = true;  dOn = true;                               // lines 1429-1431
      break;
    case ST_DEGRADED:
      eOn = true;  dOn = false;
      break;
    case ST_CRITICAL:
    case ST_SAFE_MODE:
      eOn = false; dOn = false;                              // lines 1432-1434
      break;
    case ST_RECOVERY:
      eOn = (g_recStage >= 1);                               // C+E from stage 1
      dOn = (g_recStage >= 2);                               // D from stage 2
      break;
  }
  // Python's e_shed_latched (lines 1441-1446) is omitted per spec; with this
  // load mapping it never changes the enable decision, only margin's base-w.

  // --- deferrable scheduler: energy-rich windows only (lines 1448-1450) --
  if (dOn) {
    dOn = payloadGate(t, socPrev, (float)reserve);
  }
  g_prevDOn = dOn;                                           // line 1452

  // --- SoC integration (spec requirement 7) ------------------------------
  // served_load = sum of the loads the decision just allowed.  Harvest above
  // the load charges the battery through ETA_CHG; a deficit is drawn from
  // the battery through ETA_DIS (bus_step behaviour, lines 794-828).
  float servedW = W_CRITICAL + (eOn ? W_ESSENTIAL : 0.0f) +
                              (dOn ? W_DEFERRABLE : 0.0f);
  if (harvestW > servedW) {
    g_soc += (harvestW - servedW) * ETA_CHG / 60.0f;
  } else {
    g_soc += (harvestW - servedW) / ETA_DIS / 60.0f;
  }
  if (g_soc < 0.0f)             g_soc = 0.0f;                // clip [0, cap]
  if (g_soc > BAT_CAPACITY_WH)  g_soc = BAT_CAPACITY_WH;

  g_stateT += 1;                                             // line 1453

  // --- output CSV (requirement 1): timestamp, SoC AFTER update, state,
  //     payload decision applied this step --------------------------------
  Serial.print((float)t, 2);
  Serial.print(',');
  Serial.print(g_soc, 3);
  Serial.print(',');
  Serial.print(stateName(g_state));
  Serial.print(',');
  Serial.println(dOn ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Telemetry line handling
// ---------------------------------------------------------------------------

static void processLine(char* line) {
  // Parse: timestamp,harvest_w,critical_w,essential_w,deferrable_w
  // (in-place split, identical to esp32_validation.ino)
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

  long   ts       = (long)atol(fields[0]);
  float  harvestW = (float)atof(fields[1]);
  // load_c / load_e / load_d are parsed for protocol parity but the manager
  // uses its configured tiers (the harness replays the same constants).
  (void)atof(fields[2]);
  (void)atof(fields[3]);
  (void)atof(fields[4]);

  stepManager(ts, harvestW);
}

// ---------------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(100);  // give the USB CDC time to enumerate on ESP32-S2/S3/C3

  fcInit(&g_fc);
  // Pending-ring offsets: lead k owns k+1 slots (lines 232 lead list).
  PEND_OFF[0] = 0;
  for (int l = 0; l < N_LEADS; l++) {
    PEND_OFF[l + 1] = PEND_OFF[l] + LEADS[l] + 1;
  }

  // Signal readiness so the host-side feeder script knows it can start
  // (same banner as esp32_validation.ino, requirement 10).
  Serial.print("READY,soc_wh=");
  Serial.print(g_soc, 1);
  Serial.print(",capacity_wh=");
  Serial.println(BAT_CAPACITY_WH, 1);
}

void loop() {
  // Read a full line (terminated by '\n' or '\r'), then process it.  The
  // reference harness sends consecutive minute timestamps; a gap would
  // misalign the pending-forecast rings, so a jump re-arms them (empty
  // rings simply resume learning once fresh forecasts come due).
  static long lastTs = LONG_MIN;
  static bool primed  = false;

  size_t len = 0;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) {
        g_line[len] = '\0';
        // Peek the timestamp to detect stream resyncs before stepping.
        char* comma = strchr(g_line, ',');
        long ts = LONG_MIN;
        if (comma) {
          *comma = '\0';
          ts = atol(g_line);
          *comma = ',';
        }
        if (primed && ts != lastTs + 1 && ts != lastTs) {
          for (int s = 0; s < PEND_SLOTS; s++) {
            g_fc.pendFc[s] = NAN;
            g_fc.pendDev[s] = 0.0f;
          }
        }
        lastTs = ts;
        primed = true;
        processLine(g_line);
        len = 0;
      }
      // blank lines (from \r\n) are ignored
    } else if (len < sizeof(g_line) - 1) {
      g_line[len++] = c;
    }
  }
}
