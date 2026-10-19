import matplotlib
matplotlib.use('Agg')

"""
dump_reference.py
=================

Generate ``reference_trace.csv`` -- the Python side of the hardware-in-the-
loop comparison driven by ``compare_esp32.py`` against ``esp32_full.ino``.

The script replays one randomised 24 h scenario from
``study3_energy_budgeting.make_scenario`` (1440 one-minute harvest samples)
through a *direct Python port of the ESP32 firmware's controller*: every
function below mirrors its C++ counterpart in esp32_full.ino line for line.

This is a full port -- not a simplification:

* Forecaster: ``struct Forecaster`` + ``fcInit`` / ``interpLead`` /
  ``fcPredict`` / ``fcForecastWh`` / ``fcUpdate``, same taus (1440/20/90/
  120/180), same alphas (1 - exp(-1/tau)), LEADS = {1,5,15,30,60,120,240,360},
  Z_BY_BAND = {0.5, 1.0, 1.5}, PEND_SLOTS = 839 ring buffers, fixed
  ephemeris ``shapeAt`` (half-sine over the 1080 min daylight arc with the
  360 min eclipse at minute 360 zeroed, wrapped mod 1440).
  NOTE: the firmware's sigma(k) is the |error| EWMA, ``interpLead(absEwma,
  lead)`` (struct comment "-> sigma at each lead"), NOT sqrt(var).
* Step order: identical to ``stepManager()`` --
      1. fcUpdate with this minute's harvest        (line 584)
      2. fHarvestWh = fcForecastWh(90, conservative) (line 587)
      3. fLoadWh = (C + E + D_if_prev_committed) * 1.5   (line 594)
      4. reserve from band (+ DEGRADED bump), margin from PREVIOUS SoC (599-606)
      5. target decision (SAFE exit / RECOVERY staged / thresholds)   (609-623)
      6. 10 % deep-discharge guard, then fsmDebounced (debounce + dwell) (626-634)
      7. load set from the NEW state; dOn gated by payloadGate()       (642-665)
      8. SoC integration (float32), stateT += 1                        (672-682)
      9. log timestamp, SoC AFTER update, state, gated payload         (686-692)
* FSM: full ``fsmGoto`` / ``fsmDebounced`` (severity escalation instant,
  relaxation needs 3 min candidate age AND 10 min dwell), the staged
  RECOVERY machine (C -> C+E -> C+E+D, 15 min holds, SoC 30 % / 55 %) and
  the SAFE_MODE exit at socf >= 0.30 && margin >= 0.
* Payload gate: ``payloadGate()`` including ``darkHoursAt()`` scanning the
  ephemeris, the conservative 30 min surplus test and the 45 min hold.
* Numeric parity: the firmware's float32 paths (shapeAt, harvest parse,
  SoC integrator, reserve/load floats, double-vs-float literal compares)
  are emulated with numpy float32 / widened float32 literals so threshold
  crossings land on the same side as the board.

``--debug`` prints one line at every step where the FSM state changes:
    DEBUG t=<t> soc=<soc> margin=<margin> forecast_wh=<..> load_wh=<..>
         reserve_wh=<..> state=<state>

Usage:
    python dump_reference.py [--trial 0] [--out reference_trace.csv]
                             [--horizon 90] [--debug]

Output columns (unchanged):
    timestamp,harvest_w,load_c,load_e,load_d,soc_py,fsm_py,payload_py
"""

import numpy as np
import csv
import argparse

# Only the scenario generator and the shared load/battery constants are
# imported.  SeasonalForecaster / PredictiveFSM are deliberately NOT used:
# the forecaster below is ported directly from the C++ in esp32_full.ino.
from study3_energy_budgeting import (
    make_scenario,
    W_CRITICAL,
    W_ESSENTIAL,
    W_DEFERRABLE,
    BAT_CAPACITY_WH,
    ETA_CHG,
    ETA_DIS,
)


# ---------------------------------------------------------------------------
# float32 helpers -- the firmware stores most scalars as C float; a widened
# float32 literal is the exact double the C expression compares against.
# ---------------------------------------------------------------------------
def _f32(x):
    """Value of x as stored in a C float, widened back to double."""
    return float(np.float32(x))


# ---------------------------------------------------------------------------
# Constants -- KEEP IN SYNC with esp32_full.ino (values taken from it).
# Loads / capacity / efficiencies are the study3 constants imported above and
# are identical to the firmware's copies (20 / 15 / 30 W, 500 Wh, 0.95).
# ---------------------------------------------------------------------------
N_STEPS = 1440                   # 24 h at 1-minute resolution
DT_MIN = 1.0                     # .ino: DT_MIN = 1.0f
DT_H = _f32(1.0 / 60.0)          # .ino: float DT_H = DT_MIN / 60.0f
SOC0_WH = 350.0                  # .ino: BAT_SOC0 = 350.0f
HORIZON_MIN = 90                 # .ino: HORIZON_MIN = 90

# margins / guard (float literals, widened where C compares double vs float)
MARGIN_SHED_DEFER_WH = _f32(0.0)       # 0.0f   -> DEGRADED
MARGIN_SHED_ESS_WH = _f32(-50.0)       # -50.0f -> CRITICAL
MARGIN_CRITICAL_WH = _f32(-100.0)      # -100.0f -> SAFE_MODE
DEEP_DISCH_SOC = _f32(0.10)            # 0.10f, compared against double socf
FSM_SAFE_SOC_CLEAR = _f32(0.30)        # 0.30f, SAFE_MODE -> RECOVERY exit

# reserve (esp32_full.ino lines 137-138, 599-605; computed in float)
RESERVE_BASE_FRAC = 0.20
RESERVE_LOW_FRAC = 0.30

# FSM debounce / dwell / recovery (esp32_full.ino lines 149-157)
FSM_DEBOUNCE_MIN = 3            # candidate age required to relax
FSM_MIN_DWELL_MIN = 10          # dwell required before relaxing
REC_SOC_ESS = 0.30              # SoC needed to start climbing out of RECOVERY
REC_SOC_PAYLOAD = 0.55          # SoC needed for the payload stage
REC_HOLD_MIN = 15.0             # minutes per recovery stage
FSM_SAFE_SOC_CLEAR_F = FSM_SAFE_SOC_CLEAR

# payload gate (esp32_full.ino lines 145-146)
PAYLOAD_GATE_LEAD_MIN = 30.0
PAYLOAD_GATE_HOLD_MIN = 45.0

# ephemeris (esp32_full.ino lines 180-185)
PANEL_PEAK_W = 155.0            # also study3 line 165
DAYLIGHT_MIN = 1080.0           # 18 h sunlit arc, half-sine
ECLIPSE_LEN_MIN = 360.0         # 6 h eclipse
ECLIPSE_START_MIN = 360.0       # midpoint of the randomised range
ORBIT_WRAP_MIN = 1440.0         # synthetic orbit == 1 day
LIT_GEOM_EPS = 0.0              # darkHoursAt lit test
LIT_FC_EPS = _f32(0.05)         # forecaster lit test (0.05f)

# forecaster (esp32_full.ino lines 160-167, 188-193)
RESID_TAU_MIN = 20.0            # fast residual EWMA tau
RESID_SLOW_TAU_MIN = 90.0       # slow clearness EWMA tau
RESID_SYS_TAU_MIN = 120.0       # systematic bias EWMA tau
RESID_REG_TAU_MIN = 180.0       # gain regression EWMA tau
ORBIT_PERIOD_MIN = 1440.0       # seasonal level tau
CONF_HIGH_MAE = _f32(0.12)      # ebar < 0.12f -> BAND_HIGH
CONF_MED_MAE = _f32(0.28)       # ebar < 0.28f -> BAND_MEDIUM
USE_RESIDUAL = True             # PREDICT_5 rung

N_LEADS = 8
LEADS = (1, 5, 15, 30, 60, 120, 240, 360)
Z_BY_BAND = (0.50, 1.00, 1.50)  # exact in float32 as well

BAND_HIGH, BAND_MEDIUM, BAND_LOW = 0, 1, 2

# pending ring: lead k owns k+1 slots (PEND_OFF filled like setup())
PEND_OFF = []
_off = 0
for _k in LEADS:
    PEND_OFF.append(_off)
    _off += _k + 1
PEND_SLOTS = _off               # 839

# FSM state encoding (esp32_full.ino lines 199-209)
ST_NOMINAL = "NOMINAL"
ST_DEGRADED = "DEGRADED"
ST_CRITICAL = "CRITICAL"
ST_SAFE_MODE = "SAFE_MODE"
ST_RECOVERY = "RECOVERY"

# escalation order: protection is instantaneous, recovery is debounced
FSM_SEVERITY = {
    ST_NOMINAL: 0,
    ST_DEGRADED: 2,
    ST_CRITICAL: 3,
    ST_SAFE_MODE: 4,
    ST_RECOVERY: 1,
}

HEADER = [
    "timestamp", "harvest_w", "load_c", "load_e", "load_d",
    "soc_py", "fsm_py", "payload_py",
]


# ---------------------------------------------------------------------------
# Ephemeris -- shapeAt() / geomLitAt() / darkHoursAt() (esp32_full.ino 231-266)
# ---------------------------------------------------------------------------
def shape_at(minute):
    """Normalised clear-sky geometry in [0, 1]: half-sine over the daylight
    arc (x = tt / 1080, s = sin(pi * x)), eclipse window zeroed, clamped to
    [0, 1], wrapped mod 1440.  Computed like the firmware, in float."""
    tt = int(minute) % int(ORBIT_WRAP_MIN)
    if tt < 0:
        tt += int(ORBIT_WRAP_MIN)
    x = np.float32(tt) / np.float32(DAYLIGHT_MIN)       # float division
    if 0.0 <= x <= 1.0:
        # C: sinf(PI * x) -- PI is a double macro, arg rounded to float
        s = np.float32(np.sin(_f32(np.pi * float(x))))
    else:
        s = np.float32(0.0)
    if tt >= ECLIPSE_START_MIN and tt < ECLIPSE_START_MIN + ECLIPSE_LEN_MIN:
        s = np.float32(0.0)
    if s < 0.0:
        s = np.float32(0.0)
    if s > 1.0:
        s = np.float32(1.0)
    return s


def geom_lit_at(minute):
    """geomLitAt(): shape > 0.0 (LIT_GEOM_EPS)."""
    return float(shape_at(minute)) > LIT_GEOM_EPS


def dark_hours_at(i):
    """Geometry.dark_hours port (esp32_full.ino lines 252-266): hours of
    continuous dark immediately ahead of minute i.  Returns float32 like the
    firmware's ``(float)dark * DT_H``."""
    hmax = 480
    if geom_lit_at(i):
        s = 0
        while s < hmax and geom_lit_at(i + s):
            s += 1
        if s >= hmax:
            return np.float32(0.0)
        k = i + s
    else:
        k = i
    dark = 0
    while not geom_lit_at(k + dark):
        dark += 1
    return np.float32(dark) * np.float32(DT_H)


# ---------------------------------------------------------------------------
# Forecaster -- direct port of struct Forecaster + fcInit / interpLead /
# alphaFastFor / fcPredict / fcForecastWh / fcUpdate (esp32_full.ino 311-440)
# ---------------------------------------------------------------------------
class Esp32Forecaster:
    """Line-for-line Python port of the firmware forecaster (double state,
    float32 pending rings, exactly the C++ formulas and branch order)."""

    def __init__(self):
        # --- fcInit() ------------------------------------------------------
        self.level = 1.0
        self.fast = 1.0
        self.slow = 1.0
        self.dev = 0.0
        self.gain = [0.0] * N_LEADS
        self.bias = [0.0] * N_LEADS
        self.cov = [0.0] * N_LEADS
        self.var = [0.0] * N_LEADS
        self.err_ewma = [0.0] * N_LEADS
        self.abs_ewma = [0.0] * N_LEADS
        self.seen = False
        self.band = BAND_LOW
        # alpha = 1 - exp(-dt / tau)  (double math in the C code)
        self.a_level = float(1.0 - np.exp(-1.0 / ORBIT_PERIOD_MIN))
        self.a_dev = float(1.0 - np.exp(-1.0 / RESID_TAU_MIN))
        self.a_slow = float(1.0 - np.exp(-1.0 / RESID_SLOW_TAU_MIN))
        self.a_sys = float(1.0 - np.exp(-1.0 / RESID_SYS_TAU_MIN))
        self.a_reg = float(1.0 - np.exp(-1.0 / RESID_REG_TAU_MIN))
        # pending rings, float32 like `float pendFc[PEND_SLOTS]`
        self.pend_fc = np.full(PEND_SLOTS, np.nan, dtype=np.float32)
        self.pend_dev = np.zeros(PEND_SLOTS, dtype=np.float32)

    @staticmethod
    def _interp_lead(values, lead):
        """interpLead(): clamp + linear interpolation across LEADS."""
        if lead <= LEADS[0]:
            return values[0]
        if lead >= LEADS[N_LEADS - 1]:
            return values[N_LEADS - 1]
        for l in range(1, N_LEADS):
            if lead <= LEADS[l]:
                f = (lead - LEADS[l - 1]) / float(LEADS[l] - LEADS[l - 1])
                return values[l - 1] + f * (values[l] - values[l - 1])
        return values[N_LEADS - 1]

    @staticmethod
    def _alpha_fast(lead):
        """alphaFastFor(): error statistics adapt faster at short leads."""
        return float(1.0 - np.exp(
            -1.0 / (RESID_TAU_MIN * (1.0 + 0.5 * lead / 60.0))))

    def predict(self, t, lead_min, conservative):
        """fcPredict(): point forecast of array power at t + lead (W).
        sigma(k) = interpLead(absEwma, lead)  -- as in the C++ (NOT sqrt(var))."""
        j = t + int(lead_min / DT_MIN + 0.5)            # (long)(lead/Dt + 0.5)
        shape_f = float(shape_at(j))                    # float -> double
        if shape_f <= 0.0:
            return 0.0
        lvl = self.level
        if conservative:
            lvl -= Z_BY_BAND[self.band] * self._interp_lead(self.abs_ewma,
                                                            lead_min)
        corr = 0.0
        bias_c = 0.0
        if USE_RESIDUAL:
            corr = self.dev * self._interp_lead(self.gain, lead_min)
            bias_c = self._interp_lead(self.bias, lead_min)
        val = PANEL_PEAK_W * (lvl + corr + bias_c) * shape_f
        return val if val > 0.0 else 0.0

    def forecast_wh(self, t, kmax, conservative):
        """fcForecastWh(): sum of predict() over t+1 .. t+kmax (Wh)."""
        wh = 0.0
        for k in range(1, kmax + 1):
            wh += self.predict(t, float(k), conservative) * DT_H
        return wh

    def update(self, t, shape_now, harvest_w):
        """fcUpdate(): called every minute; learning itself is sunlit-only
        (lit = shapeNow > 0.05f).  harvest_w / shape_now arrive as float32
        values widened to double, exactly like the C++ parameters."""
        lit = float(shape_now) > LIT_FC_EPS
        clr = float(harvest_w) / PANEL_PEAK_W

        # seasonal level + band-pass pair
        if lit:
            self.level += self.a_level * (clr - self.level)
            if self.level < 0.0:
                self.level = 0.0
            if self.level > 1.5:
                self.level = 1.5
            self.fast += self.a_dev * (clr - self.fast)
            self.slow += self.a_slow * (clr - self.slow)
        dev_now = (self.fast - self.slow) if lit else 0.0

        # realise pending forecasts that fall due now
        for l in range(N_LEADS):
            k = LEADS[l]
            if t < k:
                continue                               # not yet issued
            slot = PEND_OFF[l] + (t - k) % (k + 1)
            fc = self.pend_fc[slot]
            if np.isnan(fc):                           # empty slot
                continue
            dev_issue = float(self.pend_dev[slot])     # float -> double
            self.pend_fc[slot] = np.float32(np.nan)    # Python dict.pop
            if not lit:
                continue                               # dark: no learning
            # forecast error against this minute's actual (measured) harvest
            err = clr - _f32(fc) / PANEL_PEAK_W
            self.dev = dev_now

            # (1) least-squares persistence gain
            if dev_issue != 0.0:
                self.cov[l] += self.a_reg * (dev_issue * dev_now - self.cov[l])
                self.var[l] += self.a_reg * (dev_issue * dev_issue -
                                             self.var[l])
                if self.var[l] > 1e-6:
                    g = self.cov[l] / self.var[l]
                    self.gain[l] = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
            # (2) error statistics -> confidence band input
            ae = self._alpha_fast(k)
            self.err_ewma[l] += ae * (err - self.err_ewma[l])
            self.abs_ewma[l] += ae * (abs(err) - self.abs_ewma[l])
            # (3) systematic error-feedback bias
            if USE_RESIDUAL:
                self.bias[l] += self.a_sys * err
            self.seen = True

        # confidence band from the worst lead's |error| EWMA
        if self.seen:
            ebar = 0.0
            for l in range(N_LEADS):
                if self.abs_ewma[l] > ebar:
                    ebar = self.abs_ewma[l]
            if ebar < CONF_HIGH_MAE:
                self.band = BAND_HIGH
            elif ebar < CONF_MED_MAE:
                self.band = BAND_MEDIUM
            else:
                self.band = BAND_LOW
        else:
            self.band = BAND_LOW

        # register forecasts now in flight (float32 storage, like the C++)
        for l in range(N_LEADS):
            k = LEADS[l]
            slot = PEND_OFF[l] + t % (k + 1)
            self.pend_fc[slot] = np.float32(self.predict(t, float(k), False))
            self.pend_dev[slot] = np.float32(dev_now)


# ---------------------------------------------------------------------------
# Controller -- Manager state, fsmGoto / fsmDebounced / recoveryTarget /
# payloadGate / stepManager (esp32_full.ino lines 446-693)
# ---------------------------------------------------------------------------
class Esp32Controller:
    """Port of the firmware's globals + stepManager()."""

    def __init__(self):
        self.soc = np.float32(SOC0_WH)     # g_soc, float32
        self.state = ST_NOMINAL            # g_state
        self.state_t = 0                   # g_stateT
        self.cand = ST_NOMINAL             # g_cand
        self.cand_t = 0                    # g_candT
        self.rec_stage = 0                 # g_recStage
        self.rec_ess_hold = 0.0            # g_recEssHold (float)
        self.rec_def_hold = 0.0            # g_recDefHold (float)
        self.rec_bad = 0                   # g_recBad
        self.prev_d_on = False             # g_prevDOn
        self.hw_trip = False               # g_hwTrip (never set by the harness)

    # -- fsmGoto (lines 469-481) -------------------------------------------
    def _goto(self, new_s):
        if new_s != self.state:
            self.state = new_s
            self.state_t = 0
            self.cand = new_s
            self.cand_t = 0
        if new_s != ST_RECOVERY:
            self.rec_stage = 0
            self.rec_ess_hold = 0.0
            self.rec_def_hold = 0.0

    # -- fsmDebounced (lines 486-505) --------------------------------------
    def _debounced(self, target):
        if target == self.state:
            self.cand = target
            self.cand_t = 0
            return
        if FSM_SEVERITY[target] > FSM_SEVERITY[self.state]:
            self._goto(target)                       # escalation is instant
            return
        if target == self.cand:
            self.cand_t += 1
        else:
            self.cand = target
            self.cand_t = 1
        if self.cand_t >= FSM_DEBOUNCE_MIN and self.state_t >= FSM_MIN_DWELL_MIN:
            self._goto(target)

    # -- recoveryTarget (lines 518-546) ------------------------------------
    def _recovery_target(self, socf, margin):
        """socf / margin arrive already rounded to float32 by the caller."""
        if margin < 0.0:
            self.rec_bad += 1
        else:
            self.rec_bad = 0
        sustained_bad = self.rec_bad >= FSM_DEBOUNCE_MIN

        if socf < np.float32(REC_SOC_ESS) or sustained_bad:
            self.rec_stage = 0
            return ST_RECOVERY
        if self.rec_stage == 0:
            self.rec_ess_hold += DT_MIN
            if self.rec_ess_hold >= REC_HOLD_MIN:
                self.rec_stage = 1
        if self.rec_stage >= 1:
            if socf < np.float32(REC_SOC_PAYLOAD) or margin < 0.0:
                return ST_RECOVERY
            self.rec_def_hold += DT_MIN
            if self.rec_def_hold >= REC_HOLD_MIN:
                self.rec_stage = 2
        return ST_NOMINAL if self.rec_stage >= 2 else ST_RECOVERY

    # -- payloadGate (lines 558-568) ---------------------------------------
    def _payload_gate(self, t, soc, reserve, fc):
        """soc / reserve are float32 values, as passed in the C call."""
        base_w = np.float32(W_CRITICAL + W_ESSENTIAL)
        dark_req = base_w * dark_hours_at(t) + np.float32(reserve)
        if soc < dark_req:
            return False

        k = int(PAYLOAD_GATE_LEAD_MIN / DT_MIN)                  # = 30
        gate_k = np.float32(k) * np.float32(DT_H)                # k * DT_H (f)
        surplus = (fc.forecast_wh(t, k, True)
                   - (W_CRITICAL + W_ESSENTIAL + W_DEFERRABLE) * float(gate_k))
        if surplus >= 0.0:
            return True
        hold = np.float32(W_DEFERRABLE * PAYLOAD_GATE_HOLD_MIN / 60.0)
        return soc >= dark_req + hold

    # -- stepManager (lines 578-693) ---------------------------------------
    def step(self, t, harvest_tx, harvest_logged, fc, horizon, debug):
        """One control step.  harvest_tx is float32 (what the firmware
        parses from the serial line); harvest_logged is the full-precision
        value written to reference_trace.csv.  Returns the CSV row and True
        if the FSM state changed this minute."""
        soc_prev = np.float32(self.soc)                 # socPrev = g_soc
        socf = float(soc_prev) / float(BAT_CAPACITY_WH)  # (double) / (double)
        shape_now = shape_at(t)

        # 1. forecast update: learn from this minute's measurement
        fc.update(t, shape_now, harvest_tx)

        # 2. planning horizon + conservative harvest forecast
        f_harvest_wh = fc.forecast_wh(t, horizon, True)

        # 3. load forecast: base load + payload only if COMMITTED last step
        load_w = W_CRITICAL + W_ESSENTIAL + (W_DEFERRABLE
                                             if self.prev_d_on else 0.0)
        f_load_wh = float(np.float32(load_w) *
                          (np.float32(horizon) / np.float32(60.0)))

        # 4. reserve (float math) + margin from the PREVIOUS SoC
        reserve = float(np.float32(BAT_CAPACITY_WH) *
                        np.float32(RESERVE_LOW_FRAC
                                   if fc.band == BAND_LOW
                                   else RESERVE_BASE_FRAC))
        if self.state == ST_DEGRADED:
            deg_res = float(np.float32(BAT_CAPACITY_WH) *
                            np.float32(RESERVE_LOW_FRAC))
            if deg_res > reserve:
                reserve = deg_res
        margin = (float(soc_prev) + f_harvest_wh
                  - f_load_wh - float(reserve))

        # 5. margin-driven target state (branch order as in the C++)
        prev_state = self.state
        if self.state == ST_SAFE_MODE:
            target = (ST_RECOVERY
                      if (socf >= FSM_SAFE_SOC_CLEAR and margin >= 0.0)
                      else ST_SAFE_MODE)
        elif self.state == ST_RECOVERY:
            target = self._recovery_target(np.float32(socf),
                                           np.float32(margin))
        elif margin < MARGIN_CRITICAL_WH:
            target = ST_SAFE_MODE
        elif margin < MARGIN_SHED_ESS_WH:
            target = ST_CRITICAL
        elif margin < MARGIN_SHED_DEFER_WH:
            target = ST_DEGRADED
        else:
            target = ST_NOMINAL

        # 6. hard SoC floor -> SAFE_MODE (guard skipped while already SAFE)
        if socf < DEEP_DISCH_SOC and self.state != ST_SAFE_MODE:
            target = ST_SAFE_MODE
        if self.hw_trip:
            target = ST_SAFE_MODE
        self._debounced(target)

        # 7. load set implied by the NEW state
        e_on, d_on = True, False
        if self.state == ST_NOMINAL:
            e_on, d_on = True, True
        elif self.state == ST_DEGRADED:
            e_on, d_on = True, False
        elif self.state in (ST_CRITICAL, ST_SAFE_MODE):
            e_on, d_on = False, False
        elif self.state == ST_RECOVERY:
            e_on = self.rec_stage >= 1
            d_on = self.rec_stage >= 2

        # deferrable scheduler: energy-rich windows only
        if d_on:
            d_on = self._payload_gate(t, soc_prev, np.float32(reserve), fc)
        self.prev_d_on = d_on

        # 8. SoC integration (all float32, exactly like the C++)
        served_w = np.float32(W_CRITICAL
                              + (W_ESSENTIAL if e_on else 0.0)
                              + (W_DEFERRABLE if d_on else 0.0))
        if harvest_tx > served_w:
            self.soc = np.float32(self.soc + (harvest_tx - served_w)
                                  * np.float32(ETA_CHG) / np.float32(60.0))
        else:
            self.soc = np.float32(self.soc + (harvest_tx - served_w)
                                  / np.float32(ETA_DIS) / np.float32(60.0))
        if self.soc < 0.0:
            self.soc = np.float32(0.0)
        if self.soc > np.float32(BAT_CAPACITY_WH):
            self.soc = np.float32(BAT_CAPACITY_WH)

        self.state_t += 1

        # 9. log: timestamp, SoC AFTER update, state, gated payload decision
        state_changed = self.state != prev_state
        if debug and state_changed:
            print(f"DEBUG t={t} soc={float(self.soc):.3f} margin={margin:.3f} "
                  f"forecast_wh={f_harvest_wh:.3f} load_wh={f_load_wh:.3f} "
                  f"reserve_wh={reserve:.3f} state={self.state}")

        row = (t, harvest_logged, W_CRITICAL, W_ESSENTIAL, W_DEFERRABLE,
               float(self.soc), self.state, int(d_on))
        return row, state_changed


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------
def build_trace(trial, horizon, debug):
    """Replay one study3 scenario through the ported firmware controller."""
    scen = make_scenario(trial=trial)
    harvest = scen.harvest_w                    # 1440 values

    fc = Esp32Forecaster()                      # fcInit() in __init__
    ctrl = Esp32Controller()                    # firmware globals
    rows = []
    for i in range(N_STEPS):
        harvest_full = float(harvest[i])
        # compare_esp32.py sends "%.6f"; the firmware parses that into a
        # float32 -- emulate the same value so both sides compute identically
        harvest_tx = np.float32(float(f"{harvest_full:.6f}"))
        row, _ = ctrl.step(i, harvest_tx, harvest_full, fc, horizon, debug)
        rows.append(row)
    return scen, rows


def main():
    ap = argparse.ArgumentParser(
        description="Write reference_trace.csv (exact esp32_full.ino mirror).")
    ap.add_argument("--trial", type=int, default=0,
                    help="study3 Monte-Carlo trial index (default 0)")
    ap.add_argument("--out", default="reference_trace.csv",
                    help="output CSV path (default reference_trace.csv)")
    ap.add_argument("--horizon", type=int, default=HORIZON_MIN,
                    help=f"planning horizon in minutes (default {HORIZON_MIN})")
    ap.add_argument("--debug", action="store_true",
                    help="print t/soc/margin/forecast/load/reserve/state at "
                         "every FSM state change")
    args = ap.parse_args()

    scen, rows = build_trace(args.trial, args.horizon, args.debug)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    # ---- sanity summary ---------------------------------------------------
    socs = [r[5] for r in rows]
    counts = {}
    first_off_nominal = None
    for r in rows:
        counts[r[6]] = counts.get(r[6], 0) + 1
        if first_off_nominal is None and r[6] != ST_NOMINAL:
            first_off_nominal = r[0]

    print(f"wrote {args.out}  ({len(rows)} rows, trial {args.trial}, "
          f"seed {scen.seed}, horizon {args.horizon} min)")
    print(f"  SoC range : {min(socs):.2f} .. {max(socs):.2f} Wh  "
          f"(start {SOC0_WH:.1f})")
    print("  states    : " + ", ".join(f"{k}={v}"
                                       for k, v in sorted(counts.items())))
    if first_off_nominal is None:
        print("  first non-NOMINAL minute : never left NOMINAL")
    else:
        print(f"  first non-NOMINAL minute : {first_off_nominal}")


if __name__ == "__main__":
    main()
