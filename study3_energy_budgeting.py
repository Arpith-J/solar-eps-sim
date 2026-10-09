"""
study3_energy_budgeting.py
==========================

Simulation study #3 for a solar-powered CubeSat-inspired Electrical Power System
(EPS) energy manager.

The script builds a 24 h, 1-minute-resolution simulation of a single-string solar
EPS (solar array, Li-ion battery, three priority load tiers) and compares six
energy-management strategies on the *same* randomised orbital scenarios:

    Rung 0  NO_MGMT    - no management, all loads always commanded on
    Rung 1  REACTIVE   - fixed SoC thresholds (shed D < 30 %, shed E < 20 %)
    Rung 2  REACTIVE_H - thresholds + hysteresis + priority tiers + trend + debounce
    Rung 3  PREDICT_4  - predictive FSM driven by a seasonal (orbit-matched) EWMA
    Rung 4  PREDICT_5  - predictive FSM + learned residual correction
    Rung 5  ORACLE     - perfect knowledge of future harvest (upper bound)

Everything is pure ``numpy`` + ``matplotlib`` (plus the standard library ``csv`` /
``os`` for report writing).  No pandas, no scipy.

Key modelling assumptions (deliberately, and stated up front so the results can be
interpreted):

  * The orbit is treated as one 24 h revolution: the satellite sees 18 h of
    sunlight with a 6 h orbital eclipse embedded in it, so only ~12 h of the day
    actually produce array power.  The eclipse is *deterministic* and is
    therefore part of the ephemeris that every strategy can know in advance --
    this is the physical justification for predictive management (ops teams
    absolutely know eclipse entry/exit from the orbit propagator).
  * Cloud cover is *not* knowable in advance, so it is the part of the forecast
    that must be learned.  This is exactly what the seasonal EWMA + residual
    correction is for.
  * The system is energy *limited* over the orbit: the array cannot cover the
    24 h demand of Critical + Essential.  Every watt of Deferrable (payload)
    energy therefore has to be rationed, and that is what makes the
    look-ahead/predictive strategies interesting.
  * The bus is a single DC converter: when array power exceeds the load the
    surplus charges the battery (x eta_chg) and is curtailed when full; when the
    load exceeds the array the battery supplies the deficit (x 1/eta_dis).
    Unserved demand is taken from the battery in *reverse priority order*
    (Deferrable first, then Essential, Critical last) -- this is what makes the
    no-management rung the worst possible policy.

Extended (180-day) study -- ``USE_EXTENDED = True``
---------------------------------------------------
On top of the 24 h study the script can propagate a *realistic* 408 km /
92.6 min BIRDS-4 low-Earth orbit for ``SIM_DAYS = 180`` days at 1-minute
resolution and fold in three slow degradation effects:

  * orbit geometry - seasonal beta angle and the eclipse fraction it drives,
                     with one contiguous eclipse per orbit (NASA TFAWS 2014);
  * array          - 2.8 %/yr panel degradation (NASA ACTS 2005 + MDPI 2025,
                     MinXSS) plus a MinXSS/XACT pointing (attitude) loss;
  * battery        - 0.02 % capacity loss per orbit cycle (NASA PCoE),
                     floored at 30 % total loss.

The flag ``USE_EXTENDED`` at the top of the configuration section selects the
driver: ``False`` runs only the original 24 h study, ``True`` runs the
180-day extension (the 24 h code path and its plots are left intact).

Outputs
-------
    results/metrics.csv          summary table (mean/std per metric per strategy)
    results/metrics_trials.csv   raw per-trial metrics
    results/plot1_soc_*.png      SoC vs time, all strategies
    results/plot2_loads_*.png    load on/off, predictive strategy
    results/plot3_fsm_*.png      5-state FSM timeline, predictive strategy
    results/plot4_uptime_*.png   critical-load uptime per strategy (MC mean +/- std)
    results/plot5_payload_*.png  payload energy per strategy
    results/plot6_tradeoff_*.png uptime vs payload trade-off
    results/plot7_predictor_*.png predictor error over time
    results/ext_metrics.csv      180-day summary table (same columns as above)
    results/ext_metrics_trials.csv  180-day raw metrics
    results/plot_ext_soc.png     SoC over 180 days, all strategies
    results/plot_ext_harvest.png harvest over 180 days, before/after degradation
    results/plot_ext_eclipse.png eclipse fraction vs day (beta angle overlay)
    results/plot_ext_metrics.png metric bar charts over 180 days
"""

from __future__ import annotations

import csv
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: always save PNG, never open a window
import matplotlib.pyplot as plt


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

# ---- study driver flag -----------------------------------------------------
# USE_EXTENDED selects which driver main() runs:
#   False -> only the original 24 h Monte-Carlo study (plots 1-7, metrics.csv)
#   True  -> the 180-day extension (plot_ext_*.png, ext_metrics*.csv); the
#            original 24 h simulation is left untouched and still runs when
#            the flag is flipped back to False.
USE_EXTENDED = True

# ---- time base -------------------------------------------------------------
DT_MIN = 1.0                       # [min] simulation time step
DT_H = DT_MIN / 60.0               # [h]   simulation time step
N_STEPS = 1440                     # 24 h at 1-minute resolution
HOURS_AXIS = np.arange(N_STEPS) * DT_MIN / 60.0

# ---- orbital geometry (original 24 h study: synthetic one-day orbit) -------
# The legacy study models one 24 h revolution, so these constants keep their
# original values under the *_24H names; the extension uses the realistic
# constants below instead (its eclipse is recomputed every day).
ORBIT_PERIOD_MIN_24H = 1440.0      # [min] synthetic orbit == one simulated day
DAYLIGHT_MIN = 18.0 * 60.0         # [min] sunlit arc per orbit (18 h)
ECLIPSE_MIN_24H = 360.0            # [min] umbral (eclipse) segment, 6 h
# Eclipse start offset window: the eclipse has to fit inside the daylight arc, so
# the start can be anywhere in [0, DAYLIGHT - ECLIPSE] = [0, 12 h].
ECLIPSE_OFFSET_RANGE = (0.0 * 60.0, DAYLIGHT_MIN - ECLIPSE_MIN_24H)

# ---- orbital geometry (180-day extension: realistic LEO orbit) -------------
# NASA TFAWS 2014 (orbit geometry); BIRDS-4 (BRICSat-C) 408 km sun-synchronous
# orbit, 92.6 min period.  The fixed 6 h ECLIPSE_MIN of the synthetic study is
# gone here: every day recomputes the eclipse from the beta angle instead.
ALTITUDE_KM = 408.0                # [km] circular orbit altitude
EARTH_RADIUS_KM = 6371.0           # [km] mean Earth radius (umbral cone)
ORBIT_PERIOD_MIN = 92.6            # [min] BIRDS-4 orbital period
BETA_ANGLE_MIN = 0.0               # [deg] beta-angle at the start of the window
BETA_ANGLE_MAX = 23.45             # [deg] Earth obliquity -> seasonal beta swing
BETA_PERIOD_DAYS = 365.0           # [d] beta angle repeats once per year
ECLIPSE_PHASE_MIN = 0.0            # [min] every orbit starts at eclipse entry, so
                                   #       the eclipse is one contiguous block
                                   #       beginning at this fixed orbit phase

# ---- array degradation (NASA ACTS 2005 + MDPI 2025; MinXSS) ---------------
DEGRADATION_PER_YEAR = 0.028       # [-] 2.8 %/year end-of-life loss (MinXSS)
DEGRADATION_START_DAY = 0          # [d] degradation clock starts at mission day 0

# ---- attitude / pointing effects (MinXSS XACT) ----------------------------
POINTING_ERROR_ARCSEC = 25.0       # [arcsec] worst-case XACT pointing error
# cos-loss of a normally-illuminated array for that error (radians(25/3600)
# converts arcsec -> deg -> rad); the resulting loss is tiny (physics!), so it
# is bounded to at most 1 % by attitude_loss() below.
POINTING_LOSS_FRACTION = 1.0 - np.cos(np.radians(POINTING_ERROR_ARCSEC / 3600.0))

# ---- battery degradation (NASA PCoE) --------------------------------------
BATTERY_CYCLE_DEGRADATION = 0.0002  # [-] 0.02 % capacity loss per charge cycle
CYCLES_PER_DAY = 1440.0 / ORBIT_PERIOD_MIN  # [-] orbits (cycles) per 24 h day

# ---- extended (180-day) run length ---------------------------------------
SIM_DAYS = 180                     # [d]  mission length of the extension
STEPS_PER_DAY = 1440               # [min/d] 1-minute resolution (1440 min/day)
N_STEPS_TOTAL = SIM_DAYS * STEPS_PER_DAY  # [min] total extended time steps
# Number of Monte-Carlo trials used by the 180-day extension.  A 180-day run is
# ~180x longer than the 24 h study, so the extended sweep uses fewer trials than
# N_TRIALS to keep the script runnable; the aggregation is over these trials.
EXT_N_TRIALS = 3                   # [-] extended-run Monte-Carlo trials

# ---- array (solar panel) ---------------------------------------------------
PANEL_PEAK_W = 155.0               # [W] clear-sky array power at normal incidence

# ---- battery ---------------------------------------------------------------
BAT_CAPACITY_WH = 500.0            # [Wh] nameplate capacity
BAT_SOC0_NOMINAL = 0.80            # [-] nominal initial state of charge
ETA_CHG = 0.95                     # [-] charge efficiency
ETA_DIS = 0.95                     # [-] discharge efficiency
BROWNOUT_SOC = 0.05                # [-] SoC below this == brownout event
DEEP_DISCH_SOC = 0.10              # [-] SoC below this == deep-discharge event

# ---- load tiers ------------------------------------------------------------
W_CRITICAL = 20.0                  # [W] never shed
W_ESSENTIAL = 15.0                 # [W] shed only in emergencies
W_DEFERRABLE = 30.0                # [W] payload, shed first / scheduled
LOADS = ("critical", "essential", "deferrable")
LOAD_W = {"critical": W_CRITICAL, "essential": W_ESSENTIAL, "deferrable": W_DEFERRABLE}
PRIORITY = {"deferrable": 0, "essential": 1, "critical": 2}  # higher = more important
PRIORITY_ORDER = tuple(sorted(LOADS, key=lambda k: PRIORITY[k]))  # served worst-first

# ---- predictive reserve / margin thresholds (requirement 5) ---------------
RESERVE_FRAC = 0.20                # [-] reserve = 20 % of capacity
RESERVE_FRAC_LOW_CONF = 0.30       # [-] reserve = 30 % of capacity when conf == LOW
RESERVE_FRAC_DEGRADED = 0.30       # [-] DEGRADED state raises the reserve
MARGIN_SHED_DEFER_WH = 0.0         # [Wh] margin < 0    -> shed Deferrable
MARGIN_SHED_ESS_WH = -50.0         # [Wh] margin < -50  -> shed Essential
MARGIN_CRITICAL_WH = -100.0        # [Wh] margin < -100 -> CRITICAL / SAFE mode

# ---- look-ahead / scheduling ---------------------------------------------
PLAN_HORIZON_MIN_MAX = 480.0       # [min] max look-ahead (8 h: eclipse + recharge)
PLAN_HORIZON_MIN_MIN = 45.0        # [min] min look-ahead
PAYLOAD_GATE_LEAD_MIN = 30.0       # [min] lead used for the payload surplus gate
PAYLOAD_GATE_HOLD_MIN = 45.0       # [min] battery margin above reserve that allows
                                    #       a payload run on stored energy

# ---- FSM debounce / hysteresis -------------------------------------------
FSM_DEBOUNCE_MIN = 3               # [min] condition must persist before a transition
FSM_MIN_DWELL_MIN = 10             # [min] minimum dwell time in a state
FSM_HYST_WH = 15.0                 # [Wh] margin hysteresis on state exits
FSM_RECOVERY_SOC_ESS = 0.30        # [-] SoC needed to re-enable Essential
FSM_RECOVERY_SOC_DEF = 0.35        # [-] SoC needed to re-enable Deferrable
FSM_SAFE_SOC_CLEAR = 0.30          # [-] SoC needed to leave SAFE mode
FSM_RECOVERY_HOLD_ESS_MIN = 10.0   # [min] hold before Essential is restored
FSM_RECOVERY_HOLD_DEF_MIN = 15.0   # [min] hold before Deferrable is restored

# ---- reactive rungs (SoC-threshold controllers) --------------------------
REACT2 = {"d_shed": 0.30, "d_rest": 0.35, "e_shed": 0.20, "e_rest": 0.25}
REACT3 = {
    "d_shed": 0.30,                # hard shed of the payload
    "d_rest": 0.35,                # hard restore of the payload
    "e_shed": 0.20,                # hard shed of Essential
    "e_rest": 0.25,                # hard restore of Essential
    "trend_window": 10,            # [min] trailing window for dSoC/dt
    "trend_w": 0.30,               # [Wh/min] "falling fast enough" trigger
    "min_on": 20,                  # [min] debounce: minimum payload run block
    "min_off": 10,                 # [min] debounce: minimum payload off block
}

# ---- Monte Carlo ----------------------------------------------------------
N_TRIALS = 50
BASE_SEED = 20240517              # master seed; trial i uses BASE_SEED + 1009*i
SOA0_RANGE = (0.70, 0.92)          # [-] randomised initial SoC
FADE_RANGE = (0.0, 0.10)          # [-] randomised capacity fade (up to 10 %)
CLOUD_GAP_RANGE = (45.0, 210.0)   # [min] mean clear-sky gap length
CLOUD_DUR_RANGE = (8.0, 45.0)     # [min] mean cloud-gap duration
CLOUD_ATTEN_RANGE = (0.05, 0.40)  # [-] clearness level inside a cloud

# ---- forecaster -----------------------------------------------------------
TRACKED_LEADS = (1, 5, 15, 30, 60, 120, 240, 360)   # [min] leads scored for MAE
RESID_TAU_MIN = 20.0               # [min] fast clearness EWMA (band-pass)
RESID_SLOW_TAU_MIN = 90.0          # [min] medium clearness EWMA (band-pass)
RESID_SYS_TAU_MIN = 120.0          # [min] systematic (error-feedback) EWMA
RESID_REG_TAU_MIN = 180.0          # [min] EWMA of the gain regression statistics
CONF_HIGH = 0.12                   # MAE / PEAK, below this -> HIGH confidence
CONF_MED = 0.28                    # ... below this -> MEDIUM, else LOW
Z_CONFIDENCE = {"HIGH": 0.50, "MEDIUM": 1.00, "LOW": 1.50}   # conservative bound

# ---- output ---------------------------------------------------------------
RESULTS_DIR = "results"
PLOT_DPI = 130


# =============================================================================
# 2. SMALL UTILITIES
# =============================================================================

def wrap_idx(i: int, n: int) -> int:
    """Wrap an absolute step index into [0, n) (the ephemeris is orbit-periodic)."""
    return i % n


def clear_sky_shape(n: int, sunrise: float, daylight: float, eclipse: Tuple[float, float]) -> np.ndarray:
    """
    Normalised clear-sky solar geometry in [0, 1] (1 == full illumination).

    The profile is a half-sine over the daylight arc, clipped to zero outside it,
    and additionally forced to zero over the eclipse window.  The simulation
    window starts at orbit sunrise, so the daylight arc is [0, DAYLIGHT_MIN], the
    eclipse sits at a random offset inside it, and the window closes with the
    post-eclipse night.  This array is the *ephemeris*: it is deterministic,
    orbit-periodic and known in advance to every strategy, which is what lets a
    planner reason about eclipse entry/exit.
    """
    t = np.arange(n) * DT_MIN
    x = (t - sunrise) / max(daylight, 1e-9)
    shape = np.where((x >= 0.0) & (x <= 1.0), np.sin(np.pi * np.clip(x, 0.0, 1.0)), 0.0)
    e0, e1 = eclipse
    shape = np.where((t >= e0) & (t < e1), 0.0, shape)
    return np.clip(shape, 0.0, 1.0)


def cloud_clearness(rng: np.random.Generator, n: int, mean_gap: float, mean_dur: float) -> np.ndarray:
    """
    Two-state (clear / cloudy) clearness index time series in (0, 1].

    Clear minutes are not perfectly clear (thin-cirrus texture) so the forecaster
    always has residual to learn; cloud minutes are attenuated by a random factor
    drawn from CLOUD_ATTEN_RANGE and stay cloudy for a geometrically distributed
    number of minutes, which produces the characteristic "cloud gaps".
    """
    p_enter = 1.0 - np.exp(-1.0 / max(mean_gap, 1e-9))     # hazard per minute
    clr = np.ones(n)
    i = 0
    while i < n:
        if rng.random() < p_enter:
            dur = int(max(1, rng.geometric(1.0 / max(mean_dur, 1e-9))))
            dur = min(dur, n - i)
            level = rng.uniform(*CLOUD_ATTEN_RANGE)
            ramp = 1.0 - 0.25 * rng.random(dur)              # gap interior texture
            clr[i : i + dur] = level * ramp
            i += dur
        else:
            clr[i] = 1.0 - 0.08 * rng.random()               # thin texture
            i += 1
    return clr


# =============================================================================
# 2b. EXTENDED PHYSICS MODEL  (180-day extension)
# =============================================================================
# Sources: NASA TFAWS 2014 (beta angle + eclipse geometry of a circular LEO
# orbit), NASA ACTS 2005 + MDPI 2025 / MinXSS (array degradation), MinXSS
# XACT spec (attitude / pointing loss), NASA PCoE (battery cycle degradation).


def beta_angle(day_of_year: float) -> float:
    """
    Seasonal orbit beta angle [deg] for a mission day (NASA TFAWS 2014).

    The orbit plane precesses once a year against the Earth-Sun line, so the
    sun's elevation above the orbit plane (the beta angle) swings between
    BETA_ANGLE_MIN and BETA_ANGLE_MAX with a one-year period.  Over the first
    180 days the sine sweeps 0 -> 23.45 -> ~1 deg, i.e. the eclipse season
    slowly opens and closes inside the mission window.
    """
    # beta = BETA_ANGLE_MAX * sin(2*pi * day_of_year / BETA_PERIOD_DAYS)
    # (BETA_ANGLE_MIN is the lower bound of the seasonal swing and is 0 deg.)
    beta = BETA_ANGLE_MAX * np.sin(2.0 * np.pi * day_of_year / BETA_PERIOD_DAYS)
    beta = BETA_ANGLE_MIN + beta
    return float(beta)


def eclipse_fraction(beta_deg: float) -> float:
    """
    Fraction of one orbit spent inside the Earth's umbral cone, in [0, 1]
    (NASA TFAWS 2014, circular-orbit eclipse geometry).

        sin_theta = sqrt( (1/cos^2 beta) * ((Re/(Re+h))^2 - sin^2 beta) )
        theta     = asin(sin_theta)             # umbral half-angle [rad]
        eclipse   = (2*theta) / (2*pi)          # umbral arc / full circle

    If the argument of the square root goes negative the sun never enters the
    umbral cone (end of the eclipse season) and the fraction is 0.

    NOTE on the drafting formula ``1 - (2*theta)/(2*pi)``: with theta = asin()
    above that expression is the *sunlit* fraction, i.e. the complement of the
    umbral arc.  It returns 0.611 at beta = 0 and grows towards 1 as the
    eclipse season closes, which contradicts the "no eclipse season -> 0"
    branch.  The complement, (2*theta)/(2*pi), is the physical eclipse
    fraction: 0.389 at beta = 0, i.e. 36.0 min of the 92.6 min BIRDS-4 orbit
    (ISS-class reality), smoothly decaying to 0 when the season ends.  The
    sunlit fraction is computed first below so both expressions are visible.
    """
    beta = np.radians(beta_deg)
    h = ALTITUDE_KM
    Re = EARTH_RADIUS_KM
    arg = (1.0 / np.cos(beta) ** 2) * ((Re / (Re + h)) ** 2 - np.sin(beta) ** 2)
    if arg < 0.0:
        return 0.0                       # no eclipse season at this beta
    sin_theta = np.sqrt(arg)
    theta = np.arcsin(sin_theta)         # umbral half-angle [rad]
    sunlit_fraction = 1.0 - (2.0 * theta) / (2.0 * np.pi)
    eclipse_fraction = 1.0 - sunlit_fraction   # == (2*theta) / (2*pi)
    return float(np.clip(eclipse_fraction, 0.0, 1.0))


def panel_degradation(day: int) -> float:
    """
    Solar-array power multiplier at mission ``day`` (NASA ACTS 2005 + MDPI
    2025, MinXSS measured 2.8 %/year degradation), linear in time and floored
    at 15 % total loss (factor >= 0.85).
    """
    years = max(day - DEGRADATION_START_DAY, 0) / 365.0
    factor = 1.0 - DEGRADATION_PER_YEAR * years
    return float(max(factor, 0.85))


def attitude_loss(day: int) -> float:
    """
    Fractional harvest loss caused by spacecraft attitude / pointing error at
    mission ``day`` (MinXSS XACT: 25 arcsec worst case).

    Base loss is the cosine loss of POINTING_LOSS_FRACTION, modulated by a
    slow sinusoidal variation over the 180-day mission (attitude knowledge and
    thermal wobble drift with beta angle / season) and bounded to [0, 0.01],
    i.e. at most a 1 % harvest loss as required.
    """
    base = POINTING_LOSS_FRACTION
    loss = base + 0.5 * base * np.sin(2.0 * np.pi * day / 180.0)
    return float(np.clip(loss, 0.0, 0.01))


def battery_capacity(day: int) -> float:
    """
    Battery capacity multiplier at mission ``day`` (NASA PCoE cycle-life
    data): every orbit is roughly one deep cycle, each costing
    BATTERY_CYCLE_DEGRADATION of the nameplate capacity, floored at 30 %
    total loss (factor >= 0.70).
    """
    cycles = day * CYCLES_PER_DAY
    factor = 1.0 - BATTERY_CYCLE_DEGRADATION * cycles
    return float(max(factor, 0.70))


# =============================================================================
# 3. SCENARIO GENERATION  (identical for every strategy in a given trial)
# =============================================================================

class Geometry:
    """
    Pre-computed eclipse/daylight bookkeeping for the (orbit-periodic) ephemeris.

    ``plan_horizon_steps`` and ``dark_window_hours`` are queried once per simulated
    minute by the predictive strategies, so the run lengths of every sunlit and
    dark segment are pre-computed here in O(n) instead of being re-scanned.
    """

    def __init__(self, shape_ext: np.ndarray, n: int,
                 sunrise_min: float = 0.0, daylight_min: float = DAYLIGHT_MIN,
                 treat_all_dark_as_eclipse: bool = False):
        self.shape = shape_ext
        self.n = n
        lit = shape_ext > 0.0
        m = lit.size
        # -- locate the eclipse segments --------------------------------------
        # The eclipse is the dark stretch *embedded inside* the sunlit arc; the
        # trailing orbital night is not an eclipse.  Distinguishing the two needs
        # the daylight bounds, so they are passed in (defaults = the canonical
        # geometry used by make_scenario).  Both the base orbit and the tiled
        # copy are covered.
        # NASA TFAWS 2014 (extended study): the realistic ephemeris has no
        # separate orbital night -- every dark minute of a 92.6 min LEO orbit
        # is eclipse -- so ``treat_all_dark_as_eclipse=True`` classifies every
        # dark run as an eclipse segment and the daylight-bounds test is moot.
        day0 = int(round(sunrise_min / DT_MIN))
        day1 = int(round((sunrise_min + daylight_min) / DT_MIN))
        arcs = [(k * n + day0, k * n + day1) for k in range(2) if k * n + day1 <= m]
        segs: List[Tuple[int, int]] = []
        i = 0
        while i < m:
            if not lit[i] and (i == 0 or lit[i - 1]):        # start of a dark run
                j = i
                while j < m and not lit[j]:
                    j += 1
                if treat_all_dark_as_eclipse or any(a <= i and j <= b for a, b in arcs):
                    segs.append((i, j))                     # [start, end) eclipse
                i = j
            else:
                i += 1
        self.eclipse_segs = segs
        self.ecl_end = np.array([b for _, b in segs], dtype=int)
        self.ecl_start = np.array([a for a, _ in segs], dtype=int)
        # sun_run[i] = number of consecutive lit minutes starting at i
        self.sun_run = np.zeros(m, dtype=int)
        # dark_run[i] = number of consecutive dark minutes starting at i
        self.dark_run = np.zeros(m, dtype=int)
        for arr, flag in ((self.sun_run, True), (self.dark_run, False)):
            run = 0
            for i in range(m - 1, -1, -1):     # single backward pass, O(n)
                run = run + 1 if bool(lit[i]) == flag else 0
                arr[i] = run
        self.hmax = int(PLAN_HORIZON_MIN_MAX / DT_MIN)
        self.hmin = int(PLAN_HORIZON_MIN_MIN / DT_MIN)
        self.tail = int(30.0 / DT_MIN)
        # cumulative sunlit-minute count -> O(1) "how much sun is in the horizon"
        self.lit_cum = np.concatenate([[0], np.cumsum(lit.astype(int))])

    def lit_minutes(self, i: int, steps: int) -> int:
        """Number of sunlit minutes in the window ``i+1 ... i+steps``."""
        a = min(i + 1, self.shape.size)
        b = min(i + 1 + int(steps), self.shape.size)
        return int(self.lit_cum[b] - self.lit_cum[a])

    def minutes_to_eclipse_end(self, i: int) -> Optional[int]:
        """
        Minutes from step ``i`` to the END of the next eclipse segment, or None
        if no eclipse starts within the 24 h window (or the one ahead has already
        finished).  This is the margin's natural look-ahead: the decision that
        matters is "do I still have charge when the sun comes back?".
        """
        if self.ecl_end.size == 0:
            return None
        # first eclipse whose end lies ahead of i
        k = int(np.searchsorted(self.ecl_end, i + 1, side="left"))
        if k >= self.ecl_end.size:
            return None
        return int(self.ecl_end[k] - i)

    def horizon(self, i: int) -> int:
        """
        Look-ahead length (steps): the margin is evaluated at the end of the next
        eclipse, clipped to [PLAN_HORIZON_MIN_MIN, PLAN_HORIZON_MIN_MAX].

        ``horizon = min(PLAN_HORIZON_MIN_MAX, minutes to the end of the next
        eclipse)``.  With no eclipse ahead inside the 24 h window the full
        PLAN_HORIZON_MIN_MAX is used.  The clip is applied per side, so a very
        near eclipse still yields at least PLAN_HORIZON_MIN_MIN minutes of
        look-ahead rather than collapsing to zero.
        """
        to_ecl = self.minutes_to_eclipse_end(i)
        if to_ecl is None:
            return int(self.hmax)
        return int(np.clip(to_ecl, self.hmin, self.hmax))

    def dark_hours(self, i: int) -> float:
        """
        Hours of continuous eclipse/night immediately ahead of step ``i``.

        This is the quantity the payload scheduler must protect: whatever SoC the
        battery holds at eclipse entry has to cover (W_CRITICAL + W_ESSENTIAL) *
        T_dark on top of the reserve.
        """
        if self.sun_run[i] > 0:
            k = min(i + int(self.sun_run[i]), self.shape.size - 1)
            if k >= i + self.hmax:
                return 0.0
        else:
            k = i
        return float(self.dark_run[k]) * DT_H


@dataclass
class Scenario:
    """One randomised orbital day, shared by all six strategies of a trial."""

    trial: int
    seed: int
    capacity_wh: float          # after fade
    nameplate_wh: float
    soc0_wh: float
    fade_frac: float
    eclipse_start_min: float
    eclipse_len_min: float
    sunrise_min: float
    daylight_min: float
    shape: np.ndarray           # [n] clear-sky geometry (ephemeris), 0..1
    clearness: np.ndarray       # [n] realised clearness index, 0..1
    harvest_w: np.ndarray       # [n] realised array power, W
    shape_ext: np.ndarray       # [n + PLAN_HORIZON] tiled ephemeris for look-ahead
    geom: Geometry              # pre-computed segment lengths / plan horizons
    horizon_steps: np.ndarray   # [n] look-ahead length per step (identical per strategy)
    dark_hours: np.ndarray      # [n] hours of continuous eclipse/night ahead
    # ---- extended-physics fields (default = legacy 24 h synthetic study) ----
    # NASA TFAWS 2014 / NASA ACTS 2005 / MDPI 2025 / NASA PCoE.
    orbit_period_min: float = ORBIT_PERIOD_MIN_24H   # [-] EWMA / ephemeris period
    beta_deg: Optional[np.ndarray] = None            # [SIM_DAYS] beta angle per day
    eclipse_frac: Optional[np.ndarray] = None        # [SIM_DAYS] eclipse fraction/day
    harvest_ideal: Optional[np.ndarray] = None       # [n] harvest w/o degradation
    degradation: Optional[np.ndarray] = None         # [n] panel degradation factor
    attitude: Optional[np.ndarray] = None            # [n] attitude loss fraction

    @property
    def n(self) -> int:
        return int(self.harvest_w.size)


def make_scenario(trial: int) -> Scenario:
    """
    Build a randomised scenario.  The RNG is seeded per trial so that all six
    strategies face exactly the same weather, eclipse geometry, initial SoC and
    battery fade (requirement: fair Monte-Carlo comparison).
    """
    seed = BASE_SEED + 1009 * trial
    rng = np.random.default_rng(seed)

    n = N_STEPS
    # -- randomised scenario parameters -----------------------------------
    cloud_gap = rng.uniform(*CLOUD_GAP_RANGE)      # mean clear-sky gap length
    cloud_dur = rng.uniform(*CLOUD_DUR_RANGE)      # mean cloud-gap duration
    soc0_frac = rng.uniform(*SOA0_RANGE)           # initial state of charge
    fade_frac = rng.uniform(*FADE_RANGE)           # capacity fade (aging / DoD)

    # -- eclipse placement inside the daylight arc -------------------------
    # The window opens at sunrise; the eclipse starts at a random offset inside
    # the daylight arc, so every trial gets the same 12 h of sunlit time but a
    # different eclipse/night geometry to plan around.
    ecl_len = ECLIPSE_MIN_24H
    ecl_start = rng.uniform(*ECLIPSE_OFFSET_RANGE)
    sunrise = 0.0
    daylight = DAYLIGHT_MIN

    shape = clear_sky_shape(n, sunrise, daylight, (ecl_start, ecl_start + ecl_len))
    clearness = cloud_clearness(rng, n, cloud_gap, cloud_dur)
    harvest = PANEL_PEAK_W * shape * clearness

    capacity = BAT_CAPACITY_WH * (1.0 - fade_frac)
    soc0 = soc0_frac * capacity

    # Tiled ephemeris: lets a planner look past the end of the 24 h window
    # (the orbit repeats), which is legitimate for a sun-synchronous orbit.
    ext = int(PLAN_HORIZON_MIN_MAX / DT_MIN)
    shape_ext = np.concatenate([shape, shape])[: n + ext + 1]
    geom = Geometry(shape_ext, n, sunrise_min=sunrise, daylight_min=daylight)
    # every strategy must see the same planning horizons
    horizon_steps = np.array([geom.horizon(i) for i in range(n)], dtype=int)
    dark_hours = np.array([geom.dark_hours(i) for i in range(n)], dtype=float)

    return Scenario(
        trial=trial,
        seed=seed,
        capacity_wh=capacity,
        nameplate_wh=BAT_CAPACITY_WH,
        soc0_wh=soc0,
        fade_frac=fade_frac,
        eclipse_start_min=ecl_start,
        eclipse_len_min=ecl_len,
        sunrise_min=sunrise,
        daylight_min=daylight,
        shape=shape,
        clearness=clearness,
        harvest_w=harvest,
        shape_ext=shape_ext,
        geom=geom,
        horizon_steps=horizon_steps,
        dark_hours=dark_hours,
    )


def make_extended_scenario(trial: int) -> Scenario:
    """
    Build a randomised 180-day scenario with realistic orbital mechanics.

    This is the extended-study counterpart of ``make_scenario``.  It reuses the
    same per-trial seeding so all strategies face identical weather, eclipse
    geometry, initial SoC and degradation, but it spans ``N_STEPS_TOTAL`` minutes
    and folds in the three slow physical effects (requirements 1-3, 5-8, 14-15):

      * orbit geometry   - the eclipse fraction is recomputed from the seasonal
                           beta angle *every day* and applied as one contiguous
                           block per 92.6 min orbit at a fixed phase (NASA TFAWS
                           2014);
      * array            - 2.8 %/yr panel degradation plus the MinXSS XACT
                           pointing loss are multiplied into the realised harvest
                           (NASA ACTS 2005 + MDPI 2025, MinXSS);
      * battery          - capacity fade is not baked into the scenario -- it is
                           applied per step in ``simulate_extended_trial`` so the
                           SoC fraction uses the instantaneous capacity (NASA
                           PCoE).

    The nested ``day`` / ``minute`` loop below is exactly the structure required
    by the study (one eclipse window per orbit, recomputed each day) and builds
    both the (noisy) realised harvest and the ideal, un-degraded harvest used by
    the pre/post-degradation comparison plot.
    """
    seed = BASE_SEED + 1009 * trial
    rng = np.random.default_rng(seed)

    n = N_STEPS_TOTAL
    period = ORBIT_PERIOD_MIN

    # -- randomised scenario parameters (same distributions as the 24 h study) --
    cloud_gap = rng.uniform(*CLOUD_GAP_RANGE)      # mean clear-sky gap length
    cloud_dur = rng.uniform(*CLOUD_DUR_RANGE)      # mean cloud-gap duration
    soc0_frac = rng.uniform(*SOA0_RANGE)           # initial state of charge

    # -- per-day orbital geometry (NASA TFAWS 2014) --------------------------
    # beta_angle() sweeps 0 -> 23.45 -> ~0 deg over the 180 days, so the eclipse
    # fraction it drives slowly opens and closes across the mission.
    beta_day = np.array([beta_angle(d) for d in range(SIM_DAYS)], dtype=float)
    ecl_frac_day = np.array([eclipse_fraction(b) for b in beta_day], dtype=float)
    ecl_min_day = ecl_frac_day * period             # [min] eclipse per orbit, per day

    # -- ephemeris + harvest (nested day / minute loop, requirement 12) -------
    # ``shape`` is the clear-sky geometry over the whole mission *plus* a tail of
    # PLAN_HORIZON_MIN_MAX minutes so the planner can look past the last simulated
    # minute; the tail simply continues the same orbit geometry.
    ext = int(PLAN_HORIZON_MIN_MAX / DT_MIN)
    shape = np.zeros(n + ext + 1)
    clearness = cloud_clearness(rng, n, cloud_gap, cloud_dur)
    harvest = np.zeros(n)          # realised (degraded) array power [W]
    harvest_ideal = np.zeros(n)    # pre-degradation array power [W]
    degradation = np.ones(n)       # panel degradation factor per step
    attitude = np.zeros(n)         # attitude loss fraction per step

    for day in range(SIM_DAYS):
        # Requirement 4: eclipse length is recomputed from the day's beta angle.
        ecl_minutes = ecl_min_day[day]
        sun_arc = max(period - ecl_minutes, 1e-9)
        deg_day = panel_degradation(day)            # NASA ACTS 2005 / MDPI 2025
        att_day = attitude_loss(day)                # MinXSS XACT pointing loss
        for minute in range(STEPS_PER_DAY):
            step = day * STEPS_PER_DAY + minute
            # Orbit phase within the 92.6 min orbit: one contiguous eclipse
            # starting at ECLIPSE_PHASE_MIN (fixed orbital phase), then sun.
            phase = (step * DT_MIN - ECLIPSE_PHASE_MIN) % period
            if phase < ecl_minutes:
                sun = 0.0                            # inside the umbral eclipse
            else:
                sun = float(np.sin(np.pi * (phase - ecl_minutes) / sun_arc))
            shape[step] = sun
            # Ideal harvest (geometry x cloud), then the two slow losses applied
            # to the realised harvest exactly as required (steps 7 and 10).
            ideal = PANEL_PEAK_W * sun * clearness[step]
            harvest_ideal[step] = ideal
            degradation[step] = deg_day
            attitude[step] = att_day
            harvest[step] = ideal * deg_day * (1.0 - att_day)

    # -- plan the orbit-geometry tail (only the ephemeris is needed there) -----
    for step in range(n, n + ext + 1):
        day = step // STEPS_PER_DAY
        ecl_minutes = eclipse_fraction(beta_angle(day)) * period
        sun_arc = max(period - ecl_minutes, 1e-9)
        phase = (step * DT_MIN - ECLIPSE_PHASE_MIN) % period
        shape[step] = 0.0 if phase < ecl_minutes else float(
            np.sin(np.pi * (phase - ecl_minutes) / sun_arc))

    # -- battery / capacity bookkeeping --------------------------------------
    capacity = BAT_CAPACITY_WH * battery_capacity(0)   # day-0 (fresh) capacity
    soc0 = soc0_frac * capacity

    # -- ephemeris classification + per-step planning look-ahead --------------
    # In the realistic LEO ephemeris every dark minute is eclipse, so the
    # daylight-embedded-eclipse test is replaced by treat_all_dark_as_eclipse.
    geom = Geometry(shape, n, treat_all_dark_as_eclipse=True)
    horizon_steps = np.array([geom.horizon(i) for i in range(n)], dtype=int)
    dark_hours = np.array([geom.dark_hours(i) for i in range(n)], dtype=float)

    return Scenario(
        trial=trial,
        seed=seed,
        capacity_wh=capacity,
        nameplate_wh=BAT_CAPACITY_WH,
        soc0_wh=soc0,
        fade_frac=0.0,
        eclipse_start_min=ECLIPSE_PHASE_MIN,
        eclipse_len_min=float(ecl_min_day[0]),
        sunrise_min=0.0,
        daylight_min=DAYLIGHT_MIN,
        shape=shape[:n],
        clearness=clearness,
        harvest_w=harvest,
        shape_ext=shape,
        geom=geom,
        horizon_steps=horizon_steps,
        dark_hours=dark_hours,
        orbit_period_min=period,
        beta_deg=beta_day,
        eclipse_frac=ecl_frac_day,
        harvest_ideal=harvest_ideal,
        degradation=degradation,
        attitude=attitude,
    )


# =============================================================================
# 4. BATTERY + BUS MODEL
# =============================================================================

@dataclass
class BusResult:
    """Result of one power-balance step."""

    soc_wh: float                       # battery SoC after the step
    served_wh: Dict[str, float]         # energy actually delivered per tier
    charged_wh: float
    spilled_wh: float
    critical_met: bool                  # was the Critical tier fully powered?


def bus_step(
    soc_wh: float,
    cap_wh: float,
    load_w: float,
    harvest_w: float,
    enabled: Dict[str, bool],
) -> BusResult:
    """
    Single-minute bus power balance with reverse-priority load shedding.

    Energy in this minute:  ``h = harvest_w * DT_H``
    Energy required:       ``L = load_w * DT_H``

    * ``h >= L`` : the surplus ``s = h - L`` charges the battery (x eta_chg);
      anything beyond full charge is curtailed (spill).
    * ``h <  L``  : the battery supplies the deficit ``d = L - h`` (x 1/eta_dis).

    If the battery cannot cover the demand, loads are dropped in *reverse
    priority order* (Deferrable, then Essential; Critical is last) and the
    corresponding `served_wh` reflects the partial supply.  This is what makes
    an unmanaged system brown out the payload -- and, in the worst case, the
    Critical load itself.
    """
    h = float(harvest_w) * DT_H
    soc = float(soc_wh)

    # --- charge side -----------------------------------------------------
    headroom = max(cap_wh - soc, 0.0)
    charge_in = min(h, headroom / ETA_CHG) if h > 0.0 else 0.0
    soc += charge_in * ETA_CHG
    spill = max(h - charge_in, 0.0)

    # --- discharge side --------------------------------------------------
    budget_wh = spill + max(soc, 0.0) * ETA_DIS       # deliverable energy this minute
    served: Dict[str, float] = {}
    demand: Dict[str, float] = {}
    for tier in LOADS:                                 # declared order is critical-first
        demand[tier] = (LOAD_W[tier] if enabled.get(tier, True) else 0.0) * DT_H
    # consume the budget from the least important tier upwards
    for tier in PRIORITY_ORDER:
        d = demand[tier]
        take = min(d, budget_wh)
        served[tier] = take
        budget_wh -= take
        demand[tier] = d - take
    from_batt_wh = max(sum(served.values()) - spill, 0.0)
    soc -= from_batt_wh / ETA_DIS
    soc = float(np.clip(soc, 0.0, cap_wh))

    crit_met = served["critical"] >= LOAD_W["critical"] * DT_H - 1e-12
    return BusResult(soc, served, charge_in, spill, crit_met)


# =============================================================================
# 5. HARVEST FORECASTERS  (seasonal EWMA  +  learned residual correction)
# =============================================================================

class SeasonalForecaster:
    """
    Seasonal EWMA harvest forecaster matched to the orbit period.

    Model
    -----
        harvest(t)  = PEAK * clearness(t) * shape(t)                 (truth)
        clearness(t) = level(t) + dev(t)                              (model)

    ``shape(t)`` is the orbit-periodic solar geometry -- the *ephemeris*, known
    in advance.  ``level`` is a slow EWMA of the clearness whose time constant is
    matched to the orbit period, so it represents the seasonal (orbit-averaged)
    cloudiness and strips the diurnal cycle out of the data.  ``dev(t)`` is a
    fast EWMA of the part of the clearness the seasonal term cannot explain
    (i.e. the residual), and the forecast at lead k is

        hhat(t+k) = PEAK * max(0, level + gain(k)*dev + bias_sys(k) - z*sigma(k))
                    * shape(t+k)

    The two learned terms are the **residual correction**:

    * ``gain(k)`` -- how persistent the residual is at lead k.  It is fitted
      online by least squares on the paired samples (deviation when the forecast
      was issued, deviation that actually materialised k minutes later), i.e.
      straight from predicted-vs-actual comparisons.  Cloud gaps are strongly
      autocorrelated, so gain(30 min) is well above zero while gain(4 h) ~ 0.
    * ``bias_sys(k)`` -- a slow EWMA of the realised prediction error at that
      lead, which absorbs whatever systematic offset the seasonal level still
      carries (it lags the true mean clearness badly, since its time constant is
      a whole orbit).

    With ``use_residual=False`` (rung 4) both terms are frozen at zero, so the
    forecast is the bare seasonal envelope and every cloud gap is missed.

    A confidence band (HIGH / MEDIUM / LOW) is derived from the recent absolute
    prediction error over sunlit minutes, and the decision path uses the
    *conservative* bound, i.e. the clearness level is deflated by
    ``z(confidence) * sigma(k)``.
    """

    def __init__(self, shape_ext: np.ndarray, use_residual: bool,
                 orbit_period_min: float = ORBIT_PERIOD_MIN_24H,
                 n_steps: int = N_STEPS):
        self.shape = shape_ext
        self.use_residual = use_residual
        # Seasonal (orbit-matched) EWMA: alpha = 1 - exp(-dt / tau_orbit).
        # tau_orbit == the orbit period makes this a *seasonal* term: it only
        # tracks slow, orbit-scale changes in the average clearness.  The 24 h
        # study passes the synthetic 1440 min orbit, the extension the real
        # 92.6 min BIRDS-4 period.
        self.alpha_level = 1.0 - np.exp(-DT_MIN / orbit_period_min)
        self.alpha_dev = 1.0 - np.exp(-DT_MIN / RESID_TAU_MIN)
        self.alpha_slow = 1.0 - np.exp(-DT_MIN / RESID_SLOW_TAU_MIN)
        self.alpha_sys = 1.0 - np.exp(-DT_MIN / RESID_SYS_TAU_MIN)
        self.alpha_reg = 1.0 - np.exp(-DT_MIN / RESID_REG_TAU_MIN)
        self.level = 1.0                 # seasonal clearness level (init: clear)
        self.fast = 1.0                  # fast clearness EWMA
        self.slow = 1.0                  # medium clearness EWMA
        self.dev = 0.0                   # cloud-regime anomaly (fast - slow)
        # per-lead learned quantities, all in clearness units
        self.gain: Dict[int, float] = {k: 0.0 for k in TRACKED_LEADS}
        self.bias: Dict[int, float] = {k: 0.0 for k in TRACKED_LEADS}
        self.cov: Dict[int, float] = {k: 0.0 for k in TRACKED_LEADS}
        self.var: Dict[int, float] = {k: 0.0 for k in TRACKED_LEADS}
        self.err_ewma: Dict[int, float] = {k: 0.0 for k in TRACKED_LEADS}
        self.abs_ewma: Dict[int, float] = {k: 0.0 for k in TRACKED_LEADS}
        # forecasts in flight, keyed by (lead, step the forecast is for) -> (fc, dev)
        self.pending: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.mae_sum_w = 0.0             # MAE accumulator in W (sunlit minutes)
        self.mae_n = 0
        self.seen = False                # any sunlit minute seen yet?
        # confidence / traces
        self.conf = "LOW"
        self.lead_plot = 30              # [min] lead shown in the error plot
        # Diagnostic traces are sized to the simulated window: 1440 steps for
        # the 24 h study, N_STEPS_TOTAL (259 200) for the 180-day extension.
        self.trace_pred = np.full(n_steps, np.nan)   # point forecast at lead_plot (W)
        self.trace_htot = np.full(n_steps, np.nan)   # total forecast over horizon (Wh)
        self.trace_conf = np.full(n_steps, np.nan)
        self._lk = np.array(sorted(self.gain.keys()), dtype=float)
        self._gv = np.zeros_like(self._lk)
        self._bv = np.zeros_like(self._lk)
        self._sv = np.zeros_like(self._lk)

    # ---- interpolated lead-dependent tables --------------------------------
    def _refresh_tables(self) -> None:
        """Cache the per-lead gain / bias / sigma as arrays for vectorised use."""
        self._gv = np.array([self.gain[int(k)] for k in self._lk])
        self._bv = np.array([self.bias[int(k)] for k in self._lk])
        self._sv = np.array([self.abs_ewma[int(k)] for k in self._lk])

    def _corr_at(self, lead_min) -> np.ndarray:
        """Residual correction gain * dev, interpolated between tracked leads."""
        if not self.use_residual:
            return 0.0
        return self.dev * np.interp(lead_min, self._lk, self._gv)

    def _bias_at(self, lead_min) -> np.ndarray:
        """Learned systematic (error-feedback) correction, clearness units."""
        if not self.use_residual:
            return 0.0
        return np.interp(lead_min, self._lk, self._bv)

    def _sigma_at(self, lead_min) -> np.ndarray:
        """Recent absolute prediction error at a lead, clearness units."""
        return np.interp(lead_min, self._lk, self._sv)

    def level_est(self) -> float:
        return self.level

    def residual_gain(self, lead_min: float = 30.0) -> float:
        """Fitted residual-persistence gain at a given lead (for reporting)."""
        return float(np.interp(lead_min, self._lk, self._gv)) if self.use_residual else 0.0

    # ---- forecasting --------------------------------------------------------
    def predict(self, i: int, lead_min: float, conservative: bool = True) -> float:
        """
        Forecast the array power at absolute step ``i + lead_min`` (W).

        With ``conservative=True`` the lower (conservative) confidence bound is
        returned, i.e. the clearness level is deflated by ``z(conf) * sigma``.
        """
        j = min(i + int(round(lead_min / DT_MIN)), self.shape.size - 1)
        shape_f = float(self.shape[j])
        if shape_f <= 0.0:
            return 0.0
        lvl = self.level
        if conservative:
            lvl = lvl - Z_CONFIDENCE[self.conf] * float(self._sigma_at(lead_min))
        val = PANEL_PEAK_W * (lvl + float(self._corr_at(lead_min)) + float(self._bias_at(lead_min))) * shape_f
        return float(max(val, 0.0))

    def predict_block(self, i: int, steps: int, conservative: bool = True) -> np.ndarray:
        """
        Vectorised forecasts for absolute steps ``i+1 ... i+steps`` (W).

        The seasonal envelope is the *known* ephemeris, so a whole horizon is
        evaluated in one shot: ``PEAK * (level + gain(k)*dev + bias(k) - z*sigma)
        * shape(i+k)``.
        """
        steps = int(min(steps, self.shape.size - i - 1))
        if steps <= 0:
            return np.zeros(0)
        k = np.arange(1, steps + 1, dtype=float) * DT_MIN
        shape_f = self.shape[i + 1 : i + steps + 1]
        lvl = self.level
        if conservative:
            lvl = lvl - Z_CONFIDENCE[self.conf] * self._sigma_at(k)
        val = PANEL_PEAK_W * (lvl + self._corr_at(k) + self._bias_at(k)) * shape_f
        return np.maximum(val, 0.0)

    # ---- online learning ----------------------------------------------------
    def update(self, i: int, shape_now: float, harvest_now: float) -> None:
        """
        Fold the realised harvest into the model and learn from forecast errors.

        All learning happens on *sunlit* minutes only: during eclipse/night the
        harvest is identically zero, which carries no information about the
        clearness and would otherwise wash the error statistics (and hence the
        confidence band) out to a meaningless "HIGH" during the dark.
        """
        lit = shape_now > 0.05
        clr = harvest_now / PANEL_PEAK_W

        # -- seasonal (orbit-matched) level EWMA, and the band-pass anomaly ---
        # level: slow, orbit-averaged clearness (the seasonal term).
        # fast/slow: a band-pass pair of EWMAs of the clearness, so
        #     a = fast - slow
        # isolates the *cloud regime* fluctuation that the seasonal term cannot
        # explain.  It is zero-mean by construction, which is what makes the
        # least-squares persistence fit well conditioned.
        if lit:
            self.level += self.alpha_level * (clr - self.level)
            self.level = float(np.clip(self.level, 0.0, 1.5))
            self.fast += self.alpha_dev * (clr - self.fast)
            self.slow += self.alpha_slow * (clr - self.slow)
        dev_now = (self.fast - self.slow) if lit else 0.0

        # -- realise the forecasts that fall due now, then learn from them -----
        # NOTE: the lookup key must include the lead, otherwise the entry for a
        # long-lead forecast would be overwritten every minute by the (lead = 1)
        # forecast that targets the same step, and only lead 1 would ever be
        # scored.  Statistics are frozen in the dark so the confidence band is
        # not reset by the eclipse.
        for lead in TRACKED_LEADS:
            due = self.pending.pop((lead, int(i)), None)
            if due is None:
                continue
            fc, dev_issue = due
            err = clr - fc / PANEL_PEAK_W               # clearness units
            if not lit:
                continue                                # no information in eclipse
            self.dev = dev_now                          # deviation now vs at issue
            # (1) least-squares fit of the residual persistence gain at this lead
            if dev_issue != 0.0:
                self.cov[lead] += self.alpha_reg * (dev_issue * dev_now - self.cov[lead])
                self.var[lead] += self.alpha_reg * (dev_issue * dev_issue - self.var[lead])
                if self.var[lead] > 1e-6:
                    self.gain[lead] = float(np.clip(self.cov[lead] / self.var[lead], 0.0, 1.0))
            # (2) error statistics -> confidence band and the conservative bound
            a_err = self.alpha_fast_for(lead)
            self.err_ewma[lead] += a_err * (err - self.err_ewma[lead])
            self.abs_ewma[lead] += a_err * (abs(err) - self.abs_ewma[lead])
            # (3) systematic correction from the error feedback loop
            if self.use_residual:
                self.bias[lead] += self.alpha_sys * err
            self.seen = True
            self.mae_sum_w += abs(err) * PANEL_PEAK_W
            self.mae_n += 1

        # -- refresh tables and confidence band -------------------------------
        self._refresh_tables()
        if self.seen:
            ebar = float(np.max(self._sv))              # worst recent |error|
            if ebar < CONF_HIGH:
                self.conf = "HIGH"
            elif ebar < CONF_MED:
                self.conf = "MEDIUM"
            else:
                self.conf = "LOW"
        else:
            self.conf = "LOW"                           # no data yet -> be careful

        # -- register the forecasts we are making at the tracked leads -------
        for k in TRACKED_LEADS:
            self.pending[(k, int(i + k))] = (self.predict(i, float(k), conservative=False), dev_now)

        # -- traces for the predictor-error plot -------------------------------
        self.trace_pred[i] = self.predict(i, float(self.lead_plot), conservative=False)
        self.trace_conf[i] = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[self.conf]

    def alpha_fast_for(self, lead: int) -> float:
        """Error statistics adapt faster at short leads than at long ones."""
        return 1.0 - np.exp(-DT_MIN / (RESID_TAU_MIN * (1.0 + 0.5 * lead / 60.0)))

    def mae_w(self) -> float:
        """Mean absolute prediction error over sunlit minutes at lead 1 min (W)."""
        return self.mae_sum_w / self.mae_n if self.mae_n else float("nan")


# =============================================================================
# 6. PLANNING HELPERS  (horizon selection, dark-window budget)


# =============================================================================
# 7. STRATEGIES
# =============================================================================

@dataclass
class Context:
    """Everything a strategy may observe at one timestep (no future information)."""

    i: int
    soc_wh: float
    cap_wh: float
    harvest_w: float        # measured array power this minute
    shape_now: float        # measured solar geometry (ephemeris is public)
    soc_hist: np.ndarray    # SoC history up to and including i


@dataclass
class Decision:
    """A load-command decision plus the FSM label for logging."""

    enable: Dict[str, bool]
    state: str = "N/A"
    reserve_wh: float = 0.0
    margin_wh: float = float("nan")
    note: str = ""


class Strategy:
    """Base class: stateless bookkeeping, one `step` per simulated minute."""

    name = "BASE"
    short = "base"
    uses_forecast = False

    def reset(self, scen: Scenario) -> None:
        self.scen = scen

    def step(self, ctx: Context) -> Decision:  # pragma: no cover - abstract
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Rung 0: no management
# -----------------------------------------------------------------------------
class NoManagement(Strategy):
    """All tiers commanded on for the whole orbit (the do-nothing floor)."""

    name = "NO_MGMT"
    short = "no management"

    def reset(self, scen: Scenario) -> None:
        super().reset(scen)

    def step(self, ctx: Context) -> Decision:
        return Decision({"critical": True, "essential": True, "deferrable": True}, "ON", 0.0, 0.0, "all on")


# -----------------------------------------------------------------------------
# Rung 1: plain reactive threshold
# -----------------------------------------------------------------------------
class ReactiveThreshold(Strategy):
    """Fixed SoC thresholds, minimal hysteresis (requirement 7, rung 2)."""

    name = "REACTIVE"
    short = "reactive thr"

    def reset(self, scen: Scenario) -> None:
        super().reset(scen)
        self.on = {"critical": True, "essential": True, "deferrable": True}

    def step(self, ctx: Context) -> Decision:
        s = ctx.soc_wh / ctx.cap_wh
        if s < REACT2["d_shed"]:
            self.on["deferrable"] = False
        elif s > REACT2["d_rest"]:
            self.on["deferrable"] = True
        if s < REACT2["e_shed"]:
            self.on["essential"] = False
        elif s > REACT2["e_rest"]:
            self.on["essential"] = True
        st = "ON" if self.on["deferrable"] else ("C+E" if self.on["essential"] else "C")
        return Decision(dict(self.on), st, 0.0, 0.0, "SoC thresholds")


# -----------------------------------------------------------------------------
# Rung 2: reactive with hysteresis, priority tiers, trend, debounce
# -----------------------------------------------------------------------------
class ReactiveHysteresis(Strategy):
    """
    A *well tuned* reactive controller (requirement 7, rung 3).

    It is a fair, competent threshold policy and it uses strictly past/present
    information only:

    * SoC hysteresis band per tier (restore threshold > shed threshold)
    * strict priority tiers: the payload is always shed before Essential
    * a trailing dSoC/dt trend term that sheds a tier early when it is falling
      fast (still purely reactive -- no forecast)
    * debounce: minimum on/off block times to stop payload chattering
    """

    name = "REACTIVE_H"
    short = "reactive hyst"

    def reset(self, scen: Scenario) -> None:
        super().reset(scen)
        self.on = {"critical": True, "essential": True, "deferrable": True}
        self.t_on = 0
        self.t_off = 0

    def _trend(self, ctx: Context) -> float:
        w = REACT3["trend_window"]
        hist = ctx.soc_hist
        if hist.size < w + 1:
            return 0.0
        return (hist[-1] - hist[-1 - w]) / float(w)      # [Wh/min]

    def step(self, ctx: Context) -> Decision:
        s = ctx.soc_wh / ctx.cap_wh
        tr = self._trend(ctx)
        falling = tr < -REACT3["trend_w"]
        P = REACT3

        # --- payload / deferrable tier ----------------------------------
        if self.on["deferrable"]:
            self.t_on += 1
            self.t_off = 0
            shed = s < P["d_shed"] or (s < P["d_rest"] and falling)
            if shed and self.t_on >= P["min_on"]:
                self.on["deferrable"] = False
                self.t_on = 0
        else:
            self.t_off += 1
            self.t_on = 0
            if s > P["d_rest"] and not falling and self.t_off >= P["min_off"]:
                self.on["deferrable"] = True
                self.t_off = 0

        # --- essential tier (only after the payload is gone) -------------
        if self.on["essential"]:
            shed = s < P["e_shed"] or (s < P["e_rest"] and falling)
            if shed:
                self.on["essential"] = False
        else:
            if s > P["e_rest"] and not falling and self.on["deferrable"] is False:
                self.on["essential"] = True

        st = "ON" if self.on["deferrable"] else ("C+E" if self.on["essential"] else "C")
        return Decision(dict(self.on), st, 0.0, 0.0, f"SoC={s:.2f} trend={tr:+.2f}")


# -----------------------------------------------------------------------------
# Rung 3/4: predictive FSM
# -----------------------------------------------------------------------------
FSM_STATES = ("NOMINAL", "DEGRADED", "CRITICAL", "SAFE_MODE", "RECOVERY")
FSM_STATE_ID = {s: k for k, s in enumerate(FSM_STATES)}
# escalation order: protection is instantaneous, recovery is debounced
FSM_SEVERITY = {"NOMINAL": 0, "RECOVERY": 1, "DEGRADED": 2, "CRITICAL": 3, "SAFE_MODE": 4}


class PredictiveFSM(Strategy):
    """
    Look-ahead (predictive) energy manager with a 5-state FSM.

    Decision variable (requirement 5), evaluated over the planning horizon H:

        margin = SoC_now + forecast_harvest(H) - forecast_load(H) - reserve
        reserve = 20 % of capacity, raised to 30 % when confidence == LOW
                  (and to 30 % in DEGRADED as an extra margin)

        margin <    0  -> shed Deferrable                -> DEGRADED
        margin <  -50  -> shed Essential                  -> CRITICAL
        margin < -100  -> only Critical, charging kept    -> SAFE_MODE

    ``forecast_harvest`` is the *conservative* confidence bound of the seasonal
    EWMA forecaster (rung 3 = plain EWMA, rung 4 = EWMA + residual correction)
    and ``forecast_load`` assumes every currently enabled tier keeps running.

    On top of the FSM, a **deferrable scheduler** decides when the payload is
    worth running: it is enabled only in energy-rich windows, i.e. when

        (a) the SoC already covers Critical + Essential through the coming
            eclipse/night plus the reserve, and
        (b) the conservative 30-minute forecast surplus is non-negative, or the
            battery holds enough extra energy for a 45 min payload block.

    The FSM adds hysteresis (exit thresholds offset by +FSM_HYST_WH) and debounce
    (a condition has to hold for FSM_DEBOUNCE_MIN, plus a minimum dwell time),
    and RECOVERY performs a staged return: Essential first, payload last.
    """

    uses_forecast = True

    def __init__(self, use_residual: bool, label: str, name: str):
        self.use_residual = use_residual
        self.label = label
        self.name = name
        self.short = label

    def reset(self, scen: Scenario) -> None:
        super().reset(scen)
        # The seasonal EWMA time constant follows the scenario's orbit period
        # (1440 min for the synthetic 24 h study, 92.6 min for the extension)
        # and the diagnostic traces are sized to the simulated window.
        self.fc = SeasonalForecaster(
            scen.shape_ext,
            use_residual=self.use_residual,
            orbit_period_min=getattr(scen, "orbit_period_min", ORBIT_PERIOD_MIN_24H),
            n_steps=scen.n,
        )
        self.state = "NOMINAL"
        self.state_t = 0                  # minutes spent in the current state
        self.cand = "NOMINAL"             # debounce candidate
        self.cand_t = 0
        self.rec_stage = 0
        self.rec_ess_hold = 0.0
        self.rec_def_hold = 0.0
        self.rec_bad = 0                  # consecutive negative-margin minutes
        self.e_shed_latched = False
        self._d_on = False                # payload commanded on at the previous step
        # per-step debug trace (used by the step-60 diagnostic print)
        self.dbg_horizon = 0
        self.dbg_lit_min = 0
        self.dbg_f_harv = 0.0
        self.dbg_f_load = 0.0
        self.h_conf = "LOW"
        self.h_margin = float("nan")
        self.h_reserve = 0.0

    # -- forecast hooks (overridden by the perfect-knowledge oracle) -----
    def _forecast_block(self, i: int, H: int, conservative: bool) -> np.ndarray:
        """Harvest forecast over the planning horizon, [W] per minute."""
        return self.fc.predict_block(i, H, conservative=conservative)

    def _confidence(self) -> str:
        return self.fc.conf

    # -- FSM plumbing --------------------------------------------------------
    def _goto(self, new: str) -> None:
        if new != self.state:
            self.state = new
            self.state_t = 0
            self.cand = new
            self.cand_t = 0
        if new != "RECOVERY":
            self.rec_stage = 0
            self.rec_ess_hold = 0.0
            self.rec_def_hold = 0.0

    def _debounced(self, target: str) -> bool:
        """
        Debounced state transition.

        Escalation (a *more* severe state) is applied immediately -- protection
        must never wait for a debounce window.  Relaxation requires the new
        condition to persist for FSM_DEBOUNCE_MIN *and* the minimum dwell time
        FSM_MIN_DWELL_MIN in the current state, which is what stops the manager
        from chattering between tiers around a threshold.
        """
        if target == self.state:
            self.cand, self.cand_t = target, 0
            return False
        if FSM_SEVERITY[target] > FSM_SEVERITY[self.state]:
            self._goto(target)
            return True
        if target == self.cand:
            self.cand_t += 1
        else:
            self.cand, self.cand_t = target, 1
        if self.cand_t >= FSM_DEBOUNCE_MIN and self.state_t >= FSM_MIN_DWELL_MIN:
            self._goto(target)
            return True
        return False

    # -- one control step ----------------------------------------------------
    def step(self, ctx: Context) -> Decision:
        scen = self.scen
        i, cap, soc = ctx.i, ctx.cap_wh, ctx.soc_wh
        socf = soc / cap

        # --- forecast update (learn from the harvest we just measured) ----
        self.fc.update(i, ctx.shape_now, ctx.harvest_w)

        # --- planning horizon and forecasts --------------------------------
        H = int(self.scen.horizon_steps[i])
        pred_block = self._forecast_block(i, H, conservative=True)        # W, per step
        f_harvest_wh = float(pred_block.sum()) * DT_H
        # point (non-conservative) total over the same horizon, for the MAE
        self.fc.trace_htot[i] = float(self._forecast_block(i, H, conservative=False).sum()) * DT_H

        enable_e = not self.e_shed_latched
        # The load forecast is *scheduled*, not "everything stays on": the
        # Critical + Essential base load runs for the whole horizon, while the
        # payload can only be scheduled in the sunlit minutes inside it (it is
        # not runnable in the dark, and shedding it is precisely what the
        # scheduler is about to do if the margin is tight).
        base_w = W_CRITICAL + (W_ESSENTIAL if enable_e else 0.0)
        # Lit minutes *inside the horizon only* (Geometry.lit_minutes already
        # clips the window to i+1 .. i+H); the extra clip guards the invariant
        # explicitly.  Eclipse and night minutes are never lit, so the payload
        # term is zero across any eclipse the horizon contains.
        lit_min = min(scen.geom.lit_minutes(i, H), int(H))
        self.dbg_lit_min = int(lit_min)
        # Critical + Essential are booked for the whole horizon.  Deferrable is
        # booked only while the payload gate is open, and only over the sunlit
        # minutes in the horizon: booking it at full duty for every lit minute
        # kept the margin negative whenever the payload was shed, which latched
        # the FSM in DEGRADED/RECOVERY and locked the gate shut.
        f_load_wh = base_w * H * DT_H
        if self._d_on:
            f_load_wh += W_DEFERRABLE * lit_min * DT_H
        self.dbg_horizon = int(H)
        self.dbg_f_load = float(f_load_wh)
        self.dbg_f_harv = float(f_harvest_wh)

        # --- reserve, confidence, margin -----------------------------------
        conf = self._confidence()
        reserve = cap * (RESERVE_FRAC_LOW_CONF if conf == "LOW" else RESERVE_FRAC)
        if self.state == "DEGRADED":
            reserve = max(reserve, cap * RESERVE_FRAC_DEGRADED)
        margin = soc + f_harvest_wh - f_load_wh - reserve
        self.h_conf, self.h_margin, self.h_reserve = conf, margin, reserve

        # --- margin-driven escalation (requirement 5) ---------------------
        target = self.state
        if self.state == "SAFE_MODE":
            target = "RECOVERY" if (socf >= FSM_SAFE_SOC_CLEAR and margin >= 0.0) else "SAFE_MODE"
        elif self.state == "RECOVERY":
            target = self._recovery_target(socf, margin)
        else:
            if margin < MARGIN_CRITICAL_WH:
                target = "SAFE_MODE"
            elif margin < MARGIN_SHED_ESS_WH:
                target = "CRITICAL"
            elif margin < MARGIN_SHED_DEFER_WH:
                target = "DEGRADED"
            else:
                target = "NOMINAL"

        # a hard SoC floor promotes straight to SAFE_MODE (deep-discharge guard)
        if socf < DEEP_DISCH_SOC and self.state != "SAFE_MODE":
            target = "SAFE_MODE"
        self._debounced(target)

        # --- load set implied by the state (hysteresis on the way out) -----
        e_on = True
        d_on = False
        if self.state in ("NOMINAL", "DEGRADED"):
            e_on = True
            d_on = self.state == "NOMINAL"
        elif self.state in ("CRITICAL", "SAFE_MODE"):
            e_on = False
            d_on = False
        elif self.state == "RECOVERY":
            e_on = self.rec_stage >= 1          # staged return: Essential first ...
            d_on = self.rec_stage >= 1          # ... payload allowed from stage 1,
            #                                       still under the recovery SoC /
            #                                       margin conditions enforced in
            #                                       _recovery_target and the gate
        # the Essential shed is a latch: it survives a margin blip so the bus
        # cannot flap, and is only cleared by a state that tolerates Essential
        if not e_on:
            self.e_shed_latched = True
        elif self.state != "SAFE_MODE":
            self.e_shed_latched = False

        # --- deferrable scheduler (energy-rich windows only) ---------------
        if d_on:
            d_on = self._payload_gate(i, soc, reserve, pred_block)

        self._d_on = bool(d_on)            # remember the commitment for the next margin
        self.state_t += 1
        return Decision(
            {"critical": True, "essential": (not self.e_shed_latched), "deferrable": bool(d_on)},
            self.state,
            reserve,
            margin,
            f"{self.state} conf={conf} margin={margin:+.0f}Wh",
        )

    # -- RECOVERY: staged return to NOMINAL ---------------------------------
    def _recovery_target(self, socf: float, margin: float) -> str:
        # A single negative-margin minute must not wipe the staged recovery: the
        # margin legitimately blips for one or two minutes as the forecast
        # updates.  Only a sustained deficit (FSM_DEBOUNCE_MIN) resets the stage,
        # which is what previously left the FSM parked in RECOVERY.
        if margin < 0.0:
            self.rec_bad += 1
        else:
            self.rec_bad = 0
        sustained_bad = self.rec_bad >= FSM_DEBOUNCE_MIN
        if socf < FSM_RECOVERY_SOC_ESS or sustained_bad:
            self.rec_stage = 0
            return "RECOVERY"
        if self.rec_stage == 0:
            self.rec_ess_hold += DT_MIN
            if self.rec_ess_hold >= FSM_RECOVERY_HOLD_ESS_MIN:
                self.rec_stage = 1
        # The payload re-enables one stage earlier (rec_stage >= 1) rather than
        # waiting for the full NOMINAL stage, so the scheduler can start using
        # the surplus while the FSM is still relaxing.
        if self.rec_stage >= 1:
            if socf < FSM_RECOVERY_SOC_DEF or margin < 0.0:
                return "RECOVERY"
            self.rec_def_hold += DT_MIN
            if self.rec_def_hold >= FSM_RECOVERY_HOLD_DEF_MIN:
                self.rec_stage = 2
        return "NOMINAL" if self.rec_stage >= 2 else "RECOVERY"

    # -- payload gate --------------------------------------------------------
    def _payload_gate(self, i: int, soc: float, reserve: float, pred_block: np.ndarray) -> bool:
        """
        True when running the payload now is justified by the energy budget.

        (a) hard safety: the SoC must already cover Critical + Essential through
            the coming eclipse/night, plus the reserve;
        (b) opportunism: the conservative 30-min forecast surplus must pay for
            the payload, or the battery must hold an extra PAYLOAD_GATE_HOLD_MIN
            worth of payload energy above (a).
        """
        T_dark_h = float(self.scen.dark_hours[i])
        base_w = W_CRITICAL + W_ESSENTIAL
        dark_req_wh = base_w * T_dark_h + reserve
        if soc < dark_req_wh:
            return False
        k = min(int(PAYLOAD_GATE_LEAD_MIN / DT_MIN), pred_block.size)
        surplus_wh = float(pred_block[:k].sum()) * DT_H - (base_w + W_DEFERRABLE) * (k * DT_H)
        if surplus_wh >= 0.0:
            return True
        return soc >= dark_req_wh + W_DEFERRABLE * PAYLOAD_GATE_HOLD_MIN / 60.0


# -----------------------------------------------------------------------------
# Rung 5: oracle -- predictive FSM with perfect future knowledge
#
# DEPRECATED -- no longer part of the comparison; NOT in STRATEGY_REGISTRY.
#
# Reason: the oracle is *not* an upper bound in the information sense for this
# controller, because of a horizon-truncation artifact.  Geometry.horizon is
# ``min(PLAN_HORIZON_MIN_MAX, minutes to next eclipse end)``, so the margin window
# terminates at eclipse exit and the trailing orbital night (the DAYLIGHT_MIN ->
# N_STEPS dark segment) is never inside the margin.  Geometry.dark_hours, the
# payload gate's protection term, likewise only measures the *next contiguous*
# dark run, so the tail night is not added there either.  With a truncated
# window, forecast pessimism is load-bearing: the PREDICT rungs' conservative
# ``z * sigma`` deflation and their 30 % LOW-confidence reserve act as an
# accidental hedge against the truncation, while the exact-information oracle
# spends its (exact) margin right down to 0 % SoC and browns out the Critical
# load in the tail night.  Measured over 50 trials: the oracle lost critical
# uptime to PREDICT_5 in trials 5, 39 and 42 (-3.3 / -6.6 / -3.5 points) while
# winning 34 / 50 / 102 Wh more payload in exactly those trials, and ended up
# *below* PREDICT_5 on mean uptime (99.42 % vs 99.67 %).
#
# The class is kept for reference and for targeted experiments; construct it
# directly (``Oracle()``) rather than via ``make_strategy``.
# -----------------------------------------------------------------------------
class Oracle(PredictiveFSM):
    """
    Perfect-knowledge upper bound.

    DEPRECATED: see the note above -- excluded from the reported comparison.

    This is the predictive FSM (same reserve, same margin thresholds, same
    payload gate, same 5-state machine) with exactly one change: the harvest
    forecast is the *true realised* harvest array.  No forecaster, no confidence
    band -- full confidence -- so the oracle knows the future perfectly and the
    margin is exact.
    """

    name = "ORACLE"
    short = "oracle"

    def __init__(self, use_residual: bool = True, label: str = "oracle", name: str = "ORACLE"):
        super().__init__(use_residual=use_residual, label=label, name=name)

    def reset(self, scen: Scenario) -> None:
        # The oracle still creates the seasonal forecaster so the shared trial
        # harness can keep its learning/trace bookkeeping, but the *decision*
        # never touches it: _forecast_block returns the true realised array.
        super().reset(scen)

    def _forecast_block(self, i: int, H: int, conservative: bool) -> np.ndarray:
        """True realised harvest over the planning horizon, [W] per minute."""
        return self.scen.harvest_w[i : i + H]

    def _confidence(self) -> str:
        return "HIGH"


STRATEGY_REGISTRY: List[Tuple[str, type]] = [
    ("NO_MGMT", NoManagement),
    ("REACTIVE", ReactiveThreshold),
    ("REACTIVE_H", ReactiveHysteresis),
    ("PREDICT_4", lambda: PredictiveFSM(use_residual=False, label="predictive EWMA", name="PREDICT_4")),
    ("PREDICT_5", lambda: PredictiveFSM(use_residual=True, label="predictive +resid", name="PREDICT_5")),
    # Rung 5 (ORACLE) is deliberately NOT registered: see the deprecation note on
    # the Oracle class.  Dropping it here also drops it from the Monte-Carlo
    # sweep, the metrics table, the head-to-head block, the CSVs and every plot.
]
STRATEGY_KEYS = [k for k, _ in STRATEGY_REGISTRY]
STRATEGY_LABELS = {
    "NO_MGMT": "0 no mgmt",
    "REACTIVE": "1 reactive thr",
    "REACTIVE_H": "2 reactive hyst",
    "PREDICT_4": "3 pred EWMA",
    "PREDICT_5": "4 pred +resid",
    "ORACLE": "5 oracle",
}


def make_strategy(key: str) -> Strategy:
    for k, ctor in STRATEGY_REGISTRY:
        if k == key:
            return ctor()
    raise KeyError(key)


# =============================================================================
# 8. TRIAL SIMULATION
# =============================================================================

@dataclass
class TrialResult:
    """Full time history + metrics for one (scenario, strategy) pair."""

    strategy: str
    trial: int
    soc: np.ndarray                 # [n] SoC after each minute, Wh
    soc_frac: np.ndarray            # [n] SoC fraction
    served: Dict[str, np.ndarray]   # [n] delivered Wh per tier
    enabled: Dict[str, np.ndarray]  # [n] commanded on/off (bool) per tier
    states: np.ndarray              # [n] FSM state index (-1 if n/a)
    margin: np.ndarray              # [n] decision margin, Wh (nan if n/a)
    reserve: np.ndarray             # [n] reserve, Wh (nan if n/a)
    pred: Optional[np.ndarray]      # [n] forecaster trace at lead 30 min (W)
    conf: Optional[np.ndarray]      # [n] confidence band index
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.soc.size)


def simulate_trial(scen: Scenario, strategy: Strategy) -> TrialResult:
    """
    Run one 24 h orbit for one strategy on one scenario.

    The loop is the physical core: measure the array, let the strategy decide the
    load set, balance the bus with reverse-priority shedding, update SoC, then
    update the forecaster with what actually happened.
    """
    strat = strategy
    strat.reset(scen)
    n = scen.n
    cap = scen.capacity_wh

    soc = np.empty(n)
    served = {t: np.empty(n) for t in LOADS}
    enabled = {t: np.empty(n, dtype=bool) for t in LOADS}
    states = np.full(n, -1, dtype=int)
    margin = np.full(n, np.nan)
    reserve = np.full(n, np.nan)
    soc_hist = np.empty(n)
    pred_trace = np.full(n, np.nan) if getattr(strat, "uses_forecast", False) else None
    conf_trace = np.full(n, np.nan) if pred_trace is not None else None

    # brownout / deep-discharge episode latches
    bo_latch = False
    dd_latch = False
    n_brownout = 0
    n_deep = 0
    n_crit_down = 0
    switches = {"critical": 0, "essential": 0, "deferrable": 0}
    prev_en = {"critical": True, "essential": True, "deferrable": True}
    dsum = np.zeros(n)     # realised harvest energy over the planning horizon
    payload_wh = 0.0
    crit_ok = 0
    ess_ok = 0
    htot = np.full(n, np.nan)   # forecast energy over the planning horizon

    s = float(scen.soc0_wh)
    for i in range(n):
        soc_hist[i] = s
        ctx = Context(i=i, soc_wh=s, cap_wh=cap, harvest_w=float(scen.harvest_w[i]),
                      shape_now=float(scen.shape[i]), soc_hist=soc_hist[: i + 1])
        dec = strat.step(ctx)

        load_w = sum(LOAD_W[t] for t in LOADS if dec.enable.get(t, True))
        res = bus_step(s, cap, load_w, ctx.harvest_w, dec.enable)

        s = res.soc_wh
        soc[i] = s
        for t in LOADS:
            served[t][i] = res.served_wh[t]
            on = bool(dec.enable.get(t, True))
            enabled[t][i] = on
            if on != prev_en[t]:
                switches[t] += 1
                prev_en[t] = on
        if dec.state in FSM_STATE_ID:
            states[i] = FSM_STATE_ID[dec.state]
        margin[i] = dec.margin_wh
        reserve[i] = dec.reserve_wh
        if pred_trace is not None:
            pred_trace[i] = strat.fc.trace_pred[i]
            conf_trace[i] = strat.fc.trace_conf[i]
            htot[i] = strat.fc.trace_htot[i]

        # --- bookkeeping / event detection -------------------------------
        f = s / cap
        crit_ok += int(res.critical_met)
        ess_ok += int(res.served_wh["essential"] >= LOAD_W["essential"] * DT_H - 1e-12)
        payload_wh += res.served_wh["deferrable"]
        if f < BROWNOUT_SOC and not bo_latch:
            bo_latch = True
            n_brownout += 1
        elif f >= BROWNOUT_SOC + 0.01:
            bo_latch = False
        if f < DEEP_DISCH_SOC and not dd_latch:
            dd_latch = True
            n_deep += 1
        elif f >= DEEP_DISCH_SOC + 0.01:
            dd_latch = False
        if not res.critical_met:
            n_crit_down += 1

        # realised harvest over the planning horizon (for the forecast MAE)
        H = int(scen.horizon_steps[i])
        j = min(n, i + H)
        dsum[i] = float(scen.harvest_w[i:j].sum()) * DT_H

    # --- metrics ------------------------------------------------------------
    res_obj = TrialResult(
        strategy=strat.name, trial=scen.trial, soc=soc, soc_frac=soc / cap, served=served,
        enabled=enabled, states=states, margin=margin, reserve=reserve,
        pred=pred_trace, conf=conf_trace,
    )
    res_obj.metrics = compute_metrics(
        res_obj, scen, n_brownout, n_deep, n_crit_down, switches, payload_wh, crit_ok, ess_ok
    )
    # horizon-total forecast error (Wh) -- scored against the realised array
    if pred_trace is not None:
        sunlit = (scen.shape > 0.05) & (np.arange(n) + scen.horizon_steps <= n) & np.isfinite(htot)
        res_obj.metrics["pred_mae_horizon_wh"] = float(np.abs(htot[sunlit] - dsum[sunlit]).mean())
        res_obj.metrics["pred_mae_w"] = float(strat.fc.mae_w())
    else:
        res_obj.metrics["pred_mae_horizon_wh"] = float("nan")
        res_obj.metrics["pred_mae_w"] = float("nan")
    return res_obj


def simulate_extended_trial(scen: Scenario, strategy: Strategy) -> TrialResult:
    """
    Run the 180-day mission for one strategy on one extended scenario.

    Identical in structure to ``simulate_trial`` except for the two extended
    physics hooks mandated by the study:

      * the battery capacity is no longer constant -- each minute uses
        ``BAT_CAPACITY_WH * battery_capacity(day)`` (NASA PCoE cycle fade), so
        the bus balance, the SoC fraction and the ''is the bus in brownout?''
        test all track the instantaneous capacity (requirement 16);
      * the realised harvest already carries the per-day panel degradation and
        attitude loss because ``make_extended_scenario`` folded them in.

    Keeping this as a separate driver leaves the original 24 h path byte-for-byte
    intact (requirement 20).
    """
    strat = strategy
    strat.reset(scen)
    n = scen.n

    soc = np.empty(n)
    soc_frac = np.empty(n)          # soc / instantaneous (faded) capacity
    served = {t: np.empty(n) for t in LOADS}
    enabled = {t: np.empty(n, dtype=bool) for t in LOADS}
    states = np.full(n, -1, dtype=int)
    margin = np.full(n, np.nan)
    reserve = np.full(n, np.nan)
    soc_hist = np.empty(n)
    pred_trace = np.full(n, np.nan) if getattr(strat, "uses_forecast", False) else None
    conf_trace = np.full(n, np.nan) if pred_trace is not None else None

    bo_latch = False
    dd_latch = False
    n_brownout = 0
    n_deep = 0
    n_crit_down = 0
    switches = {"critical": 0, "essential": 0, "deferrable": 0}
    prev_en = {"critical": True, "essential": True, "deferrable": True}
    dsum = np.zeros(n)
    payload_wh = 0.0
    crit_ok = 0
    ess_ok = 0
    htot = np.full(n, np.nan)

    s = float(scen.soc0_wh)
    for i in range(n):
        day = i // STEPS_PER_DAY
        # Instantaneous faded battery capacity (NASA PCoE), floored at 70 %.
        cap = BAT_CAPACITY_WH * battery_capacity(day)
        soc_hist[i] = s
        ctx = Context(i=i, soc_wh=s, cap_wh=cap, harvest_w=float(scen.harvest_w[i]),
                      shape_now=float(scen.shape[i]), soc_hist=soc_hist[: i + 1])
        dec = strat.step(ctx)

        load_w = sum(LOAD_W[t] for t in LOADS if dec.enable.get(t, True))
        res = bus_step(s, cap, load_w, ctx.harvest_w, dec.enable)

        s = res.soc_wh
        soc[i] = s
        soc_frac[i] = s / cap
        for t in LOADS:
            served[t][i] = res.served_wh[t]
            on = bool(dec.enable.get(t, True))
            enabled[t][i] = on
            if on != prev_en[t]:
                switches[t] += 1
                prev_en[t] = on
        if dec.state in FSM_STATE_ID:
            states[i] = FSM_STATE_ID[dec.state]
        margin[i] = dec.margin_wh
        reserve[i] = dec.reserve_wh
        if pred_trace is not None:
            pred_trace[i] = strat.fc.trace_pred[i]
            conf_trace[i] = strat.fc.trace_conf[i]
            htot[i] = strat.fc.trace_htot[i]

        f = s / cap
        crit_ok += int(res.critical_met)
        ess_ok += int(res.served_wh["essential"] >= LOAD_W["essential"] * DT_H - 1e-12)
        payload_wh += res.served_wh["deferrable"]
        if f < BROWNOUT_SOC and not bo_latch:
            bo_latch = True
            n_brownout += 1
        elif f >= BROWNOUT_SOC + 0.01:
            bo_latch = False
        if f < DEEP_DISCH_SOC and not dd_latch:
            dd_latch = True
            n_deep += 1
        elif f >= DEEP_DISCH_SOC + 0.01:
            dd_latch = False
        if not res.critical_met:
            n_crit_down += 1

        H = int(scen.horizon_steps[i])
        j = min(n, i + H)
        dsum[i] = float(scen.harvest_w[i:j].sum()) * DT_H

    res_obj = TrialResult(
        strategy=strat.name, trial=scen.trial, soc=soc, soc_frac=soc_frac, served=served,
        enabled=enabled, states=states, margin=margin, reserve=reserve,
        pred=pred_trace, conf=conf_trace,
    )
    res_obj.metrics = compute_metrics(
        res_obj, scen, n_brownout, n_deep, n_crit_down, switches, payload_wh, crit_ok, ess_ok
    )
    if pred_trace is not None:
        sunlit = (scen.shape > 0.05) & (np.arange(n) + scen.horizon_steps <= n) & np.isfinite(htot)
        res_obj.metrics["pred_mae_horizon_wh"] = float(np.abs(htot[sunlit] - dsum[sunlit]).mean())
        res_obj.metrics["pred_mae_w"] = float(strat.fc.mae_w())
    else:
        res_obj.metrics["pred_mae_horizon_wh"] = float("nan")
        res_obj.metrics["pred_mae_w"] = float("nan")
    return res_obj


# =============================================================================
# 9. METRICS
# =============================================================================

METRIC_COLS = [
    ("crit_uptime_pct", "critical uptime [%]"),
    ("ess_uptime_pct", "essential uptime [%]"),
    ("min_soc_pct", "min SoC [%]"),
    ("end_soc_pct", "end SoC [%]"),
    ("brownout_events", "brownout events"),
    ("deep_events", "deep-discharge events"),
    ("payload_wh", "payload energy [Wh]"),
    ("switches_total", "load switching events"),
    ("switches_defer", "payload switching events"),
    ("crit_unmet_min", "critical outage [min]"),
    ("pred_mae_w", "predictor MAE [W]"),
    ("pred_mae_horizon_wh", "predictor MAE horizon [Wh]"),
]


def compute_metrics(res: TrialResult, scen: Scenario, n_brownout: int, n_deep: int,
                    n_crit_down: int, switches: Dict[str, int], payload_wh: float,
                    crit_ok: int, ess_ok: int) -> Dict[str, float]:
    """Per-trial metric dictionary (requirement 9)."""
    n = res.n
    return {
        "crit_uptime_pct": 100.0 * crit_ok / n,
        "ess_uptime_pct": 100.0 * ess_ok / n,
        "min_soc_pct": 100.0 * float(res.soc_frac.min()),
        "end_soc_pct": 100.0 * float(res.soc_frac[-1]),
        "brownout_events": float(n_brownout),
        "deep_events": float(n_deep),
        "payload_wh": float(payload_wh),
        "switches_total": float(sum(switches.values())),
        "switches_defer": float(switches["deferrable"]),
        "crit_unmet_min": float(n_crit_down),
        "pred_mae_w": float("nan"),
        "pred_mae_horizon_wh": float("nan"),
    }


def aggregate(results: List[TrialResult]) -> Dict[str, Dict[str, float]]:
    """Mean/std over the Monte-Carlo trials, per metric, per strategy."""
    agg: Dict[str, Dict[str, float]] = {}
    for key in STRATEGY_KEYS:
        rows = [r.metrics for r in results if r.strategy == key]
        if not rows:
            continue
        d: Dict[str, float] = {}
        for col, _ in METRIC_COLS:
            vals = np.array([r[col] for r in rows], dtype=float)
            # a column is undefined for some strategies (e.g. predictor MAE for the
            # non-predictive rungs) -> keep it NaN instead of emitting an
            # all-NaN-slice RuntimeWarning from nanmean/nanmin/nanmax
            if np.all(np.isnan(vals)):
                d[col + "_mean"] = float("nan")
                d[col + "_std"] = float("nan")
                d[col + "_min"] = float("nan")
                d[col + "_max"] = float("nan")
                continue
            d[col + "_mean"] = float(np.nanmean(vals))
            d[col + "_std"] = float(np.nanstd(vals))
            d[col + "_min"] = float(np.nanmin(vals))
            d[col + "_max"] = float(np.nanmax(vals))
        # head-to-head: fraction of trials in which the best reactive was beaten
        agg[key] = d
    return agg


def head_to_head(results: List[TrialResult]) -> Dict[str, Dict[str, float]]:
    """
    Pair-wise win rate of each strategy against the *best* reactive strategy
    (max mean critical uptime and mean payload among the reactive rungs), on the
    same trial -- the direct test of the study hypothesis.
    """
    by_trial: Dict[str, Dict[int, TrialResult]] = {}
    for r in results:
        by_trial.setdefault(r.strategy, {})[r.trial] = r
    react = [k for k in ("REACTIVE", "REACTIVE_H")]
    best_react = max(react, key=lambda k: np.nanmean([r.metrics["crit_uptime_pct"] for r in by_trial[k].values()]))
    out: Dict[str, Dict[str, float]] = {"best_reactive": best_react}
    for key in STRATEGY_KEYS:
        wu, wp, both = [], [], []
        for t in by_trial[key]:
            a = by_trial[key][t].metrics
            b = by_trial[best_react][t].metrics
            u = a["crit_uptime_pct"] > b["crit_uptime_pct"] + 1e-9
            p = a["payload_wh"] > b["payload_wh"] + 1e-9
            wu.append(float(u))
            wp.append(float(p))
            both.append(float(u and p))
        out[key] = {
            "win_uptime_pct": 100.0 * float(np.mean(wu)) if wu else float("nan"),
            "win_payload_pct": 100.0 * float(np.mean(wp)) if wp else float("nan"),
            "win_both_pct": 100.0 * float(np.mean(both)) if both else float("nan"),
        }
    return out


# =============================================================================
# 10. MONTE-CARLO DRIVER
# =============================================================================

def run_strategy(scenarios: Sequence[Scenario], key: str) -> List[TrialResult]:
    """Run one strategy over every scenario (all strategies share the scenarios)."""
    out = []
    for scen in scenarios:
        out.append(simulate_trial(scen, make_strategy(key)))
    return out


def run_monte_carlo(n_trials: int = N_TRIALS, verbose: bool = True) -> Tuple[Dict[str, List[TrialResult]], List[Scenario]]:
    """Generate the scenarios once, then run every strategy on all of them."""
    scenarios = [make_scenario(t) for t in range(n_trials)]
    all_res: Dict[str, List[TrialResult]] = {}
    for key in STRATEGY_KEYS:
        all_res[key] = run_strategy(scenarios, key)
        if verbose:
            up = np.nanmean([r.metrics["crit_uptime_pct"] for r in all_res[key]])
            pl = np.nanmean([r.metrics["payload_wh"] for r in all_res[key]])
            print(f"  ran {key:<11s} n={n_trials:3d}  uptime={up:6.2f}%  payload={pl:7.1f} Wh")
    return all_res, scenarios


# =============================================================================
# 11. REPORTING
# =============================================================================

def print_summary(agg: Dict[str, Dict[str, float]], h2h: Dict[str, Dict[str, float]],
                  n_trials: int, title: Optional[str] = None) -> None:
    """Human-readable summary table (mean +/- std over the Monte-Carlo trials).

    ``title`` lets the 180-day extension reuse the exact same column layout
    with its own header; the default keeps the original 24 h wording.
    """
    cols = METRIC_COLS
    w0 = 12
    w1 = 21
    print("\n" + "=" * (w0 + w1 * len(STRATEGY_KEYS)))
    print(title if title is not None else
          "SUMMARY -- mean +/- std over {} Monte-Carlo trials (identical scenarios per trial)".format(n_trials))
    print("=" * (w0 + w1 * len(STRATEGY_KEYS)))
    head = f"{'metric':<20s}" + "".join(f"{STRATEGY_LABELS[k]:>21s}" for k in STRATEGY_KEYS)
    print(head)
    print("-" * len(head))
    for col, label in cols:
        line = f"{label:<20s}"
        for k in STRATEGY_KEYS:
            m = agg[k].get(col + "_mean", float("nan"))
            s = agg[k].get(col + "_std", float("nan"))
            if col in ("brownout_events", "deep_events", "switches_total", "switches_defer", "crit_unmet_min"):
                line += f"{m:9.2f}+/-{s:<10.2f}"
            elif np.isnan(m):
                line += f"{'-':>21s}"
            else:
                line += f"{m:11.2f}+/-{s:<8.2f}"
        print(line)
    print("-" * len(head))
    print("head-to-head vs best reactive ('{}'):".format(h2h["best_reactive"]))
    for k in STRATEGY_KEYS:
        h = h2h[k]
        print(f"  {STRATEGY_LABELS[k]:<16s} wins uptime {h['win_uptime_pct']:5.1f}% | "
              f"payload {h['win_payload_pct']:5.1f}% | both {h['win_both_pct']:5.1f}%")


def write_csv(agg: Dict[str, Dict[str, float]], results: Dict[str, List[TrialResult]],
              outdir: str = RESULTS_DIR, prefix: str = "") -> None:
    """Write the summary table (metrics.csv) and the raw per-trial table.

    ``prefix`` lets the 180-day extension write ext_metrics*.csv alongside the
    24 h files instead of overwriting them (default "" keeps the old names).
    """
    os.makedirs(outdir, exist_ok=True)

    # -- summary ---------------------------------------------------------
    path = os.path.join(outdir, prefix + "metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "label", "metric", "mean", "std", "min", "max"])
        for k in STRATEGY_KEYS:
            for col, _ in METRIC_COLS:
                w.writerow([k, STRATEGY_LABELS[k], col,
                            f"{agg[k][col + '_mean']:.6g}", f"{agg[k][col + '_std']:.6g}",
                            f"{agg[k][col + '_min']:.6g}", f"{agg[k][col + '_max']:.6g}"])

    # -- per-trial -------------------------------------------------------
    path2 = os.path.join(outdir, prefix + "metrics_trials.csv")
    with open(path2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "trial", "seed"] + [c for c, _ in METRIC_COLS])
        for k in STRATEGY_KEYS:
            for r in results[k]:
                w.writerow([k, r.trial, BASE_SEED + 1009 * r.trial] +
                           [f"{r.metrics[c]:.6g}" for c, _ in METRIC_COLS])
    print(f"\nwrote {path} and {path2}")


# =============================================================================
# 12. PLOTS
# =============================================================================

def _style(ax) -> None:
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_xlim(0, 24)


def plot_soc(all_res: Dict[str, List[TrialResult]], scen: Scenario, trial: int, outdir: str) -> str:
    """Plot 1: SoC over 24 h for all strategies, one representative trial."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c, k in enumerate(STRATEGY_KEYS):
        r = next(x for x in all_res[k] if x.trial == trial)
        ax.plot(HOURS_AXIS, 100.0 * r.soc_frac, lw=1.4, color=colors[c], label=STRATEGY_LABELS[k])
    cap = scen.capacity_wh
    for frac, lbl, ls in ((BROWNOUT_SOC, "brownout 5%", ":"), (DEEP_DISCH_SOC, "deep disch 10%", "--"),
                          (RESERVE_FRAC, "reserve 20%", "-.")):
        ax.axhline(100 * frac, color="k", lw=0.7, ls=ls, alpha=0.6)
        ax.text(0.15, 100 * frac + 0.7, lbl, fontsize=7, color="k", alpha=0.7)
    # eclipse window
    e0, e1 = scen.eclipse_start_min / 60.0, (scen.eclipse_start_min + scen.eclipse_len_min) / 60.0
    ax.axvspan(e0, e1, color="0.15", alpha=0.18)
    ax.text((e0 + e1) / 2, 104, "eclipse", ha="center", fontsize=8, color="0.2")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("battery SoC [%]")
    ax.set_title(f"Plot 1 - State of charge over 24 h, all strategies (trial {trial}, "
                 f"cap={cap:.0f} Wh, SoC0={100*scen.soc0_wh/cap:.0f}%, fade={100*scen.fade_frac:.1f}%)")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    _style(ax)
    fig.tight_layout()
    p = os.path.join(outdir, f"plot1_soc_trial{trial}.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_loads(r: TrialResult, scen: Scenario, outdir: str) -> str:
    """Plot 2: commanded load states over 24 h for the predictive strategy."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ymap = {"deferrable": 2, "essential": 1, "critical": 0}
    for tier, y in ymap.items():
        ax.step(HOURS_AXIS, y + 0.85 * r.enabled[tier].astype(float), where="post",
                lw=1.3, label=f"{tier} ({LOAD_W[tier]:.0f} W)")
    e0, e1 = scen.eclipse_start_min / 60.0, (scen.eclipse_start_min + scen.eclipse_len_min) / 60.0
    ax.axvspan(e0, e1, color="0.15", alpha=0.15)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["critical", "essential", "deferrable"])
    ax.set_ylim(-0.3, 2.9)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("load tier")
    ax.set_title(f"Plot 2 - commanded load states, {STRATEGY_LABELS[r.strategy]} (trial {r.trial})")
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    _style(ax)
    fig.tight_layout()
    p = os.path.join(outdir, f"plot2_loads_{r.strategy}_trial{r.trial}.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_fsm(r: TrialResult, scen: Scenario, outdir: str) -> str:
    """Plot 3: FSM state timeline for the predictive strategy."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.step(HOURS_AXIS, r.states, where="post", lw=1.2, color="tab:blue")
    # colour the background per state for readability
    s = r.states.astype(int)
    edges = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [s.size]])
    # plt.cm.get_cmap() was removed in matplotlib 3.9; use the top-level accessor
    cmap = plt.get_cmap("tab10")
    for st, en in zip(starts, ends):
        if s[st] < 0:
            continue
        ax.axvspan(st * DT_MIN / 60.0, en * DT_MIN / 60.0, color=cmap(s[st] % 10), alpha=0.12)
    e0, e1 = scen.eclipse_start_min / 60.0, (scen.eclipse_start_min + scen.eclipse_len_min) / 60.0
    ax.axvspan(e0, e1, color="0.15", alpha=0.25)
    ax.set_yticks(range(len(FSM_STATES)))
    ax.set_yticklabels(FSM_STATES)
    ax.set_ylim(-0.5, len(FSM_STATES) - 0.5)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("FSM state")
    ax.set_title(f"Plot 3 - 5-state energy manager FSM, {STRATEGY_LABELS[r.strategy]} (trial {r.trial})")
    _style(ax)
    fig.tight_layout()
    p = os.path.join(outdir, f"plot3_fsm_{r.strategy}_trial{r.trial}.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def _bar(ax, labels: Sequence[str], means: Sequence[float], stds: Sequence[float],
         ylabel: str, title: str, color: str) -> None:
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=color, alpha=0.85, edgecolor="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, lw=0.6)
    for xi, m in zip(x, means):
        ax.text(xi, m, f"{m:.1f}", ha="center", va="bottom", fontsize=7)


def plot_uptime(agg: Dict[str, Dict[str, float]], n_trials: int, outdir: str) -> str:
    """Plot 4: critical-load uptime per strategy (MC mean +/- std)."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels = [STRATEGY_LABELS[k] for k in STRATEGY_KEYS]
    m = [agg[k]["crit_uptime_pct_mean"] for k in STRATEGY_KEYS]
    s = [agg[k]["crit_uptime_pct_std"] for k in STRATEGY_KEYS]
    _bar(ax, labels, m, s, "critical load uptime [%]", f"Plot 4 - critical uptime, {n_trials} trials (mean +/- std)", "tab:blue")
    lo = max(0.0, min(m) - 4.0)
    ax.set_ylim(lo, 100.6)
    fig.tight_layout()
    p = os.path.join(outdir, "plot4_critical_uptime.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_payload(agg: Dict[str, Dict[str, float]], n_trials: int, outdir: str) -> str:
    """Plot 5: payload (Deferrable) energy delivered per strategy."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels = [STRATEGY_LABELS[k] for k in STRATEGY_KEYS]
    m = [agg[k]["payload_wh_mean"] for k in STRATEGY_KEYS]
    s = [agg[k]["payload_wh_std"] for k in STRATEGY_KEYS]
    _bar(ax, labels, m, s, "payload energy delivered [Wh]", f"Plot 5 - payload energy, {n_trials} trials (mean +/- std)", "tab:orange")
    fig.tight_layout()
    p = os.path.join(outdir, "plot5_payload_energy.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_tradeoff(agg: Dict[str, Dict[str, float]], outdir: str) -> str:
    """Plot 6: critical uptime (x) vs payload energy (y), one point per strategy."""
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    xs, ys = [], []
    for c, k in enumerate(STRATEGY_KEYS):
        x = agg[k]["crit_uptime_pct_mean"]
        y = agg[k]["payload_wh_mean"]
        xe = agg[k]["crit_uptime_pct_std"]
        ye = agg[k]["payload_wh_std"]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=7, capsize=3, color=colors[c],
                    label=STRATEGY_LABELS[k], lw=1.2)
        ax.annotate(STRATEGY_LABELS[k], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
        xs.append(x)
        ys.append(y)
    # connect the rungs in ladder order to expose the monotone trend
    order = np.argsort(xs)
    ax.plot([xs[i] for i in order], [ys[i] for i in order], ls="--", lw=0.9, color="0.5",
            label="ladder (sorted by uptime)")
    ax.set_xlabel("critical load uptime [%]  (higher is better)")
    ax.set_ylabel("payload energy delivered [Wh]  (higher is better)")
    ax.set_title("Plot 6 - availability vs payload trade-off (mean over trials)")
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    p = os.path.join(outdir, "plot6_tradeoff.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_predictor(r4: TrialResult, r5: TrialResult, scen: Scenario, outdir: str) -> str:
    """Plot 7: predictor error over time, predicted vs actual array power."""
    lead = 30                                   # [min] lead shown
    actual = np.full(N_STEPS, np.nan)
    src = scen.harvest_w
    actual[: N_STEPS - lead] = src[lead:]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(HOURS_AXIS, src, color="0.3", lw=1.6, label="actual harvest", zorder=1)
    ax.plot(HOURS_AXIS, r4.pred, lw=1.1, label=f"pred EWMA (lead {lead} min)", zorder=2)
    ax.plot(HOURS_AXIS, r5.pred, lw=1.1, label=f"pred EWMA + residual (lead {lead} min)", zorder=3)
    ax.set_ylabel("array power [W]")
    ax.set_title(f"Plot 7 - harvest forecast vs actual, trial {r4.trial} "
                 f"(MAE EWMA={r4.metrics['pred_mae_w']:.1f} W, "
                 f"EWMA+residual={r5.metrics['pred_mae_w']:.1f} W)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_xlim(0, 24)

    ax = axes[1]
    e4 = r4.pred - actual
    e5 = r5.pred - actual
    ax.plot(HOURS_AXIS, e4, lw=0.9, label="error EWMA")
    ax.plot(HOURS_AXIS, e5, lw=0.9, label="error EWMA + residual")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("prediction error [W]")
    ax.set_xlabel("time [h]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, lw=0.6)
    fig.tight_layout()
    p = os.path.join(outdir, f"plot7_predictor_trial{r4.trial}.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def pick_representative(all_res: Dict[str, List[TrialResult]], scenarios: Sequence[Scenario]) -> int:
    """
    Choose the trial used for the time-domain plots: the one whose unmanaged
    minimum SoC is closest to the median (a 'typical' day, not a lucky one).
    """
    mins = np.array([min(r.metrics["min_soc_pct"] for r in all_res["NO_MGMT"] if r.trial == s.trial)
                     for s in scenarios])
    med = np.median(mins)
    return int(scenarios[int(np.argmin(np.abs(mins - med)))].trial)


def plot_results(all_res: Dict[str, List[TrialResult]], agg: Dict[str, Dict[str, float]],
                 scenarios: Sequence[Scenario], n_trials: int, outdir: str = RESULTS_DIR) -> List[str]:
    """Generate all seven required figures."""
    os.makedirs(outdir, exist_ok=True)
    trial = pick_representative(all_res, scenarios)
    scen = next(s for s in scenarios if s.trial == trial)
    r4 = next(x for x in all_res["PREDICT_4"] if x.trial == trial)
    r5 = next(x for x in all_res["PREDICT_5"] if x.trial == trial)
    paths = [
        plot_soc(all_res, scen, trial, outdir),
        plot_loads(r5, scen, outdir),
        plot_fsm(r5, scen, outdir),
        plot_uptime(agg, n_trials, outdir),
        plot_payload(agg, n_trials, outdir),
        plot_tradeoff(agg, outdir),
        plot_predictor(r4, r5, scen, outdir),
    ]
    for p in paths:
        print(f"wrote {p}")
    return paths


# =============================================================================
# 13. MAIN
# =============================================================================

def _ext_representative(all_res: Dict[str, List[TrialResult]],
                        scenarios: Sequence[Scenario]) -> int:
    """
    Representative extended trial for the time-domain plot: the one whose
    unmanaged minimum SoC is closest to the median (same idea as
    ``pick_representative`` but tolerant of a small extended trial count).
    """
    mins = np.array([r.metrics["min_soc_pct"] for r in all_res["NO_MGMT"]], dtype=float)
    if mins.size == 0:
        return scenarios[0].trial
    med = float(np.median(mins))
    idx = int(np.argmin(np.abs(mins - med)))
    return int(all_res["NO_MGMT"][idx].trial)


def plot_ext_soc(all_res: Dict[str, List[TrialResult]], scen: Scenario, trial: int,
                 outdir: str) -> str:
    """Extended plot 1: battery SoC over the 180-day mission, all strategies."""
    fig, ax = plt.subplots(figsize=(12, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    days_axis = np.arange(scen.n) * DT_MIN / STEPS_PER_DAY     # [d]
    stride = 5                                                 # 5-min downsample
    for c, k in enumerate(STRATEGY_KEYS):
        r = next(x for x in all_res[k] if x.trial == trial)
        ax.plot(days_axis[::stride], 100.0 * r.soc_frac[::stride], lw=0.6,
                color=colors[c], alpha=0.85, label=STRATEGY_LABELS[k])
    for frac, lbl, ls in ((BROWNOUT_SOC, "brownout 5 %", ":"),
                          (DEEP_DISCH_SOC, "deep disch 10 %", "--"),
                          (RESERVE_FRAC, "reserve 20 %", "-.")):
        ax.axhline(100 * frac, color="k", lw=0.7, ls=ls, alpha=0.6)
        ax.text(0.5, 100 * frac + 0.8, lbl, fontsize=7, color="k", alpha=0.7)
    ax.set_xlabel("mission time [d]")
    ax.set_ylabel("battery SoC [%]")
    ax.set_xlim(0, SIM_DAYS)
    ax.set_ylim(0, 100)
    ax.set_title(f"Extended plot 1 - SoC over {SIM_DAYS} days "
                 f"(trial {trial}, sustainable LEO + degradation)")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.grid(True, alpha=0.25, lw=0.6)
    fig.tight_layout()
    p = os.path.join(outdir, "plot_ext_soc.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_ext_harvest(scen: Scenario, outdir: str) -> str:
    """Extended plot 2: daily harvest before vs after degradation + loss curve."""
    ideal = scen.harvest_ideal
    real = scen.harvest_w
    if ideal is None or real is None:
        return ""
    # daily mean array power [W] (reshape 180 x 1440 and average over the day)
    ideal_d = ideal.reshape(SIM_DAYS, STEPS_PER_DAY).mean(axis=1)
    real_d = real.reshape(SIM_DAYS, STEPS_PER_DAY).mean(axis=1)
    # daily mean combined degradation * (1 - attitude) factor
    factor = (scen.degradation * (1.0 - scen.attitude)).reshape(SIM_DAYS, STEPS_PER_DAY).mean(axis=1)
    days = np.arange(SIM_DAYS)

    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.plot(days, ideal_d, lw=1.4, color="tab:orange", label="harvest before degradation")
    ax.plot(days, real_d, lw=1.4, color="tab:blue", label="harvest after degradation")
    ax.set_xlabel("mission time [d]")
    ax.set_ylabel("daily mean array power [W]")
    ax.set_xlim(0, SIM_DAYS)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(days, 100.0 * factor, lw=1.0, ls="--", color="0.35",
             label="array capability (degr. x attitude)")
    ax2.set_ylabel("array capability [% of nameplate]")
    ax2.set_ylim(96.0, 100.5)
    ax2.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Extended plot 2 - array harvest over {SIM_DAYS} days "
                 f"(2.8 %/yr panel + MinXSS attitude loss)")
    fig.tight_layout()
    p = os.path.join(outdir, "plot_ext_harvest.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_ext_eclipse(scen: Scenario, outdir: str) -> str:
    """Extended plot 3: eclipse fraction and beta angle vs day (NASA TFAWS 2014)."""
    beta = scen.beta_deg
    ecl = scen.eclipse_frac
    if beta is None or ecl is None:
        return ""
    days = np.arange(beta.size)
    ecl_min = ecl * ORBIT_PERIOD_MIN

    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.plot(days, 100.0 * ecl, lw=1.5, color="tab:purple",
            label="eclipse fraction of orbit")
    ax.set_xlabel("mission time [d]")
    ax.set_ylabel("eclipse fraction [% of orbit]")
    ax.set_xlim(0, SIM_DAYS)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(days, beta, lw=1.2, ls="--", color="tab:red", label="beta angle")
    ax2.set_ylabel("beta angle [deg]")
    ax2.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Extended plot 3 - eclipse fraction vs beta angle, {SIM_DAYS} days "
                 f"(alt={ALTITUDE_KM:.0f} km, T={ORBIT_PERIOD_MIN:.1f} min, "
                 f"{np.mean(ecl_min):.1f} min mean eclipse)")
    fig.tight_layout()
    p = os.path.join(outdir, "plot_ext_eclipse.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_ext_metrics(agg: Dict[str, Dict[str, float]], n_trials: int, outdir: str) -> str:
    """Extended plot 4: bar charts of the key metrics aggregated over 180 days."""
    labels = [STRATEGY_LABELS[k] for k in STRATEGY_KEYS]
    panels = [
        ("crit_uptime_pct", "critical uptime [%]", "tab:blue"),
        ("ess_uptime_pct", "essential uptime [%]", "tab:green"),
        ("min_soc_pct", "min SoC [%]", "tab:red"),
        ("payload_wh", "payload energy [Wh]", "tab:orange"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.0))
    for ax, (col, ylabel, color) in zip(axes.ravel(), panels):
        m = [agg[k][col + "_mean"] for k in STRATEGY_KEYS]
        s = [agg[k][col + "_std"] for k in STRATEGY_KEYS]
        _bar(ax, labels, m, s, ylabel,
             f"{ylabel} over {SIM_DAYS} d ({n_trials} trials)", color)
    fig.suptitle(f"Extended plot 4 - EPS metrics over {SIM_DAYS} days "
                 f"(mean +/- std, {n_trials} trials)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(outdir, "plot_ext_metrics.png")
    fig.savefig(p, dpi=PLOT_DPI)
    plt.close(fig)
    return p


def plot_ext_results(all_res: Dict[str, List[TrialResult]], agg: Dict[str, Dict[str, float]],
                     scenarios: Sequence[Scenario], n_trials: int,
                     outdir: str = RESULTS_DIR) -> List[str]:
    """Generate the four required 180-day extension figures."""
    os.makedirs(outdir, exist_ok=True)
    trial = _ext_representative(all_res, scenarios)
    scen = next(s for s in scenarios if s.trial == trial)
    paths = [
        plot_ext_soc(all_res, scen, trial, outdir),
        plot_ext_harvest(scen, outdir),
        plot_ext_eclipse(scen, outdir),
        plot_ext_metrics(agg, n_trials, outdir),
    ]
    for p in paths:
        if p:
            print(f"wrote {p}")
    return paths


def run_strategy_extended(scenarios: Sequence[Scenario], key: str) -> List[TrialResult]:
    """Run one strategy over every extended scenario (shared scenarios)."""
    out = []
    for scen in scenarios:
        out.append(simulate_extended_trial(scen, make_strategy(key)))
    return out


def run_extended(n_trials: int = EXT_N_TRIALS, verbose: bool = True) -> None:
    """
    180-day extension driver (requirements 11-19).

    Builds the extended scenarios once, runs every strategy on the *same*
    scenarios, then reuses the 24 h reporting/plotting machinery where it is
    strategy-agnostic (aggregate / head_to_head / print_summary / write_csv) and
    adds the four extended figures.  The 24 h plots already on disk are left
    untouched (requirement 19).
    """
    print("=" * 78)
    print("Study 3 - EXTENDED EPS energy budgeting (180-day realistic LEO)")
    print(f"  {N_STEPS_TOTAL} steps @ {DT_MIN:g} min = {SIM_DAYS} d, "
          f"altitude {ALTITUDE_KM:.0f} km, orbit {ORBIT_PERIOD_MIN:.1f} min")
    print(f"  beta angle {BETA_ANGLE_MIN:.0f}..{BETA_ANGLE_MAX:.2f} deg "
          f"(period {BETA_PERIOD_DAYS:.0f} d, NASA TFAWS 2014)")
    print(f"  panel degradation {100*DEGRADATION_PER_YEAR:.1f} %/yr (floor 15 %), "
          f"attitude cosine loss {POINTING_LOSS_FRACTION:.2e} "
          f"(<= {100.0*np.clip(1.5*POINTING_LOSS_FRACTION, 0.0, 0.01):.2e} % cap)")
    print(f"  battery cycle fade {100*BATTERY_CYCLE_DEGRADATION:.2f} %/cycle "
          f"(floor 30 %, {CYCLES_PER_DAY:.1f} cycles/d)")
    print(f"  Monte Carlo: {n_trials} extended trials, master seed {BASE_SEED}")
    print("=" * 78)

    scenarios = [make_extended_scenario(t) for t in range(n_trials)]
    all_res: Dict[str, List[TrialResult]] = {}
    for key in STRATEGY_KEYS:
        all_res[key] = run_strategy_extended(scenarios, key)
        if verbose:
            up = np.nanmean([r.metrics["crit_uptime_pct"] for r in all_res[key]])
            pl = np.nanmean([r.metrics["payload_wh"] for r in all_res[key]])
            print(f"  ran {key:<11s} n={n_trials:3d}  uptime={up:6.2f}%  payload={pl:9.1f} Wh")

    flat = [r for k in STRATEGY_KEYS for r in all_res[k]]
    agg = aggregate(flat)
    h2h = head_to_head(flat)

    print_summary(agg, h2h, n_trials,
                  title=f"EXTENDED SUMMARY -- {SIM_DAYS}-day mean +/- std over "
                        f"{n_trials} trials (identical scenarios per trial)")
    write_csv(agg, all_res, RESULTS_DIR, prefix="ext_")
    plot_ext_results(all_res, agg, scenarios, n_trials, RESULTS_DIR)

    best_react = h2h["best_reactive"]
    print("\n" + "-" * 78)
    print(f"EXTENDED CONCLUSION ({SIM_DAYS} days)")
    print(f"  best reactive rung        : {best_react}")
    print(f"  PREDICT_5 wins uptime in  : {h2h['PREDICT_5']['win_uptime_pct']:.0f}% of trials")
    print(f"  PREDICT_5 wins payload in : {h2h['PREDICT_5']['win_payload_pct']:.0f}% of trials")
    print(f"  PREDICT_5 wins both in    : {h2h['PREDICT_5']['win_both_pct']:.0f}% of trials")
    print("-" * 78 + "\n")


def print_step60_diagnostic(scen: Scenario) -> None:
    """
    Single-step transparency dump: run PREDICT_5 for 61 minutes of trial 0 and
    print the margin bookkeeping at step 60.  Makes the load-forecast and
    horizon logic auditable from the console.
    """
    strat = make_strategy("PREDICT_5")
    strat.reset(scen)
    n, cap = scen.n, scen.capacity_wh
    soc_hist = np.empty(n)
    s = float(scen.soc0_wh)
    for i in range(61):
        soc_hist[i] = s
        ctx = Context(i=i, soc_wh=s, cap_wh=cap, harvest_w=float(scen.harvest_w[i]),
                      shape_now=float(scen.shape[i]), soc_hist=soc_hist[: i + 1])
        dec = strat.step(ctx)
        s = bus_step(s, cap, sum(LOAD_W[t] for t in LOADS if dec.enable.get(t, True)),
                     ctx.harvest_w, dec.enable).soc_wh
    d = strat.dbg_f_load - W_DEFERRABLE * strat.dbg_lit_min * DT_H
    print("\n" + "-" * 78)
    print("STEP-60 DIAGNOSTIC (trial 0, PREDICT_5)")
    print(f"  eclipse window          : [{scen.eclipse_start_min / 60:.2f} h, "
          f"{(scen.eclipse_start_min + scen.eclipse_len_min) / 60:.2f} h]")
    print(f"  horizon_min             : {strat.dbg_horizon}")
    print(f"  lit_min_in_horizon      : {strat.dbg_lit_min}")
    print(f"  payload booked          : {W_DEFERRABLE * strat.dbg_lit_min * DT_H:8.1f} Wh")
    print(f"  base (C+E) booked       : {d:8.1f} Wh")
    print(f"  f_load_wh               : {strat.dbg_f_load:8.1f} Wh")
    print(f"  f_harvest_wh            : {strat.dbg_f_harv:8.1f} Wh")
    print(f"  reserve                 : {strat.h_reserve:8.1f} Wh")
    print(f"  margin                  : {strat.h_margin:+8.1f} Wh")
    print(f"  SoC                     : {s:8.1f} Wh ({100.0 * s / cap:.2f} %)")
    print(f"  FSM state               : {strat.state}")
    print("-" * 78 + "\n")


def run_24h() -> None:
    """Original 24 h Monte-Carlo study (unchanged behaviour, plots 1-7)."""
    print("=" * 78)
    print("Study 3 - EPS energy budgeting: reactive vs predictive vs oracle")
    print(f"  {N_STEPS} steps @ {DT_MIN:g} min = {N_STEPS * DT_MIN / 60:.0f} h, "
          f"panel {PANEL_PEAK_W:g} W peak, battery {BAT_CAPACITY_WH:g} Wh")
    print(f"  loads: critical {W_CRITICAL:g} W / essential {W_ESSENTIAL:g} W / "
          f"deferrable {W_DEFERRABLE:g} W")
    print(f"  orbit: {ORBIT_PERIOD_MIN_24H / 60:.0f} h period, {DAYLIGHT_MIN / 60:.0f} h daylight, "
          f"{ECLIPSE_MIN_24H / 60:.0f} h eclipse (offset randomised per trial)")
    print(f"  Monte Carlo: {N_TRIALS} trials, master seed {BASE_SEED}")
    print("=" * 78)
    print("Oracle omitted — see README for rationale.")
    print("Reactive thresholds tuned to match predictive 20% reserve for fair comparison.")

    print_step60_diagnostic(make_scenario(0))

    all_res, scenarios = run_monte_carlo(N_TRIALS, verbose=True)

    # flatten for aggregation / head-to-head
    flat = [r for k in STRATEGY_KEYS for r in all_res[k]]
    agg = aggregate(flat)
    h2h = head_to_head(flat)

    print_summary(agg, h2h, N_TRIALS)
    write_csv(agg, all_res, RESULTS_DIR)
    plot_results(all_res, agg, scenarios, N_TRIALS, RESULTS_DIR)

    # ---- study conclusion --------------------------------------------------
    best_react = h2h["best_reactive"]
    p5 = h2h["PREDICT_5"]
    print("\n" + "-" * 78)
    print("CONCLUSION")
    print(f"  best reactive rung        : {best_react}")
    print(f"  PREDICT_5 wins uptime in  : {p5['win_uptime_pct']:.0f}% of trials")
    print(f"  PREDICT_5 wins payload in : {p5['win_payload_pct']:.0f}% of trials")
    print(f"  PREDICT_5 wins both in    : {p5['win_both_pct']:.0f}% of trials")
    print("-" * 78 + "\n")


def main() -> None:
    """
    Study driver selected by the top-level ``USE_EXTENDED`` flag:
    ``False`` -> the original 24 h Monte-Carlo study; ``True`` -> the 180-day
    realistic-orbit extension.  The two paths are independent, so flipping the
    flag never changes or deletes the other study's outputs.
    """
    if USE_EXTENDED:
        run_extended()
    else:
        run_24h()


if __name__ == "__main__":
    main()
