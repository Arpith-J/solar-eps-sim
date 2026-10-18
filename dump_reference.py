"""
dump_reference.py
=================

Generate ``reference_trace.csv`` for the ESP32 hardware-in-the-loop
comparison (see ``compare_esp32.py`` and
``esp32_validation/esp32_validation.ino``).

One randomised 24 h trial (1440 one-minute steps) is built with
``study3_energy_budgeting.make_scenario`` -- the same Monte-Carlo scenario
generator the main study uses -- so the harvest trace contains realistic
solar geometry, the embedded eclipse and stochastic cloud gaps.  The trace
is then replayed through the *simplified* SoC integrator + 5-state FSM that
the ESP32 firmware implements.

Important: the battery model here is deliberately NOT ``study3.bus_step``
(no charge/discharge efficiency, no spill, no reverse-priority shedding)
and the controller is deliberately NOT ``study3.PredictiveFSM``.  The
firmware harness runs the simplified model (fixed 500 Wh pack, eta = 1,
absolute-Wh thresholds), so the Python reference must mirror that model
exactly -- the point of the HIL comparison is plumbing / numeric parity,
not controller fidelity.  Only the *harvest trace* comes from study3.

Usage:
    python dump_reference.py [--trial 0] [--out reference_trace.csv]

With the nominal 155 W array and study3's fixed load tiers the battery can
stay NOMINAL for a whole day; pass e.g. ``--load-scale 1.6`` or pick a very
cloudy trial to drive the SoC through the DEGRADED / CRITICAL thresholds.

Output columns:
    timestamp,harvest_w,load_c,load_e,load_d,soc_py,fsm_py,payload_py

ORDERING ASSUMPTION (must match the firmware exactly):
    The load served during minute i is decided from the FSM state computed
    at the end of minute i-1 (i.e. from the previous SoC).  After the SoC is
    integrated the FSM is re-evaluated against the NEW SoC, and that new
    state -- plus its payload flag -- is what gets logged on row i; it will
    govern the load of row i+1.  ``processLine()`` in the .ino does exactly
    this: it reads ``deferrableOn(g_state)`` before updating SoC, then calls
    ``evaluateFsm()`` after.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import study3_energy_budgeting as s3

# ---------------------------------------------------------------------------
# Firmware-parity constants -- KEEP IN SYNC with esp32_validation.ino.
# These intentionally differ from study3's configuration (faded capacity,
# eta_chg/eta_dis, margin-based FSM thresholds); see module docstring.
# ---------------------------------------------------------------------------
BATTERY_CAPACITY_WH = 500.0       # firmware clips SoC to this fixed pack size
SOC0_WH = 350.0                   # firmware's fixed initial SoC
SOC_CRITICAL_WH = 50.0            # [Wh] below this -> CRITICAL
SOC_DEGRADED_WH = 100.0           # [Wh] below this -> DEGRADED
SOC_SAFE_MODE_WH = 25.0           # [Wh] deep-discharge floor -> SAFE_MODE
SOC_RECOVERY_EXIT_WH = 100.0      # [Wh] RECOVERY -> NOMINAL threshold
DT_HOURS = 1.0 / 60.0             # one simulation step = 1 minute
N_STEPS_DEFAULT = 1440            # 24 h at 1-minute resolution

HEADER = [
    "timestamp", "harvest_w", "load_c", "load_e", "load_d",
    "soc_py", "fsm_py", "payload_py",
]


def evaluate_fsm(soc_wh: float, prev_state: str) -> str:
    """
    Mirror of ``evaluateFsm()`` in the firmware -- branch order matters and
    is reproduced verbatim:

      1. SoC <  25                -> SAFE_MODE  (deep-discharge guard wins)
      2. prev state == SAFE_MODE  -> RECOVERY   (staged exit, one step)
      3. SoC <  50                -> CRITICAL
      4. SoC < 100                -> DEGRADED
      5. prev state == RECOVERY and SoC < 100 -> RECOVERY
      6. otherwise                -> NOMINAL

    NOTE: branch 5 is currently unreachable in the firmware because any
    SoC < 100 is caught by branch 4 first (SOC_RECOVERY_EXIT_WH ==
    SOC_DEGRADED_WH == 100), so RECOVERY lasts exactly one step.  The
    branch is kept here anyway so this reference stays line-for-line with
    the .ino; if the firmware thresholds are ever re-tuned, both sides
    change together.
    """
    if soc_wh < SOC_SAFE_MODE_WH:
        return "SAFE_MODE"
    if prev_state == "SAFE_MODE":
        return "RECOVERY"
    if soc_wh < SOC_CRITICAL_WH:
        return "CRITICAL"
    if soc_wh < SOC_DEGRADED_WH:
        return "DEGRADED"
    if prev_state == "RECOVERY" and soc_wh < SOC_RECOVERY_EXIT_WH:
        return "RECOVERY"
    return "NOMINAL"


def deferrable_on(state: str) -> int:
    """Firmware's ``deferrableOn()``: the payload runs only in NOMINAL."""
    return 1 if state == "NOMINAL" else 0


def build_trace(trial: int, n_steps: int, load_c: float, load_e: float,
                load_d: float):
    """Replay one study3 scenario through the simplified firmware model."""
    scen = s3.make_scenario(trial)
    n = min(n_steps, scen.n)

    soc = SOC0_WH
    state = "NOMINAL"
    rows = []
    for i in range(n):
        harvest_w = float(scen.harvest_w[i])

        # Load served this minute comes from the PREVIOUS step's state
        # (ordering assumption -- see module docstring).
        total_load_w = load_c + load_e + (load_d if deferrable_on(state) else 0)

        # Simplified integrator: eta = 1, no spill, hard clip to [0, cap].
        soc += (harvest_w - total_load_w) * DT_HOURS
        if soc < 0.0:
            soc = 0.0
        if soc > BATTERY_CAPACITY_WH:
            soc = BATTERY_CAPACITY_WH

        # FSM is evaluated against the NEW SoC; its result drives row i's
        # output and row i+1's served load.
        state = evaluate_fsm(soc, state)

        rows.append((i, harvest_w, load_c, load_e, load_d,
                     soc, state, deferrable_on(state)))
    return scen, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Write reference_trace.csv for the ESP32 HIL comparison.")
    ap.add_argument("--trial", type=int, default=0, help="study3 Monte-Carlo trial index (default 0)")
    ap.add_argument("--out", type=Path, default=Path("reference_trace.csv"), help="output CSV path")
    ap.add_argument("--n-steps", type=int, default=N_STEPS_DEFAULT, help="steps to simulate (default 1440)")
    ap.add_argument("--load-scale", type=float, default=1.0,
                    help="multiply all three load tiers (use >1 to force the FSM off NOMINAL)")
    args = ap.parse_args()

    load_c = s3.W_CRITICAL * args.load_scale
    load_e = s3.W_ESSENTIAL * args.load_scale
    load_d = s3.W_DEFERRABLE * args.load_scale

    scen, rows = build_trace(args.trial, args.n_steps, load_c, load_e, load_d)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            # timestamp written as an integer minute index; soc at full
            # double precision so compare_esp32.py can diff it against the
            # firmware's 3-decimal print without losing information.
            w.writerow([r[0], repr(r[1]), r[2], r[3], r[4],
                        repr(r[5]), r[6], r[7]])

    # ---- quick sanity summary so a boring trial is obvious at a glance ----
    socs = [r[5] for r in rows]
    counts = Counter(r[6] for r in rows)
    first_off_nominal = next((r[0] for r in rows if r[6] != "NOMINAL"), None)
    print(f"wrote {args.out}  ({len(rows)} rows, trial {args.trial}, seed {scen.seed})")
    print(f"  SoC range : {min(socs):.2f} .. {max(socs):.2f} Wh  (start {SOC0_WH:.1f})")
    print(f"  states    : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if first_off_nominal is None:
        print("  NOTE: never left NOMINAL -- the FSM was not exercised;")
        print("        try a cloudier --trial or --load-scale 1.6")
    else:
        print(f"  first non-NOMINAL state at minute {first_off_nominal}")


if __name__ == "__main__":
    main()
