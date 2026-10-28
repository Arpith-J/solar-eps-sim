import numpy as np
import csv
import argparse

# Import everything from dump_reference
from dump_reference import *
from study3_energy_budgeting import make_scenario, W_CRITICAL, W_ESSENTIAL, W_DEFERRABLE, BAT_CAPACITY_WH, ETA_CHG, ETA_DIS

# Override constants for 3-day simulation with 35-minute eclipse per 92.6 min orbit
N_STEPS = 4320  # 3 days * 1440 minutes
ORBIT_WRAP_MIN = 92.6
ECLIPSE_LEN_MIN = 35.0
ECLIPSE_START_MIN = 0.0  # Eclipse starts at beginning of orbit as per example
# Eclipse offset no longer needed as per requirement 4

# Override shape_at for new orbit
def shape_at(minute):
    """Normalised clear-sky geometry in [0, 1]: half-sine over sunlit arc
    with 35-minute eclipse in each 92.6-minute orbit."""
    tt = float(minute) % float(ORBIT_WRAP_MIN)
    if tt < 0:
        tt += float(ORBIT_WRAP_MIN)
    # Eclipse is 35 minutes - shape is 0 during eclipse
    # Sunlit is the rest of the orbit - half-sine during sunlit period
    if tt < ECLIPSE_LEN_MIN:  # First 35 minutes are eclipse
        return np.float32(0.0)
    else:  # Rest of orbit (92.6 - 35 = 57.6 minutes) is sunlit
        # Map sunlit period to half-sine
        sunlit_duration = ORBIT_WRAP_MIN - ECLIPSE_LEN_MIN
        t_sunlit = tt - ECLIPSE_LEN_MIN  # time into sunlit period
        x = np.float32(t_sunlit) / np.float32(sunlit_duration)
        if 0.0 <= x <= 1.0:
            s = np.float32(np.sin(_f32(np.pi * float(x))))
        else:
            s = np.float32(0.0)
        if s < 0.0:
            s = np.float32(0.0)
        if s > 1.0:
            s = np.float32(1.0)
        return s


# Override dark_hours_at to use new shape_at
def dark_hours_at(i):
    """Geometry.dark_hours port for new orbit geometry."""
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


def build_trace_3day(trial, horizon, debug):
    """Replay scenario through the ported firmware controller for 3 days."""
    scen = make_scenario(trial=trial)
    # For 3 days, we need to repeat or extend the harvest pattern
    harvest = scen.harvest_w  # 1440 values
    # Repeat for 3 days
    harvest_3day = np.tile(harvest, 3)
    
    fc = Esp32Forecaster()  # fcInit() in __init__
    ctrl = Esp32Controller()  # firmware globals
    rows = []
    state_transitions = []
    prev_state = ctrl.state
    
    for i in range(N_STEPS):
        harvest_full = float(harvest_3day[i])
        # compare_esp32.py sends "%.6f"; the firmware parses that into a
        # float32 -- emulate the same value so both sides compute identically
        harvest_tx = np.float32(float(f"{harvest_full:.6f}"))
        row, state_changed = ctrl.step(i, harvest_tx, harvest_full, fc, horizon, debug)
        rows.append(row)
        if state_changed or (i == 0):
            if i == 0 or ctrl.state != prev_state:
                state_transitions.append((i, prev_state, ctrl.state))
                prev_state = ctrl.state
    return scen, rows, state_transitions


def main():
    ap = argparse.ArgumentParser(
        description="Write reference_trace_3day.csv (exact esp32_full.ino mirror with 35-min eclipse).")
    ap.add_argument("--trial", type=int, default=0,
                    help="study3 Monte-Carlo trial index (default 0)")
    ap.add_argument("--out", default="reference_trace_3day.csv",
                    help="output CSV path (default reference_trace_3day.csv)")
    ap.add_argument("--horizon", type=int, default=HORIZON_MIN,
                    help=f"planning horizon in minutes (default {HORIZON_MIN})")
    ap.add_argument("--debug", action="store_true",
                    help="print t/soc/margin/forecast/load/reserve/state at "
                         "every FSM state change")
    args = ap.parse_args()

    scen, rows, state_transitions = build_trace_3day(args.trial, args.horizon, args.debug)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    # ---- sanity summary ---------------------------------------------------
    socs = [r[5] for r in rows]
    counts = {}
    for r in rows:
        counts[r[6]] = counts.get(r[6], 0) + 1

    print(f"wrote {args.out}  ({len(rows)} rows, trial {args.trial}, "
          f"seed {scen.seed}, horizon {args.horizon} min)")
    print(f"  SoC range : {min(socs):.2f} .. {max(socs):.2f} Wh  "
          f"(start {SOC0_WH:.1f})")
    print("  states    : " + ", ".join(f"{k}={v}"
                                       for k, v in sorted(counts.items())))
    print("  State transitions:")
    for trans in state_transitions:
        print(f"    t={trans[0]}: {trans[1]} -> {trans[2]}")


if __name__ == "__main__":
    main()
