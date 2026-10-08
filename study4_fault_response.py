"""
study4_fault_response.py
========================

Study 4 - fault response of a solar EPS load rail: MCU protection vs
hardware protection.

Two protection approaches are compared against one identical fault:

  1. MCU protection      : periodic ADC polling (1 ms tick) -> firmware
                           debounce -> GPIO write -> gate driver turn-off.
                           Total response ~ 2 ms.
  2. Hardware protection : current shunt + sense amp -> comparator -> SR latch
                           -> gate driver turn-off.
                           Total response ~ 2.1 us.

For each approach the script computes trip time, peak fault current, I^2t
let-through and dissipated fault energy, prints a summary table, writes
results/fault_metrics.csv and saves four 150 dpi comparison plots.

Dependencies: numpy, matplotlib (nothing else).  Fully deterministic.
"""

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")  # head-less backend: only PNG files are produced
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration constants - all times in seconds unless stated otherwise
# ---------------------------------------------------------------------------
DT = 1.0e-6          # simulation time step [s] (1 us)
T_END = 1.0e-2       # total simulated window [s] (10 ms)
T_FAULT = 1.0e-3     # instant at which the short-circuit fault starts [s]
I_NOMINAL = 2.0      # pre-fault (healthy) load current [A]
DI_DT = 1.0e6        # fault current ramp rate [A/s] (1 A per microsecond)
V_BUS = 12.0         # stiff DC bus voltage [V]; fault energy goes into R, not the bus
R_FAULT = 0.05       # fault path (shunt) resistance [ohm]

# MCU protection timing
ADC_PERIOD = 1.0e-3             # firmware ADC sampling tick [s]
ADC_CONVERT_READ = 0.05e-3      # ADC conversion + bus read time [s]
DEBOUNCE_SAMPLES = 2            # consecutive confirming samples required
MCU_PROCESSING_DELAY = 1.5e-3   # confirm sample + conversion/read + ISR overhead [s]
MCU_TURNOFF_DELAY = 0.5e-3      # GPIO write + gate driver turn-off [s]

# Hardware protection timing
HW_SHUNT_DELAY = 1.0e-6         # shunt + sense amplifier [s]
HW_COMPARATOR_DELAY = 0.5e-6    # comparator propagation delay [s]
HW_LATCH_DELAY = 0.1e-6         # SR latch propagation delay [s]
HW_GATE_DELAY = 0.5e-6          # gate driver turn-off [s]
HW_TRIP_DELAY = HW_SHUNT_DELAY + HW_COMPARATOR_DELAY + HW_LATCH_DELAY + HW_GATE_DELAY

# Output
RESULTS_DIR = "results"
DPI = 150

# Plot colours (kept consistent across all figures)
COLOR_MCU = "tab:orange"
COLOR_HW = "tab:green"
COLOR_RAW = "tab:blue"


# ---------------------------------------------------------------------------
# Fault model
# ---------------------------------------------------------------------------
def model_current(t):
    """Return the fault current for the time value(s) ``t``.

    Piecewise definition:
        i(t) = I_NOMINAL                       for t <= T_FAULT
        i(t) = I_NOMINAL + DI_DT*(t - T_FAULT)  for t >  T_FAULT
    """
    t_arr = np.asarray(t, dtype=float)
    return np.where(t_arr <= T_FAULT, I_NOMINAL, I_NOMINAL + DI_DT * (t_arr - T_FAULT))


# ---------------------------------------------------------------------------
# Protection timing
# ---------------------------------------------------------------------------
def find_trip_times():
    """Compute the absolute trip instants of both protection paths.

    MCU path      : detection happens on the first ADC tick at/after the
                    fault, then the processing delay and the turn-off delay
                    are added.
    Hardware path : fixed analogue chain delay added to the fault instant.
    """
    # Build the firmware ADC tick grid: 0, 1 ms, 2 ms, ... (integer indexing
    # keeps the ticks exactly representable, unlike np.arange with a float step)
    n_ticks = int(round(T_END / ADC_PERIOD))
    adc_ticks = np.arange(n_ticks) * ADC_PERIOD

    # First ADC tick at or after the fault (T_FAULT = 1 ms lands on tick 1);
    # a tiny tolerance keeps the search robust against float representation
    detect_idx = int(np.searchsorted(adc_ticks, T_FAULT - 1e-12, side="right"))
    detect_idx = min(detect_idx, n_ticks - 1)
    detect_time = float(adc_ticks[detect_idx])

    # MCU: detect -> firmware processing -> gate driver off
    mcu_trip = detect_time + MCU_PROCESSING_DELAY + MCU_TURNOFF_DELAY

    # Hardware: shunt -> comparator -> latch -> gate driver off
    hw_trip = T_FAULT + HW_TRIP_DELAY

    return {
        "adc_ticks": adc_ticks,
        "detect_time": detect_time,
        "mcu_trip": mcu_trip,
        "hw_trip": hw_trip,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _trapz(y, x):
    """Trapezoidal integral of ``y`` over ``x`` (no dependency on np.trapz)."""
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def compute_metrics(t_trip, n_sub=4096):
    """Return trip time, peak current, I^2t and energy for a trip instant.

    The integral is evaluated on a refined sub-grid between T_FAULT and
    ``t_trip`` because the hardware trip lands between two 1 us samples.
    """
    response = float(t_trip) - T_FAULT                 # trip latency after the fault [s]
    u = np.linspace(0.0, response, n_sub + 1)          # time since fault [s]
    i = I_NOMINAL + DI_DT * u                          # current ramp [A]
    i_sq = i * i                                       # i^2 [A^2]
    i2t = _trapz(i_sq, u)                              # let-through I^2t [A^2 s]
    return {
        "trip_time": float(t_trip),                     # absolute trip instant [s]
        "response_time": response,                      # latency after fault [s]
        "peak_current": float(i[-1]),                   # current at trip [A]
        "i2t": i2t,                                    # let-through I^2t [A^2 s]
        "energy": i2t * R_FAULT,                       # dissipated energy [J]
    }


def build_table_rows(mcu, hw):
    """Assemble the (label, MCU, hardware, ratio, number-format) summary rows."""
    return [
        ("Trip time [us]", mcu["response_time"] * 1e6, hw["response_time"] * 1e6, "{:.4f}"),
        ("Peak current [A]", mcu["peak_current"], hw["peak_current"], "{:.4f}"),
        ("I2t [A^2s]", mcu["i2t"], hw["i2t"], "{:.4e}"),
        ("Fault energy [J]", mcu["energy"], hw["energy"], "{:.4e}"),
    ]


def print_summary_table(rows):
    """Print the summary table: Parameter | MCU | Hardware | Ratio."""
    header = ("Parameter", "MCU", "Hardware", "Ratio")
    cells = [header]
    for label, mcu_v, hw_v, fmt in rows:
        cells.append((label, fmt.format(mcu_v), fmt.format(hw_v), "{:.4e}".format(mcu_v / hw_v)))
    widths = [max(len(c[i]) for c in cells) for i in range(4)]

    print("  " + " | ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells[0], widths))))
    print("  " + "-+-".join("-" * w for w in widths))
    for row in cells[1:]:
        print("  " + " | ".join(
            c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))
        ))


def save_csv(rows, path):
    """Write the summary table to a comma-separated file (no pandas)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Parameter,MCU,Hardware,Ratio\n")
        for label, mcu_v, hw_v, _fmt in rows:
            fh.write("{},{:.6e},{:.6e},{:.6e}\n".format(label, mcu_v, hw_v, mcu_v / hw_v))
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _bar_annotate(ax, positions, values, text_fmt, log_scale=False):
    """Annotate every bar with its value just above the bar top."""
    for pos, val in zip(positions, values):
        label = text_fmt.format(val)
        if log_scale:
            ax.annotate(label, xy=(pos, val), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=9)
        else:
            ax.annotate(label, xy=(pos, val), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=9)


def _set_log_limits(ax, values, pad_lo=15.0, pad_hi=40.0):
    """Give a log-scaled bar axis enough head/foot room for the annotations."""
    lo = min(values) / pad_lo
    hi = max(values) * pad_hi
    ax.set_ylim(lo, hi)


def make_plots(t, i, trips, mcu, hw, outdir):
    """Create and save the four comparison figures as PNG files."""
    labels = ["MCU", "Hardware"]
    colors = [COLOR_MCU, COLOR_HW]
    mcu_trip, hw_trip = trips["mcu_trip"], trips["hw_trip"]

    # ---------------- 1. fault current waveform -----------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    t_ms = t * 1e3
    i_mcu = np.where(t <= mcu_trip, i, 0.0)   # current after the MCU has tripped
    i_hw = np.where(t <= hw_trip, i, 0.0)     # current after the hardware latch
    ax.plot(t_ms, i, color=COLOR_RAW, lw=1.0, ls=":", label="Unprotected fault current")
    ax.plot(t_ms, i_mcu, color=COLOR_MCU, lw=1.6, label="MCU trip @ {:.3f} ms".format(mcu_trip * 1e3))
    ax.plot(t_ms, i_hw, color=COLOR_HW, lw=1.6, label="Hardware trip @ {:.4f} ms".format(hw_trip * 1e3))
    ax.axvline(mcu_trip * 1e3, color=COLOR_MCU, ls="--", lw=1.2, label="MCU trip time")
    ax.axvline(hw_trip * 1e3, color=COLOR_HW, ls="--", lw=1.2, label="Hardware trip time")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Load-rail current [A]")
    ax.set_title("Study 4 - fault current and protection trip instants")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "study4_fault_current.png"), dpi=DPI)
    plt.close(fig)

    # ---------------- 2. trip time (log scale) ------------------------------
    trip_us = [mcu["response_time"] * 1e6, hw["response_time"] * 1e6]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    pos = np.arange(len(labels))
    ax.bar(pos, trip_us, width=0.55, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    _set_log_limits(ax, trip_us)
    _bar_annotate(ax, pos, trip_us, "{:.2f} us", log_scale=True)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Trip time after fault [us]  (log scale)")
    ax.set_title("Protection trip time: MCU vs hardware")
    ax.grid(True, axis="y", which="both", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "study4_trip_time.png"), dpi=DPI)
    plt.close(fig)

    # ---------------- 3. peak current ---------------------------------------
    peak = [mcu["peak_current"], hw["peak_current"]]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.bar(pos, peak, width=0.55, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0.0, max(peak) * 1.25)
    _bar_annotate(ax, pos, peak, "{:.2f} A")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Peak fault current [A]")
    ax.set_title("Peak current at trip: MCU vs hardware")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "study4_peak_current.png"), dpi=DPI)
    plt.close(fig)

    # ---------------- 4. I^2t let-through (log scale) -----------------------
    i2t = [mcu["i2t"], hw["i2t"]]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.bar(pos, i2t, width=0.55, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    _set_log_limits(ax, i2t)
    _bar_annotate(ax, pos, i2t, "{:.4e}", log_scale=True)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"I$^2$t let-through [A$^2\cdot$s]  (log scale)")
    ax.set_title("I^2t let-through: MCU vs hardware")
    ax.grid(True, axis="y", which="both", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "study4_i2t.png"), dpi=DPI)
    plt.close(fig)

    return [
        os.path.join(outdir, "study4_fault_current.png"),
        os.path.join(outdir, "study4_trip_time.png"),
        os.path.join(outdir, "study4_peak_current.png"),
        os.path.join(outdir, "study4_i2t.png"),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Run the study: model the fault, evaluate both paths, report and plot."""
    # Create the output folder if it does not exist yet
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 1. Build the 1 us time base and the fault current waveform --------
    t = np.arange(0.0, T_END + 0.5 * DT, DT)   # 0 .. 10 ms in 1 us steps
    i = model_current(t)
    print("=" * 68)
    print("Study 4 - EPS load rail fault response (MCU vs hardware protection)")
    print("=" * 68)
    print("Bus voltage        : {:.1f} V (stiff)".format(V_BUS))
    print("Fault resistance   : {:.3f} ohm".format(R_FAULT))
    print("Time step          : {:.0f} us   Window: 0 - {:.0f} ms   Samples: {}".format(
        DT * 1e6, T_END * 1e3, t.size))
    print("Fault start        : {:.0f} ms   Nominal current: {:.1f} A   di/dt: {:.0f} A/s".format(
        T_FAULT * 1e3, I_NOMINAL, DI_DT))
    print("-" * 68)

    # ---- 2. Protection timing ----------------------------------------------
    trips = find_trip_times()
    print("MCU timing         : ADC tick {:.0f} ms -> detect {:.3f} ms -> "
          "process {:.2f} ms -> turn-off {:.2f} ms".format(
              ADC_PERIOD * 1e3, trips["detect_time"] * 1e3,
              MCU_PROCESSING_DELAY * 1e3, MCU_TURNOFF_DELAY * 1e3))
    print("Hardware timing    : shunt {:.1f} us + comparator {:.1f} us + latch {:.1f} us "
          "+ gate {:.1f} us = {:.1f} us".format(
              HW_SHUNT_DELAY * 1e6, HW_COMPARATOR_DELAY * 1e6, HW_LATCH_DELAY * 1e6,
              HW_GATE_DELAY * 1e6, HW_TRIP_DELAY * 1e6))
    print("-" * 68)

    # ---- 3. Metrics for both protection paths ------------------------------
    mcu = compute_metrics(trips["mcu_trip"])
    hw = compute_metrics(trips["hw_trip"])
    print("MCU      trip at {:.6f} s ({:.3f} ms after fault)".format(
        mcu["trip_time"], mcu["response_time"] * 1e3))
    print("Hardware trip at {:.6f} s ({:.3f} us after fault)".format(
        hw["trip_time"], hw["response_time"] * 1e6))
    print("-" * 68)

    # Cross-check the refined integral against the raw 1 us sample grid
    for name, metrics in (("MCU", mcu), ("Hardware", hw)):
        mask = (t >= T_FAULT) & (t <= metrics["trip_time"])
        grid_i2t = _trapz(i[mask] ** 2, t[mask])
        print("  [check] {:8s} I2t refined {:.6e} A^2s | 1 us grid {:.6e} A^2s".format(
            name, metrics["i2t"], grid_i2t))
    print("-" * 68)

    # ---- 4. Summary table + CSV --------------------------------------------
    rows = build_table_rows(mcu, hw)
    print_summary_table(rows)
    csv_path = save_csv(rows, os.path.join(RESULTS_DIR, "fault_metrics.csv"))
    print("\nTable written to : {}".format(csv_path))

    # ---- 5. Figures ---------------------------------------------------------
    paths = make_plots(t, i, trips, mcu, hw, RESULTS_DIR)
    print("\nPlots written to :")
    for path in paths:
        print("  {}".format(path))
    print("=" * 68)
    print("Done. MCU lets through {:.3e} x the hardware path energy ({:.1f} J vs {:.3e} J).".format(
        mcu["energy"] / hw["energy"], mcu["energy"], hw["energy"]))


if __name__ == "__main__":
    main()
