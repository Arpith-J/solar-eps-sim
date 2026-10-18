"""
compare_esp32.py
================

Hardware-in-the-loop comparison between the Python reference model and the
ESP32 firmware (``esp32_validation/esp32_validation.ino``).

The script replays ``reference_trace.csv`` (written by ``dump_reference.py``)
over Serial: every row is sent to the board, the board runs its simplified
SoC integrator + FSM on it, and the board's CSV answer is diffed against the
Python reference.

Usage:
    pip install pyserial
    python dump_reference.py --trial 0          # once -> reference_trace.csv
    python compare_esp32.py COM3

Outputs:
    comparison_report.csv   timestamp,soc_py,soc_esp32,soc_diff,fsm_py,fsm_esp32,match

ORDERING ASSUMPTION (mirrored by the firmware and the reference):
    The load served during minute i is decided from the FSM state computed
    at the end of minute i-1 (from the previous SoC); the state logged on
    row i is re-evaluated against the NEW SoC and governs row i+1's load.
    The two models must step in this same order or the traces will drift at
    every threshold crossing.

Expected residual error: the firmware prints SoC with 3 decimals and
accumulates in 32-bit float, so max |soc_diff| below ~1e-2 Wh is normal
numeric noise, not a logic bug.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import serial
    from serial import SerialException
except ImportError:
    sys.exit("ERROR: pyserial is required.  Install it with:  pip install pyserial")

REFERENCE_COLUMNS = [
    "timestamp", "harvest_w", "load_c", "load_e", "load_d",
    "soc_py", "fsm_py", "payload_py",
]
REPORT_COLUMNS = [
    "timestamp", "soc_py", "soc_esp32", "soc_diff", "fsm_py", "fsm_esp32", "match",
]
TS_TOLERANCE = 0.011  # firmware echoes timestamps at 2 decimals


@dataclass
class RowResult:
    ts: float
    soc_py: float
    soc_esp32: float
    soc_diff: float
    fsm_py: str
    fsm_esp32: str
    payload: int          # load_deferrable_on from the firmware
    match: int


def load_reference(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REFERENCE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"ERROR: {path} is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        sys.exit(f"ERROR: {path} contains no data rows")
    return rows


def hard_reset(ser: serial.Serial) -> None:
    """
    Best-effort manual reset via the classic DTR/RTS auto-reset circuit:
    DTR low keeps IO0 high (run mode), pulsing RTS low resets through EN.
    Opening the port usually resets the board anyway, and on some drivers
    these ioctls are rejected -- both cases are fine, we just wait for READY.
    """
    try:
        ser.dtr = False
        ser.rts = True
        time.sleep(0.1)
        ser.rts = False
        time.sleep(0.05)
    except (SerialException, OSError):
        pass
    try:
        ser.reset_input_buffer()
    except (SerialException, OSError):
        pass


def wait_for_ready(ser: serial.Serial, port: str, timeout_s: float) -> str:
    """
    Read lines until the firmware's boot banner "READY,..." arrives.
    Any boot noise (ROM banner etc.) seen beforehand is swallowed; on
    timeout we bail out with a hint list instead of hanging.
    """
    deadline = time.monotonic() + timeout_s
    noise: list[str] = []
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if line.startswith("READY"):
            return line
        if line:
            noise.append(line)
    msg = [
        f"ERROR: no READY line received on {port} within {timeout_s:.1f} s.",
        "  - check the port name and that esp32_validation.ino is flashed",
        "  - install the correct USB driver for your ESP32 board",
        "  - some boards do not auto-reset on open: press RESET and re-run",
    ]
    if noise:
        msg.append(f"  - last lines seen instead: {noise[-3:]!r}")
    sys.exit("\n".join(msg))


def ask(ser: serial.Serial, msg: str, timeout_s: float) -> str:
    """Send one line, read one response line, with a clear timeout error."""
    ser.write(msg.encode("ascii"))
    ser.flush()
    raw = ser.readline()
    if not raw:
        sys.exit(
            f"ERROR: ESP32 sent no response within {timeout_s:.1f} s "
            f"after sending:\n  {msg.strip()!r}\n"
            "  - the board may have reset mid-stream or the sketch crashed; "
            "re-run (the board restarts the trace from its initial SoC)"
        )
    return raw.decode("ascii", errors="replace").strip()


def run(rows: list[dict], port: str, baud: int, ready_timeout: float,
        resp_timeout: float) -> list[RowResult]:
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
    except SerialException as exc:
        sys.exit(f"ERROR: could not open {port}: {exc}")

    results: list[RowResult] = []
    try:
        hard_reset(ser)
        ready = wait_for_ready(ser, port, ready_timeout)
        print(f"ESP32 ready: {ready}")
        ser.timeout = resp_timeout

        n = len(rows)
        for idx, r in enumerate(rows):
            ts = float(r["timestamp"])
            # Send at higher precision than the firmware prints; the board's
            # float32 parse is the binding constraint anyway.
            ser_line = (
                f"{ts:g},{float(r['harvest_w']):.6f},"
                f"{float(r['load_c']):.3f},{float(r['load_e']):.3f},"
                f"{float(r['load_d']):.3f}\n"
            )
            rx = ask(ser, ser_line, resp_timeout)

            if rx.startswith("ERR"):
                sys.exit(f"ERROR: firmware rejected row {idx} (ts {ts:g}): {rx}")
            parts = rx.split(",")
            if len(parts) != 4:
                sys.exit(f"ERROR: unexpected response at row {idx} (ts {ts:g}): {rx!r}")
            try:
                ts_rx, soc_esp32, fsm_esp32, payload = (
                    float(parts[0]), float(parts[1]), parts[2].strip(), int(parts[3]),
                )
            except ValueError:
                sys.exit(f"ERROR: unparsable response at row {idx}: {rx!r}")

            # The board echoes the timestamp it parsed -- a mismatch means the
            # two ends have lost sync on line boundaries, not just noise.
            if abs(ts_rx - ts) > TS_TOLERANCE:
                sys.exit(
                    f"ERROR: serial desync at row {idx}: sent ts {ts:g}, "
                    f"board echoed {ts_rx:g}. Re-run the comparison."
                )

            soc_py = float(r["soc_py"])
            results.append(RowResult(
                ts=ts,
                soc_py=soc_py,
                soc_esp32=soc_esp32,
                soc_diff=soc_esp32 - soc_py,
                fsm_py=r["fsm_py"].strip(),
                fsm_esp32=fsm_esp32,
                payload=payload,
                match=int(r["fsm_py"].strip() == fsm_esp32),
            ))
            if (idx + 1) % 200 == 0 or idx + 1 == n:
                print(f"  {idx + 1}/{n} rows exchanged")
    finally:
        ser.close()
    return results


def write_report(path: Path, results: list[RowResult]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(REPORT_COLUMNS)
        for r in results:
            w.writerow([f"{r.ts:g}", repr(r.soc_py), repr(r.soc_esp32),
                        repr(r.soc_diff), r.fsm_py, r.fsm_esp32, r.match])


def print_summary(results: list[RowResult], report_path: Path) -> None:
    mismatches = [r for r in results if not r.match]
    max_abs = max((abs(r.soc_diff) for r in results), default=0.0)
    print()
    print("=" * 62)
    print("COMPARISON SUMMARY")
    print(f"  rows compared        : {len(results)}")
    print(f"  max |SoC| error [Wh] : {max_abs:.6f}")
    print(f"  FSM mismatches       : {len(mismatches)}")
    if mismatches:
        print("  first mismatches:")
        for r in mismatches[:5]:
            print(f"    t={r.ts:g}  py={r.fsm_py}  esp32={r.fsm_esp32}  "
                  f"(soc py={r.soc_py:.3f} esp32={r.soc_esp32:.3f})")
        if len(mismatches) > 5:
            print(f"    ... {len(mismatches) - 5} more in {report_path}")
    print(f"  report written to    : {report_path}")
    print("=" * 62)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay reference_trace.csv through the ESP32 and diff against Python.")
    ap.add_argument("port", help="serial port, e.g. COM3 or /dev/ttyUSB0")
    ap.add_argument("--reference", type=Path, default=Path("reference_trace.csv"),
                    help="input trace from dump_reference.py (default reference_trace.csv)")
    ap.add_argument("--report", type=Path, default=Path("comparison_report.csv"),
                    help="output report path (default comparison_report.csv)")
    ap.add_argument("--baud", type=int, default=115200, help="baud rate (default 115200)")
    ap.add_argument("--ready-timeout", type=float, default=12.0,
                    help="seconds to wait for the READY banner (default 12; boot takes ~1-2 s)")
    ap.add_argument("--timeout", type=float, default=3.0,
                    help="seconds to wait for each per-row response (default 3)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only replay the first N rows (0 = all, for a quick smoke test)")
    args = ap.parse_args()

    rows = load_reference(args.reference)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} rows from {args.reference}")

    results = run(rows, args.port, args.baud, args.ready_timeout, args.timeout)
    write_report(args.report, results)
    print_summary(results, args.report)


if __name__ == "__main__":
    main()
