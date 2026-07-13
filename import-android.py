"""
vigil — import sleep recordings from Android device

Scans a connected Android phone (via adb) for vigil sleep recordings and
imports any new sessions into the local data/ directory.

The Android app stores each recording as a timestamped folder under:
    /storage/emulated/0/Android/data/com.vigil.android/files/Documents/vigil/<timestamp>/

Each session folder contains:
    sleep_ppg_<timestamp>.csv   (55 Hz PPG, 4 channels)
    sleep_acc_<timestamp>.csv   (52 Hz ACC, 3 axes)

This script copies them into the local structure:
    data/<timestamp>/input/sleep_ppg_<timestamp>.csv
    data/<timestamp>/input/sleep_acc_<timestamp>.csv

Only sessions not already present locally are imported. Use --force to
re-import existing ones, or --dry-run to preview without copying.

On import, timestamps are normalized from Polar-epoch (2000-01-01, the
Polar sensor's native epoch) to Unix-epoch (1970-01-01) by adding
946684800000000000 ns when needed. Recordings made with older Android app
versions that wrote raw sensor timestamps are auto-fixed in place.

Usage:
    uv run import-android.py                                   # import new sessions
    uv run import-android.py --dry-run                         # preview only
    uv run import-android.py --force                           # re-import all
    uv run import-android.py --session 20260711_231213         # import one session

After a successful import of a single session, the script prompts to run
sleep_staging.py and plot_hypnogram.py automatically (skip with input
redirected from a non-tty, or answer 'n').
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

REMOTE_BASE = (
    "/storage/emulated/0/Android/data/com.vigil.android/files/Documents/vigil"
)

# Polar sensors report nanoseconds since 2000-01-01 (Polar epoch), not
# 1970-01-01 (Unix epoch). The Android SDK writes the raw sensor timestamp
# without applying this offset, so recordings appear ~30 years in the past.
# We add it on import so vigil works with Unix-epoch nanoseconds throughout.
# (The Python polar-python SDK already adds this offset internally.)
# Reference: https://github.com/polarofficial/polar-ble-sdk/blob/master/documentation/TimeSystemExplained.md
POLAR_TO_UNIX_EPOCH_NS = 946684800_000_000_000

ADB_PATH = os.environ.get(
    "ADB",
    os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
)


def adb(*args, check=True):
    """Run an adb command, return CompletedProcess with captured output."""
    return subprocess.run(
        [ADB_PATH, *args],
        check=check,
        capture_output=True,
        text=True,
    )


def is_timestamp_dir(name):
    """True if name looks like a vigil session timestamp (YYYYMMDD_HHMMSS)."""
    return (
        len(name) == 15
        and name[8] == "_"
        and name[:8].isdigit()
        and name[9:].isdigit()
    )


def needs_epoch_fix(csv_path):
    """True if the first timestamp_ns in csv_path predates 2020 UTC.

    Polar sensor timestamps are nanoseconds since 2000-01-01. The Android
    SDK writes them without adding the Unix-epoch offset, so they appear
    ~30 years in the past (e.g. 1996 instead of 2026). We detect that here
    so we can add the offset on import.
    """
    try:
        import csv as _csv
        with open(csv_path, newline="") as f:
            reader = _csv.reader(f)
            next(reader, None)  # header
            row = next(reader, None)
            if not row:
                return False
            ts_ns = int(row[0])
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        # A correct Unix-epoch timestamp in 2026 lands in 2026; an unfixed
        # Polar-epoch timestamp lands ~30 years earlier (1996-ish). 2020 is a
        # safe threshold — vigil recordings only started in 2026.
        return dt.year < 2020
    except (ValueError, OSError, StopIteration):
        return False


def fix_epoch(csv_path):
    """Add POLAR_TO_UNIX_EPOCH_NS to every timestamp_ns in csv_path (in place)."""
    import tempfile
    import csv as _csv
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".vigil_fix_", suffix=".csv", dir=os.path.dirname(csv_path)
    )
    os.close(tmp_fd)
    try:
        with open(csv_path, newline="") as src, open(tmp_path, "w", newline="") as dst:
            reader = _csv.reader(src)
            writer = _csv.writer(dst)
            header = next(reader, None)
            if header:
                writer.writerow(header)
            for row in reader:
                if row:
                    row[0] = str(int(row[0]) + POLAR_TO_UNIX_EPOCH_NS)
                    writer.writerow(row)
        os.replace(tmp_path, csv_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def list_remote_sessions():
    """List session timestamp directories on the device. Returns sorted list."""
    result = adb("shell", "ls", REMOTE_BASE, check=False)
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"adb ls failed: {msg}")
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return sorted(n for n in names if is_timestamp_dir(n))


def list_remote_files(session):
    """List files inside a remote session directory."""
    result = adb("shell", "ls", f"{REMOTE_BASE}/{session}", check=False)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_local_sessions():
    """List session timestamp directories already present in data/."""
    if not os.path.isdir(DATA_DIR):
        return set()
    return {n for n in os.listdir(DATA_DIR) if is_timestamp_dir(n)}


def pull_session(session, force=False):
    """Pull a single session's files into data/<session>/input/. Returns count pulled."""
    input_dir = os.path.join(DATA_DIR, session, "input")
    os.makedirs(input_dir, exist_ok=True)

    remote_files = list_remote_files(session)
    if not remote_files:
        print(f"  ! No files found in remote {session}/")
        return 0

    imported = 0
    skipped = 0
    for fname in remote_files:
        local_path = os.path.join(input_dir, fname)
        if os.path.exists(local_path) and not force:
            skipped += 1
            continue
        remote_path = f"{REMOTE_BASE}/{session}/{fname}"
        r = adb("pull", remote_path, local_path, check=False)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else "failed"
            print(f"  x {fname}: {msg}")
            continue
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  + {fname}  ({size_mb:.1f} MB)")
        # Normalize Polar-epoch timestamps to Unix epoch if needed
        if needs_epoch_fix(local_path):
            fix_epoch(local_path)
            print(f"    ↳ added Polar→Unix epoch offset to timestamps")
        imported += 1

    if skipped:
        print(f"  . {skipped} file(s) already present, skipped (use --force to overwrite)")
    return imported


def main():
    global ADB_PATH
    parser = argparse.ArgumentParser(
        description="Import sleep recordings from a connected Android device."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="List new sessions without copying")
    parser.add_argument("--force", action="store_true",
                        help="Re-import files even if they already exist locally")
    parser.add_argument("--session", type=str, default=None,
                        help="Import a specific session timestamp (e.g. 20260711_231213)")
    parser.add_argument("--adb", type=str, default=ADB_PATH,
                        help=f"Path to adb binary (default: {ADB_PATH})")
    args = parser.parse_args()

    ADB_PATH = args.adb

    print("╔══════════════════════════════════════════╗")
    print("║  vigil — Android import                  ║")
    print("╚══════════════════════════════════════════╝")

    try:
        subprocess.run([ADB_PATH, "version"], capture_output=True, check=False)
    except FileNotFoundError:
        print(f"Error: adb not found at {ADB_PATH}")
        print("Install Android platform-tools, set ADB env var, or use --adb PATH.")
        sys.exit(1)

    devices = adb("devices", check=False)
    connected = [
        l for l in devices.stdout.splitlines()
        if "\tdevice" in l and "List of devices" not in l
    ]
    if not connected:
        print("No Android device connected. Plug one in, enable USB debugging, then run:")
        print("  adb devices")
        sys.exit(1)

    print(f"Scanning device: {REMOTE_BASE}")
    try:
        remote_sessions = list_remote_sessions()
    except RuntimeError as e:
        print(f"  {e}")
        print("  (Is the vigil app installed and has it recorded at least one session?)")
        sys.exit(1)

    local_sessions = list_local_sessions()
    print(f"  {len(remote_sessions)} session(s) on device, "
          f"{len(local_sessions)} already imported locally")

    if args.session:
        if args.session not in remote_sessions:
            print(f"  Session {args.session} not found on device")
            sys.exit(1)
        to_import = [args.session]
    else:
        to_import = [s for s in remote_sessions if s not in local_sessions]

    if not to_import:
        print("\n  No new sessions to import.")
        return

    print(f"\n  Sessions to {'preview' if args.dry_run else 'import'}: {len(to_import)}")
    for s in to_import:
        files = list_remote_files(s)
        status = "exists" if s in local_sessions else "new"
        print(f"    {s}  [{status}]  "
              f"({len(files)} file(s): {', '.join(files) if files else 'empty'})")

    if args.dry_run:
        print("\n  --dry-run: nothing copied.")
        return

    imported_count = 0
    for s in to_import:
        print(f"\n  Importing {s}...")
        if pull_session(s, force=args.force):
            imported_count += 1

    print()
    print("=" * 50)
    print(f"  Imported {imported_count} session(s)")
    print("=" * 50)
    if imported_count > 0:
        last = to_import[-1]
        ppg_path = f"data/{last}/input/sleep_ppg_{last}.csv"
        stages_path = f"data/{last}/analysis/sleep_stages_{last}.csv"
        hypno_path = f"data/{last}/analysis/hypnogram_{last}.png"

        print("\n  Next:")
        print(f"    uv run sleep_staging.py {ppg_path}")
        print(f"    uv run plot_hypnogram.py {stages_path}")

        if args.session or imported_count == 1:
            try:
                answer = input("\n  Run sleep staging + hypnogram now? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in ("", "y", "yes"):
                print(f"\n  → uv run sleep_staging.py {ppg_path}")
                ret = subprocess.run(
                    ["uv", "run", "sleep_staging.py", ppg_path],
                    cwd=SCRIPT_DIR,
                )
                if ret.returncode != 0:
                    print("  sleep_staging.py failed, skipping hypnogram.")
                    sys.exit(ret.returncode)
                if not os.path.exists(os.path.join(SCRIPT_DIR, stages_path)):
                    print(f"  Expected output not found: {stages_path}")
                    sys.exit(1)
                print(f"\n  → uv run plot_hypnogram.py {stages_path}")
                ret = subprocess.run(
                    ["uv", "run", "plot_hypnogram.py", stages_path],
                    cwd=SCRIPT_DIR,
                )
                if ret.returncode != 0:
                    print("  plot_hypnogram.py failed.")
                    sys.exit(ret.returncode)
                if os.path.exists(os.path.join(SCRIPT_DIR, hypno_path)):
                    print(f"\n  Hypnogram: {hypno_path}")


if __name__ == "__main__":
    main()