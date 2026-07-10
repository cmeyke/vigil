"""
vigil — overnight sleep recording

Run: uv run record_sleep.py

Streams raw PPG (55 Hz) + ACC (52 Hz) from the Polar Verity Sense
for the duration of your sleep. Saves data to CSV files. Auto-reconnects
on BLE disconnect. Shows minimal status display.

Ctrl+C to stop and save.

Usage:
  uv run record_sleep.py                    # record until Ctrl+C
  uv run record_sleep.py --hours 8          # auto-stop after 8 hours
"""

import asyncio
import csv
import time
import os
import signal
import sys
from datetime import datetime
from collections import deque
from bleak import BleakScanner
from polar_python import PolarDevice
from polar_python.models import PPGData, ACCData
import numpy as np
from sensor import find_sensor

FS_PPG = 55
FS_ACC = 52
BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
OUTPUT_DIR = "data"
FLUSH_INTERVAL = 60  # flush CSV every N seconds
STATUS_INTERVAL = 5  # update display every N seconds
RECONNECT_DELAY = 3  # seconds between reconnect attempts

# Parse args
max_hours = 0
args = sys.argv[1:]
for i, arg in enumerate(args):
    if arg == "--hours" and i + 1 < len(args):
        max_hours = float(args[i + 1])

# State
recording = True
ppg_count = 0
acc_count = 0
ppg_file = None
acc_file = None
ppg_writer = None
acc_writer = None
battery_level = 0
last_flush = 0
last_status = 0
start_time = 0
session_timestamp = None


async def connect_and_stream():
    """Connect to sensor and stream until disconnected or stopped."""
    global ppg_count, acc_count, ppg_writer, acc_writer, battery_level
    global ppg_file, acc_file, recording, start_time, max_hours

    print(f"  Scanning for sensor...")
    device = await find_sensor(timeout=15.0)
    if not device:
        return False
    polar_device = PolarDevice(device)
    await polar_device.connect()

    # Read battery
    try:
        battery_data = await polar_device._client.read_gatt_char(BATTERY_UUID)
        battery_level = battery_data[0]
    except Exception:
        pass

    # Open CSV writers (append mode if resuming)
    is_new_ppg = ppg_file is None
    if is_new_ppg:
        ppg_path = f"{OUTPUT_DIR}/sleep_ppg_{session_timestamp}.csv"
        ppg_file = open(ppg_path, "a", newline="")
        ppg_writer = csv.writer(ppg_file)
        ppg_writer.writerow(["timestamp_ns", "ch0", "ch1", "ch2", "ch3"])

        acc_path = f"{OUTPUT_DIR}/sleep_acc_{session_timestamp}.csv"
        acc_file = open(acc_path, "a", newline="")
        acc_writer = csv.writer(acc_file)
        acc_writer.writerow(["timestamp_ns", "x_mg", "y_mg", "z_mg"])
        print(f"  📁 Saving to: {ppg_path}")
        print(f"  📁 Saving to: {acc_path}")

    def ppg_callback(data: PPGData):
        global ppg_count
        for sample in data.samples:
            ppg_writer.writerow([data.timestamp, sample[0], sample[1], sample[2], sample[3]])
            ppg_count += 1

    def acc_callback(data: ACCData):
        global acc_count
        for sample in data.data:
            acc_writer.writerow([data.timestamp, sample[0], sample[1], sample[2]])
            acc_count += 1

    await polar_device.start_ppg_stream(
        ppg_callback=ppg_callback, sample_rate=55, resolution=22, channels=4
    )
    await polar_device.start_acc_stream(
        acc_callback=acc_callback, sample_rate=52, resolution=16, range=8, channels=3
    )

    print("  ✅ Streaming started")

    # Keep streaming until disconnected or stopped
    while recording:
        await asyncio.sleep(STATUS_INTERVAL)

        # Check max duration
        if max_hours > 0 and (time.monotonic() - start_time) >= max_hours * 3600:
            print("\n  ⏰ Duration limit reached")
            recording = False
            break

        # Periodic status
        elapsed = time.monotonic() - start_time
        hours = elapsed / 3600
        print(f"\r  ⏱  {hours:.1f}h | PPG: {ppg_count:,} | ACC: {acc_count:,} | 🔋 {battery_level}%",
              end="", flush=True)

        # Flush CSV periodically
        if ppg_file:
            ppg_file.flush()
        if acc_file:
            acc_file.flush()

    try:
        await polar_device.stop_ppg_stream()
        await polar_device.stop_acc_stream()
        await polar_device.disconnect()
    except Exception:
        pass

    return True


async def main():
    global recording, start_time, session_timestamp, last_flush

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.monotonic()

    print("╔══════════════════════════════════════════╗")
    print("║  vigil — sleep recording                 ║")
    print("╚══════════════════════════════════════════╝")
    if max_hours > 0:
        print(f"  Duration: {max_hours:.2f} hours (auto-stop)")
    else:
        print("  Duration: until Ctrl+C")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Handle Ctrl+C
    def stop_handler():
        global recording
        recording = False
        print("\n  Stopping...")
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_handler)
        except NotImplementedError:
            pass

    # Main loop with auto-reconnect
    while recording:
        success = await connect_and_stream()

        if not success:
            print(f"  Retrying in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)
            continue

        if recording:
            # Connection dropped but we're still supposed to be recording
            print(f"  Reconnecting in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)

    # Close files
    if ppg_file:
        ppg_file.close()
    if acc_file:
        acc_file.close()

    # Summary
    elapsed = time.monotonic() - start_time
    hours = elapsed / 3600
    ppg_minutes = ppg_count / (FS_PPG * 60)
    acc_minutes = acc_count / (FS_ACC * 60)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║  recording complete                       ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Duration:      {hours:.2f} hours ({elapsed:.0f}s)")
    print(f"  PPG samples:    {ppg_count:,}  ({ppg_minutes:.1f} min @ {FS_PPG} Hz)")
    print(f"  ACC samples:    {acc_count:,}  ({acc_minutes:.1f} min @ {FS_ACC} Hz)")
    print(f"  Battery:        {battery_level}%")
    print(f"  Session:        {session_timestamp}")
    if ppg_file:
        ppg_size = os.path.getsize(f"{OUTPUT_DIR}/sleep_ppg_{session_timestamp}.csv")
        acc_size = os.path.getsize(f"{OUTPUT_DIR}/sleep_acc_{session_timestamp}.csv")
        total_mb = (ppg_size + acc_size) / (1024 * 1024)
        print(f"  Disk:           {total_mb:.1f} MB")
    print()
    print(f"  Next: uv run analyze_ppg.py data/sleep_ppg_{session_timestamp}.csv")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass