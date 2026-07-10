"""
vigil — record 30 seconds of PPG + ACC and plot the PPG waveform

Run: uv run record_test.py

Records 30 seconds of raw data, saves to CSV, and plots the PPG signal
so you can visually verify your heartbeat is detectable.

Make sure the sensor is on your skin (forearm or upper arm).
"""

import asyncio
import csv
import time
from datetime import datetime
from bleak import BleakScanner
from polar_python import PolarDevice
from polar_python.models import PPGData, ACCData
import matplotlib.pyplot as plt
import numpy as np

SENSOR_ADDRESS = "24:AC:AC:1F:72:FF"
RECORD_SECONDS = 30
OUTPUT_DIR = "data"

ppg_all = []  # (timestamp_ns, ch0, ch1, ch2, ch3)
acc_all = []  # (timestamp_ns, x, y, z)


async def main():
    print(f"Looking for sensor at {SENSOR_ADDRESS}...")
    device = await BleakScanner.find_device_by_address(SENSOR_ADDRESS, timeout=10.0)
    if not device:
        print("❌ Sensor not found.")
        return

    print(f"✅ Found {device.name}")

    polar_device = PolarDevice(device)
    await polar_device.connect()
    print("Connected!\n")

    def ppg_callback(data: PPGData):
        for sample in data.samples:
            ppg_all.append((data.timestamp, sample[0], sample[1], sample[2], sample[3]))

    def acc_callback(data: ACCData):
        for sample in data.data:
            acc_all.append((data.timestamp, sample[0], sample[1], sample[2]))

    # Start streams
    await polar_device.start_ppg_stream(
        ppg_callback=ppg_callback, sample_rate=55, resolution=22, channels=4
    )
    await polar_device.start_acc_stream(
        acc_callback=acc_callback, sample_rate=52, resolution=16, range=8, channels=3
    )
    print(f"Streaming PPG (55 Hz) + ACC (52 Hz) for {RECORD_SECONDS} seconds...")
    print("Keep the sensor still on your skin.\n")

    await asyncio.sleep(RECORD_SECONDS)

    await polar_device.stop_ppg_stream()
    await polar_device.stop_acc_stream()
    await polar_device.disconnect()

    # Save raw data
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ppg_file = f"{OUTPUT_DIR}/ppg_{timestamp}.csv"
    with open(ppg_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "ch0", "ch1", "ch2", "ch3"])
        w.writerows(ppg_all)
    print(f"\nPPG data saved: {ppg_file} ({len(ppg_all)} samples)")

    acc_file = f"{OUTPUT_DIR}/acc_{timestamp}.csv"
    with open(acc_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "x_mg", "y_mg", "z_mg"])
        w.writerows(acc_all)
    print(f"ACC data saved: {acc_file} ({len(acc_all)} samples)")

    # Plot PPG — channel 0 (typically green LED, best for HR detection)
    if len(ppg_all) > 100:
        ppg_arr = np.array(ppg_all)
        t = np.arange(len(ppg_arr)) / 55.0  # seconds (55 Hz sample rate)
        ch0 = ppg_arr[:, 1].astype(float)
        ch1 = ppg_arr[:, 2].astype(float)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Full PPG signal — channel 0
        axes[0].plot(t, ch0, linewidth=0.5, color="green")
        axes[0].set_title(f"PPG Channel 0 (green LED) — {len(ppg_all)} samples @ 55 Hz")
        axes[0].set_ylabel("Raw ADC value")
        axes[0].set_xlabel("Time (seconds)")

        # Zoomed in — first 5 seconds (should show heartbeat pulses)
        mask5 = t <= 5.0
        axes[1].plot(t[mask5], ch0[mask5], linewidth=1.0, color="green")
        axes[1].set_title("PPG Channel 0 — first 5 seconds (zoomed)")
        axes[1].set_ylabel("Raw ADC value")
        axes[1].set_xlabel("Time (seconds)")

        # Accelerometer magnitude
        if len(acc_all) > 10:
            acc_arr = np.array(acc_all)
            t_acc = np.arange(len(acc_arr)) / 52.0
            acc_mag = np.sqrt(
                acc_arr[:, 1].astype(float) ** 2
                + acc_arr[:, 2].astype(float) ** 2
                + acc_arr[:, 3].astype(float) ** 2
            )
            axes[2].plot(t_acc, acc_mag, linewidth=0.5, color="blue")
            axes[2].set_title(f"Accelerometer magnitude — {len(acc_all)} samples @ 52 Hz")
            axes[2].set_ylabel("Magnitude (mG)")
            axes[2].set_xlabel("Time (seconds)")
            axes[2].axhline(y=1000, color="r", linestyle="--", alpha=0.5, label="1G (gravity)")
            axes[2].legend()

        plt.tight_layout()
        plot_file = f"{OUTPUT_DIR}/ppg_plot_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved: {plot_file}")

        # Quick stats
        print(f"\n{'='*50}")
        print(f"PPG channel 0 stats:")
        print(f"  Samples:    {len(ppg_all)}")
        print(f"  Duration:   {len(ppg_all)/55.0:.1f} seconds")
        print(f"  Min:        {ch0.min():.0f}")
        print(f"  Max:        {ch0.max():.0f}")
        print(f"  Mean:       {ch0.mean():.0f}")
        print(f"  Std:        {ch0.std():.0f}")
        print(f"  Range:      {ch0.max() - ch0.min():.0f}")
        # If std is very low, sensor may not be on skin properly
        if ch0.std() < 100:
            print(f"\n⚠️  Low signal variation — check sensor contact with skin")
        else:
            print(f"\n✅ Good signal variation — heartbeat should be visible in the plot")
        print(f"{'='*50}")
    else:
        print("⚠️ Not enough PPG data to plot.")


if __name__ == "__main__":
    asyncio.run(main())