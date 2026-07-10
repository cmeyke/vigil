"""
vigil — live HR + HRV terminal display

Run: uv run python live_hr.py

Connects to the Verity Sense, streams raw PPG, filters and detects
heartbeats in real-time, and displays HR + HRV metrics to the terminal.

Ctrl+C to stop.
"""

import asyncio
import time
from collections import deque
from bleak import BleakScanner
from polar_python import PolarDevice
from polar_python.models import PPGData, ACCData
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np

SENSOR_ADDRESS = "24:AC:AC:1F:72:FF"
FS = 55  # PPG sample rate
LOWCUT = 0.7
HIGHCUT = 4.0
BUFFER_SECONDS = 15  # rolling buffer length
DISPLAY_INTERVAL = 1.0  # update terminal every N seconds
HRV_WINDOW = 20  # beats for rolling HRV

# Rolling buffers
ppg_buffer = deque(maxlen=BUFFER_SECONDS * FS)
acc_buffer = deque(maxlen=BUFFER_SECONDS * 52)
peak_times = deque(maxlen=HRV_WINDOW + 5)  # timestamps of detected peaks

# State
start_time = time.monotonic()
last_display = 0.0


def bandpass(signal, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def fmt_bars(val, vmin, vmax, width=20):
    """Simple horizontal bar."""
    pct = max(0, min(1, (val - vmin) / (vmax - vmin)))
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)


def sparkline(values, width=30):
    """Simple text sparkline from a list of values."""
    if len(values) < 2:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return chars[3] * min(len(values), width)
    scaled = [(v - vmin) / (vmax - vmin) * (len(chars) - 1) for v in values]
    # Sample to width
    step = max(1, len(scaled) // width)
    sampled = scaled[::step][-width:]
    return "".join(chars[int(s)] for s in sampled)


async def main():
    global start_time, last_display

    print(f"Looking for sensor at {SENSOR_ADDRESS}...")
    device = await BleakScanner.find_device_by_address(SENSOR_ADDRESS, timeout=10.0)
    if not device:
        print("❌ Sensor not found. Turn it on and make sure Flow app is closed.")
        return

    print(f"✅ Found {device.name}")
    polar_device = PolarDevice(device)
    await polar_device.connect()
    print("Connected! Starting PPG stream...\n")

    def ppg_callback(data: PPGData):
        for sample in data.samples:
            ppg_buffer.append(sample[0])  # channel 0

    def acc_callback(data: ACCData):
        for sample in data.data:
            acc_buffer.append(
                (sample[0] ** 2 + sample[1] ** 2 + sample[2] ** 2) ** 0.5
            )

    await polar_device.start_ppg_stream(
        ppg_callback=ppg_callback, sample_rate=55, resolution=22, channels=4
    )
    await polar_device.start_acc_stream(
        acc_callback=acc_callback, sample_rate=52, resolution=16, range=8, channels=3
    )

    print("Streaming. Press Ctrl+C to stop.\n")
    start_time = time.monotonic()

    try:
        while True:
            await asyncio.sleep(DISPLAY_INTERVAL)

            if len(ppg_buffer) < FS * 5:
                # Not enough data yet
                elapsed = time.monotonic() - start_time
                print(f"\r  Collecting data... {len(ppg_buffer)}/{FS*5} samples "
                      f"({elapsed:.0f}s)", end="", flush=True)
                continue

            # Process buffer
            raw = np.array(ppg_buffer)
            filtered = bandpass(raw, FS, LOWCUT, HIGHCUT)

            # Motion mask from accelerometer
            motion_mask = None
            if len(acc_buffer) > 10:
                acc_arr = np.array(acc_buffer)
                # Resample ACC to PPG length
                t_acc = np.arange(len(acc_arr)) / 52.0
                t_ppg = np.arange(len(raw)) / FS
                acc_resampled = np.interp(t_ppg, t_acc, acc_arr)
                motion_mask = np.abs(acc_resampled - 1000) > 150
                filtered[motion_mask] = 0

            # Detect peaks
            threshold = np.mean(filtered) + 0.5 * np.std(filtered)
            peaks, _ = find_peaks(
                filtered, height=threshold, distance=int(0.4 * FS),
                prominence=0.3 * np.std(filtered),
            )

            # Convert peak indices to absolute timestamps
            buf_start = time.monotonic() - len(raw) / FS
            new_peak_times = [buf_start + p / FS for p in peaks]

            # Merge with previously detected peaks (avoid duplicates)
            if peak_times:
                last_known = peak_times[-1]
                for pt in new_peak_times:
                    if pt > last_known + 0.3:  # at least 0.3s after last known
                        peak_times.append(pt)
            else:
                peak_times.extend(new_peak_times)

            # Compute RR intervals from recent peaks
            recent = list(peak_times)[-HRV_WINDOW:]
            if len(recent) < 2:
                continue

            rr_ms = np.diff(recent) * 1000

            # Reject RR outliers (>20% deviation from median)
            if len(rr_ms) >= 3:
                med = np.median(rr_ms)
                clean_mask = np.abs(rr_ms - med) / med < 0.20
                rr_clean = rr_ms[clean_mask]
            else:
                rr_clean = rr_ms

            if len(rr_clean) < 2:
                continue

            # Metrics
            hr = 60000 / np.mean(rr_clean)
            hr_now = 60000 / rr_clean[-1]
            rmssd = np.sqrt(np.mean(np.diff(rr_clean) ** 2)) if len(rr_clean) >= 3 else 0
            sdnn = np.std(rr_clean, ddof=1) if len(rr_clean) >= 3 else 0
            n_beats = len(peak_times)
            elapsed = time.monotonic() - start_time

            # HR sparkline (last 30 clean RR intervals)
            hr_history = [60000 / r for r in rr_clean]
            spark = sparkline(hr_history)

            # Terminal display
            print("\033[2J\033[H", end="")  # clear screen + cursor home
            print(f"  ╔══════════════════════════════════════════╗")
            print(f"  ║  vigil — live HR + HRV                   ║")
            print(f"  ╚══════════════════════════════════════════╝")
            print()
            print(f"  Duration:    {elapsed:.0f}s    Beats: {n_beats}")
            print()
            print(f"  Heart Rate (now):   {hr_now:.0f} BPM")
            print(f"  Heart Rate (avg):   {hr:.0f} BPM   {fmt_bars(hr, 40, 120)}")
            print(f"  Last RR interval:   {rr_clean[-1]:.0f} ms")
            print()
            print(f"  HRV (last {len(rr_clean)} beats):")
            print(f"    RMSSD:  {rmssd:.1f} ms   {fmt_bars(rmssd, 0, 150)}")
            print(f"    SDNN:   {sdnn:.1f} ms   {fmt_bars(sdnn, 0, 150)}")
            print()
            print(f"  HR trend:  {spark}")
            print()
            motion_pct = 100 * np.mean(motion_mask) if motion_mask is not None else 0
            print(f"  Signal:    {len(ppg_buffer)} samples in buffer"
                  f"  |  Motion: {motion_pct:.0f}%"
                  + ("  ⚠️" if motion_pct > 10 else "  ✅"))
            print()
            print(f"  Ctrl+C to stop")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        await polar_device.stop_ppg_stream()
        await polar_device.stop_acc_stream()
        await polar_device.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())