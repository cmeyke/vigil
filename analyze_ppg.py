"""
vigil — PPG signal processing: filter, detect heartbeats, compute HR + HRV

Run: uv run analyze_ppg.py

Loads the most recent PPG recording from data/, applies band-pass filtering,
detects heartbeat pulses, computes instant heart rate and HRV metrics,
and plots the results.

Usage:
  uv run analyze_ppg.py                    # auto-finds latest recording
  uv run analyze_ppg.py data/ppg_xxx.csv   # specify a file
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# PPG sample rate (Verity Sense)
FS = 55  # Hz

# Cardiac band-pass filter: 0.7 Hz (42 BPM) to 4.0 Hz (240 BPM)
LOWCUT = 0.7
HIGHCUT = 4.0


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """Butterworth band-pass filter, zero-phase (filtfilt)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_peaks(filtered, fs, min_distance=0.4, motion_mask=None):
    """Detect heartbeat peaks in filtered PPG signal with sub-sample interpolation.

    min_distance: minimum seconds between peaks (avoids double-counting)
    motion_mask: boolean array, True = motion artifact (skip peaks there)
    Returns peak indices (integer) and refined peak positions (float, in samples).
    """
    signal = filtered.copy()

    # Zero out motion segments so peaks aren't detected there
    if motion_mask is not None:
        signal[motion_mask] = 0

    # Adaptive threshold: mean + 0.5 * std
    threshold = np.mean(signal) + 0.5 * np.std(signal)
    min_distance_samples = int(min_distance * fs)

    peaks, properties = find_peaks(
        signal,
        height=threshold,
        distance=min_distance_samples,
        prominence=0.3 * np.std(signal),
    )

    # Parabolic interpolation for sub-sample peak precision
    # Fits a parabola to the 3 points around each peak and finds the vertex
    refined = peaks.astype(float)
    for i, p in enumerate(peaks):
        if p == 0 or p == len(signal) - 1:
            continue
        y0, y1, y2 = signal[p - 1], signal[p], signal[p + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            refined[i] = p + delta  # delta ∈ [-0.5, +0.5]

    return peaks, refined


def filter_rr_outliers(rr_intervals_ms, max_change_pct=20):
    """Remove ectopic/artifact RR intervals.

    Rejects any RR interval that changes more than max_change_pct from
    the running median. This removes false peaks from motion artifacts
    that create impossibly short or long intervals.
    """
    if len(rr_intervals_ms) < 3:
        return rr_intervals_ms, np.array([], dtype=int)

    rr = np.array(rr_intervals_ms)
    keep = np.ones(len(rr), dtype=bool)
    window = 5  # running median window

    for i in range(len(rr)):
        # Running median of surrounding intervals
        lo = max(0, i - window)
        hi = min(len(rr), i + window + 1)
        neighbors = np.concatenate([rr[lo:i], rr[i+1:hi]])
        if len(neighbors) == 0:
            continue
        running_median = np.median(neighbors)
        # Reject if this interval deviates too much
        if running_median > 0:
            change_pct = abs(rr[i] - running_median) / running_median * 100
            if change_pct > max_change_pct:
                keep[i] = False

    return rr[keep], np.where(~keep)[0]


def compute_hrv(rr_intervals_ms):
    """Compute standard HRV time-domain metrics from RR intervals (ms)."""
    if len(rr_intervals_ms) < 2:
        return {"error": "Not enough RR intervals for HRV"}

    rr = np.array(rr_intervals_ms)
    diff_rr = np.diff(rr)

    metrics = {
        "n_beats": len(rr),
        "mean_rr_ms": np.mean(rr),
        "sdnn_ms": np.std(rr, ddof=1),  # Standard deviation of RR intervals
        "mean_hr_bpm": 60000 / np.mean(rr),
        "rmssd_ms": np.sqrt(np.mean(diff_rr ** 2)),  # Root mean square of successive differences
        "pnn50_pct": 100 * np.mean(np.abs(diff_rr) > 50),  # % of successive diffs > 50ms
        "min_rr_ms": np.min(rr),
        "max_rr_ms": np.max(rr),
        "range_rr_ms": np.max(rr) - np.min(rr),
    }
    return metrics


def load_latest_ppg(data_dir="data"):
    """Find the most recent PPG CSV file in data/*/input/."""
    files = sorted(glob.glob(f"{data_dir}/*/input/sleep_ppg_*.csv"))
    if not files:
        # Fallback: check data/ directly (old structure)
        files = sorted(glob.glob(f"{data_dir}/*ppg_*.csv"))
    if not files:
        print(f"❌ No PPG files found in {data_dir}/*/input/")
        sys.exit(1)
    return files[-1]


def main():
    # Get input file
    if len(sys.argv) > 1:
        ppg_file = sys.argv[1]
    else:
        ppg_file = load_latest_ppg()
        print(f"Auto-selected latest: {ppg_file}")

    if not os.path.exists(ppg_file):
        print(f"❌ File not found: {ppg_file}")
        sys.exit(1)

    # Load data
    df = pd.read_csv(ppg_file)
    print(f"Loaded {len(df)} samples ({len(df)/FS:.1f}s at {FS} Hz)\n")

    # Use channel 0 (green LED — best for pulse detection)
    raw = df["ch0"].values.astype(float)
    t = np.arange(len(raw)) / FS

    # Also load ACC if available for motion artifact flagging
    # ACC is in the same input/ directory as PPG
    acc_file = ppg_file.replace("ppg_", "acc_")
    if not os.path.exists(acc_file):
        acc_file = os.path.join(os.path.dirname(ppg_file), ppg_file.replace("ppg_", "acc_").replace(os.path.basename(ppg_file), ""))
        acc_file = os.path.join(os.path.dirname(ppg_file), os.path.basename(ppg_file).replace("ppg_", "acc_"))
    acc_mag = None
    if os.path.exists(acc_file):
        df_acc = pd.read_csv(acc_file)
        acc_mag = np.sqrt(
            df_acc["x_mg"].values.astype(float) ** 2
            + df_acc["y_mg"].values.astype(float) ** 2
            + df_acc["z_mg"].values.astype(float) ** 2
        )
        print(f"Accelerometer loaded: {len(acc_mag)} samples")

    # ─── Filter ───
    print(f"\nFiltering: band-pass {LOWCUT}-{HIGHCUT} Hz (order 4, zero-phase)...")
    filtered = bandpass_filter(raw, FS, LOWCUT, HIGHCUT)

    # ─── Build motion mask ───
    motion_mask = None
    motion_segments = []
    if acc_mag is not None:
        acc_resampled = np.interp(t, np.arange(len(acc_mag)) / 52.0, acc_mag)
        motion_mask = np.abs(acc_resampled - 1000) > 150
        motion_changes = np.diff(motion_mask.astype(int))
        motion_starts = np.where(motion_changes == 1)[0]
        motion_ends = np.where(motion_changes == -1)[0]
        for s, e in zip(motion_starts, motion_ends):
            motion_segments.append((s / FS, e / FS))
        motion_pct = 100 * np.mean(motion_mask)
        print(f"Motion detected in {motion_pct:.1f}% of recording ({len(motion_segments)} segments)")

    # ─── Detect peaks (with motion masking) ───
    print("Detecting heartbeat peaks (motion-aware)...")
    peaks, refined = detect_peaks(filtered, FS, min_distance=0.4, motion_mask=motion_mask)
    peak_times = refined / FS  # use interpolated positions for sub-sample precision
    print(f"Found {len(peaks)} peaks in {len(raw)/FS:.1f}s")

    if len(peaks) < 3:
        print("⚠️  Too few peaks detected. Signal may be too noisy.")
        return

    # ─── Compute RR intervals ───
    rr_all = np.diff(peak_times) * 1000  # ms

    # ─── Filter RR outliers ───
    rr_clean, rejected_idx = filter_rr_outliers(rr_all, max_change_pct=20)
    print(f"RR intervals: {len(rr_all)} total → {len(rr_clean)} after outlier rejection ({len(rejected_idx)} removed)")

    if len(rr_clean) < 3:
        print("⚠️  Too few clean RR intervals for HRV.")
        return

    instant_hr_clean = 60000 / rr_clean
    # For plotting, also compute the full (uncleaned) HR
    instant_hr_all = 60000 / rr_all

    # ─── HRV metrics (from clean RR only) ───
    hrv = compute_hrv(rr_clean)

    # ─── Print results ───
    print(f"\n{'='*55}")
    print(f"  HEART RATE & HRV RESULTS")
    print(f"{'='*55}")
    print(f"  Recording duration:   {len(raw)/FS:.1f} seconds")
    print(f"  Heartbeats detected:  {hrv['n_beats']}")
    print(f"  Mean heart rate:      {hrv['mean_hr_bpm']:.1f} BPM")
    print(f"  Mean RR interval:     {hrv['mean_rr_ms']:.0f} ms")
    print(f"  RR range:             {hrv['min_rr_ms']:.0f}–{hrv['max_rr_ms']:.0f} ms")
    print(f"")
    print(f"  ── HRV Time Domain ──")
    print(f"  SDNN:                 {hrv['sdnn_ms']:.1f} ms")
    print(f"  RMSSD:                {hrv['rmssd_ms']:.1f} ms")
    print(f"  pNN50:                {hrv['pnn50_pct']:.1f} %")
    print(f"{'='*55}")

    # ─── Plot ───
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # 1. Raw vs filtered (full recording)
    axes[0].plot(t, raw, linewidth=0.3, color="gray", alpha=0.5, label="Raw PPG")
    axes[0].plot(t, filtered, linewidth=0.8, color="green", label="Filtered (0.7–4 Hz)")
    axes[0].set_title(f"Raw vs Filtered PPG — {len(raw)} samples @ {FS} Hz")
    axes[0].set_ylabel("ADC value")
    axes[0].legend(loc="upper right")
    # Mark motion segments
    for s, e in motion_segments:
        axes[0].axvspan(s, e, color="red", alpha=0.15, label="motion" if s == motion_segments[0][0] else "")

    # 2. Filtered signal with detected peaks (first 10 seconds)
    mask10 = t <= 10.0
    axes[1].plot(t[mask10], filtered[mask10], linewidth=1.0, color="green")
    peaks10_mask = peak_times <= 10.0
    axes[1].plot(peak_times[peaks10_mask], filtered[peaks[peaks10_mask]], "ro", markersize=6, label="Detected beats")
    axes[1].set_title("Filtered PPG with detected heartbeats — first 10 seconds")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend()

    # 3. Instantaneous heart rate (clean + raw for comparison)
    axes[2].plot(peak_times[1:], instant_hr_all, "lightgray", marker="o", markersize=2, linewidth=0.5, label="Raw (all beats)")
    # Clean HR — need to map back to peak times
    clean_peak_times = np.delete(peak_times[1:], rejected_idx)
    axes[2].plot(clean_peak_times, instant_hr_clean, "b-o", markersize=4, linewidth=1, label="Clean (after rejection)")
    axes[2].axhline(y=hrv["mean_hr_bpm"], color="r", linestyle="--", alpha=0.5, label=f"Mean: {hrv['mean_hr_bpm']:.1f} BPM")
    axes[2].set_title("Instantaneous Heart Rate (per beat)")
    axes[2].set_ylabel("BPM")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()

    # 4. RR intervals (tachogram) — clean only
    axes[3].plot(np.arange(1, len(rr_all) + 1), rr_all, "lightgray", marker="o", markersize=2, linewidth=0.5, label="Raw RR")
    axes[3].plot(np.arange(1, len(rr_clean) + 1), rr_clean, "mo-", markersize=4, linewidth=1, label="Clean RR")
    axes[3].axhline(y=hrv["mean_rr_ms"], color="r", linestyle="--", alpha=0.5, label=f"Mean RR: {hrv['mean_rr_ms']:.0f} ms")
    axes[3].set_title(f"RR Intervals — SDNN: {hrv['sdnn_ms']:.1f}ms, RMSSD: {hrv['rmssd_ms']:.1f}ms ({len(rejected_idx)} rejected)")
    axes[3].set_ylabel("RR interval (ms)")
    axes[3].set_xlabel("Beat number")
    axes[3].legend()

    plt.tight_layout()

    # Save plot and RR intervals to analysis/ directory
    session_dir = os.path.dirname(os.path.dirname(ppg_file))  # data/<timestamp>/
    analysis_dir = os.path.join(session_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    timestamp = os.path.basename(ppg_file).replace("sleep_ppg_", "").replace(".csv", "")

    output_file = os.path.join(analysis_dir, f"sleep_analysis_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved: {output_file}")

    # Save RR intervals
    rr_file = os.path.join(analysis_dir, f"sleep_rr_{timestamp}.csv")
    pd.DataFrame({
        "beat": np.arange(1, len(rr_clean) + 1),
        "rr_ms": rr_clean,
        "hr_bpm": instant_hr_clean,
    }).to_csv(rr_file, index=False)
    print(f"RR intervals saved: {rr_file}")


if __name__ == "__main__":
    main()