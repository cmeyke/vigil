"""
vigil — sleep staging with wav2sleep

Converts vigil PPG CSV → wav2sleep format, runs PPG-only inference,
and prints a hypnogram + sleep stage summary.

Usage:
    uv run sleep_staging.py data/20260711_231213/input/sleep_ppg_20260711_231213.csv

Output:
    data/20260711_231213/analysis/sleep_stages_20260711_231213.csv

Requires: wav2sleep installed in separate venv at ~/code/python/ai/wav2sleep-env
"""

import sys
import os
import subprocess
import tempfile
import shutil
from collections import Counter
import pandas as pd

# Labels: 0=Wake, 1=Light, 2=Deep, 3=REM
STAGE_LABELS = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
STAGE_EMOJI = {0: "🟡", 1: "🔵", 2: "🟣", 3: "🔴"}

# Path to wav2sleep venv (separate from vigil's venv due to numpy conflict)
WAV2SLEEP_PYTHON = os.path.expanduser("~/code/python/ai/wav2sleep-env/.venv/bin/python")


def convert_ppg(ppg_csv: str) -> str:
    """Convert vigil PPG CSV to wav2sleep format. Returns temp CSV path."""
    df = pd.read_csv(ppg_csv)
    ppg = df[["ch0", "ch1", "ch2", "ch3"]].mean(axis=1).values
    start_ns = df["timestamp_ns"].iloc[0]
    timestamps_sec = (df["timestamp_ns"].values - start_ns) / 1e9

    out = pd.DataFrame({"timestamp": timestamps_sec, "PPG": ppg})

    tmp = tempfile.NamedTemporaryFile(suffix="_wav2sleep.csv", delete=False, mode="w")
    out.to_csv(tmp.name, index=False)
    return tmp.name


def run_wav2sleep(ppg_csv: str, output_dir: str) -> str:
    """Run wav2sleep prediction. Returns predictions CSV path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert PPG to wav2sleep format
        wav2sleep_csv = convert_ppg(ppg_csv)

        # Set up input
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir, exist_ok=True)
        shutil.copy2(wav2sleep_csv, input_dir)

        # Auto-detect recording duration (avoid zero-padding → over-predicts Wake)
        df = pd.read_csv(wav2sleep_csv)
        duration_hours = df["timestamp"].iloc[-1] / 3600
        max_length_hours = int(duration_hours) + 1

        script = f"""
from wav2sleep import predict_on_folder
predict_on_folder(
    input_folder="{input_dir}",
    output_folder="{tmpdir}/output",
    model_folder="hf://joncarter/wav2sleep",
    signals=["PPG"],
    batch_size=1,
    max_length_hours={max_length_hours},
)
"""

        subprocess.run(
            [WAV2SLEEP_PYTHON, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        # Clean up temp converter file
        os.unlink(wav2sleep_csv)

        # Find predictions file
        preds_path = None
        for root, dirs, files in os.walk(os.path.join(tmpdir, "output")):
            for f in files:
                if f.endswith(".preds.csv"):
                    preds_path = os.path.join(root, f)
                    break
            if preds_path:
                break

        if not preds_path:
            raise RuntimeError("No predictions file found")

        # Copy to analysis/ directory with clean name
        os.makedirs(output_dir, exist_ok=True)
        timestamp = os.path.basename(ppg_csv).replace("sleep_ppg_", "").replace(".csv", "")
        final_path = os.path.join(output_dir, f"sleep_stages_{timestamp}.csv")
        shutil.copy2(preds_path, final_path)

    return final_path


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run sleep_staging.py <ppg_csv>")
        print("  Input:  data/<session>/input/sleep_ppg_<timestamp>.csv")
        print("  Output: data/<session>/analysis/sleep_stages_<timestamp>.csv")
        sys.exit(1)

    ppg_csv = sys.argv[1]
    if not os.path.exists(ppg_csv):
        print(f"Error: {ppg_csv} not found")
        sys.exit(1)

    # Output to analysis/ directory
    session_dir = os.path.dirname(os.path.dirname(ppg_csv))  # data/<timestamp>/
    analysis_dir = os.path.join(session_dir, "analysis")

    print(f"Running wav2sleep on {ppg_csv}...")
    preds_path = run_wav2sleep(ppg_csv, analysis_dir)

    # Read predictions
    import csv
    with open(preds_path) as f:
        reader = csv.DictReader(f)
        preds = [(float(row["Timestamp"]), int(row["Pred"])) for row in reader]

    # Summary
    stages = [p for _, p in preds]
    c = Counter(stages)
    total_hours = len(preds) * 0.5 / 60

    print()
    print("=" * 50)
    print("  SLEEP STAGING RESULTS (wav2sleep)")
    print("=" * 50)
    print(f"  Total recording: {total_hours:.1f} hours")
    print()

    for stage in sorted(c.keys()):
        count = c[stage]
        hours = count * 0.5 / 60
        pct = 100 * count / len(preds)
        label = STAGE_LABELS.get(stage, f"Stage {stage}")
        emoji = STAGE_EMOJI.get(stage, "❓")
        print(f"  {emoji} {label:6s}: {hours:.1f}h ({pct:.1f}%)")

    # Sleep efficiency
    sleep_epochs = sum(c[s] for s in [1, 2, 3])
    sleep_eff = 100 * sleep_epochs / len(preds)
    print()
    print(f"  Sleep efficiency: {sleep_eff:.1f}%")
    print(f"  Total sleep time: {sleep_epochs * 0.5 / 60:.1f}h")
    print("=" * 50)

    # Hourly hypnogram
    print()
    print("  Hypnogram (hourly):")
    for hour in range(int(preds[-1][0] / 3600) + 1):
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for ts, p in preds:
            h = ts / 3600
            if hour <= h < hour + 1:
                counts[p] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        bar = ""
        for s in [0, 1, 2, 3]:
            n = counts[s]
            if n > 0:
                bar += STAGE_EMOJI[s] * min(n, 20)
        print(f"  {hour:2d}h   {bar}")

    print()
    print(f"  Predictions saved: {preds_path}")


if __name__ == "__main__":
    main()