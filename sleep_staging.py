"""
vigil — sleep staging with wav2sleep

Runs wav2sleep PPG-only inference on vigil PPG data and produces
a hypnogram (sleep stage timeline).

Usage:
    uv run sleep_staging.py data/20260711_231213/sleep_ppg_20260711_231213_wav2sleep.csv

Requires: wav2sleep installed in separate venv at ~/code/python/ai/wav2sleep-env
"""

import sys
import os
import subprocess
import tempfile
from collections import Counter

# Labels: 0=Wake, 1=Light, 2=Deep, 3=REM
STAGE_LABELS = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
STAGE_EMOJI = {0: "🟡", 1: "🔵", 2: "🟣", 3: "🔴"}

# Path to wav2sleep venv (separate from vigil's venv due to numpy conflict)
WAV2SLEEP_PYTHON = os.path.expanduser("~/code/python/ai/wav2sleep-env/.venv/bin/python")


def run_wav2sleep(ppg_csv: str, output_dir: str) -> str:
    """Run wav2sleep prediction on a PPG CSV file. Returns predictions CSV path."""
    import shutil
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.join(output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    shutil.copy2(ppg_csv, input_dir)

    script = f"""
from wav2sleep import predict_on_folder
predict_on_folder(
    input_folder="{input_dir}",
    output_folder="{output_dir}",
    model_folder="hf://joncarter/wav2sleep",
    signals=["PPG"],
    batch_size=1,
    max_length_hours=10,
)
"""

    result = subprocess.run(
        [WAV2SLEEP_PYTHON, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    # Find predictions file
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".preds.csv"):
                return os.path.join(root, f)

    raise RuntimeError("No predictions file found")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run sleep_staging.py <ppg_wav2sleep.csv>")
        print("  Input: CSV from convert_to_wav2sleep.py")
        print("  Output: hypnogram + summary to stdout")
        sys.exit(1)

    ppg_csv = sys.argv[1]
    if not os.path.exists(ppg_csv):
        print(f"Error: {ppg_csv} not found")
        sys.exit(1)

    print(f"Running wav2sleep on {ppg_csv}...")
    output_dir = os.path.join(os.path.dirname(ppg_csv), "wav2sleep_output")
    preds_path = run_wav2sleep(ppg_csv, output_dir)

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

    # Hourly timeline
    print()
    print("  Hypnogram (hourly):")
    print("  Hour  W  L  D  R")
    for hour in range(int(preds[-1][0] / 3600) + 1):
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for ts, p in preds:
            h = ts / 3600
            if hour <= h < hour + 1:
                counts[p] += 1
        # Show dominant stage
        dominant = max(counts, key=counts.get)
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