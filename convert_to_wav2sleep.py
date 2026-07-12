"""
vigil — convert raw PPG CSV to wav2sleep input format

Converts vigil's PPG CSV (timestamp_ns, ch0, ch1, ch2, ch3) to
wav2sleep's expected format (timestamp_seconds, PPG).

Usage:
    uv run convert_to_wav2sleep.py data/20260711_231213/sleep_ppg_20260711_231213.csv

Output:
    data/20260711_231213/sleep_ppg_wav2sleep.csv
"""

import sys
import os
import pandas as pd
import numpy as np


def convert(ppg_csv_path: str, output_path: str | None = None):
    """Convert vigil PPG CSV to wav2sleep format.

    vigil format:    timestamp_ns,ch0,ch1,ch2,ch3
    wav2sleep format: timestamp,PPG  (timestamp in seconds from start)
    """
    print(f"Loading {ppg_csv_path}...")

    df = pd.read_csv(ppg_csv_path)
    print(f"  {len(df)} samples, columns: {list(df.columns)}")

    # Average 4 PPG channels into single PPG signal
    # (wav2sleep expects a single PPG trace)
    ppg = df[["ch0", "ch1", "ch2", "ch3"]].mean(axis=1).values

    # Convert nanosecond timestamps to seconds from start
    start_ns = df["timestamp_ns"].iloc[0]
    timestamps_sec = (df["timestamp_ns"].values - start_ns) / 1e9

    # Build output dataframe
    out = pd.DataFrame({
        "timestamp": timestamps_sec,
        "PPG": ppg,
    })

    # Determine output path
    if output_path is None:
        base = os.path.dirname(ppg_csv_path)
        name = os.path.basename(ppg_csv_path).replace(".csv", "_wav2sleep.csv")
        output_path = os.path.join(base, name)

    out.to_csv(output_path, index=False)
    print(f"  Output: {output_path}")
    print(f"  Duration: {timestamps_sec[-1] / 3600:.1f} hours")
    print(f"  Samples: {len(out)}")
    print(f"  PPG range: {ppg.min():.0f} - {ppg.max():.0f}")
    print(f"  Ready for wav2sleep: predict_on_folder() or predict.py --signals PPG")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run convert_to_wav2sleep.py <ppg_csv> [output_csv]")
        sys.exit(1)

    ppg_csv = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    convert(ppg_csv, output)