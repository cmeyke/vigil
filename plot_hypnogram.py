"""
vigil — hypnogram plot (Pixel Watch / Fitbit style)

Generates a single-line stepped hypnogram from wav2sleep predictions.
Dark background, colored segments, clock time on x-axis.

Usage:
    uv run plot_hypnogram.py data/20260711_231213/analysis/sleep_stages_20260711_231213.csv
"""

import sys
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from matplotlib.patches import Patch

# Sleep stage colors (distinct, color-blind friendly)
STAGE_COLORS = {
    0: "#FF3B6B",   # Awake — pink/magenta
    1: "#4A9EFF",   # Light — blue
    2: "#7B3FE4",   # Deep — purple
    3: "#FF9F45",   # REM — orange
}
STAGE_NAMES = {0: "Awake", 1: "Light", 2: "Deep", 3: "REM"}

# Y positions: Deep at bottom, Awake at top (standard hypnogram convention)
STAGE_Y = {0: 3, 3: 2, 1: 1, 2: 0}  # Wake, REM, Light, Deep
EPOCH_SECONDS = 30


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run plot_hypnogram.py <sleep_stages_*.csv>")
        sys.exit(1)

    stages_csv = sys.argv[1]
    if not os.path.exists(stages_csv):
        print(f"Error: {stages_csv} not found")
        sys.exit(1)

    # Read predictions
    with open(stages_csv) as f:
        reader = csv.DictReader(f)
        preds = [(float(row["Timestamp"]), int(row["Pred"])) for row in reader]

    if not preds:
        print("No predictions found")
        sys.exit(1)

    # Parse start time from filename
    basename = os.path.basename(stages_csv)
    try:
        ts_str = basename.replace("sleep_stages_", "").replace(".csv", "")
        start_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except ValueError:
        start_dt = datetime(2026, 1, 1, 0, 0)

    # Summary — trim leading/trailing Awake epochs from stats
    # (they're still drawn in the graph, just not counted as sleep)
    stages = [p[1] for p in preds]
    first_sleep = next((i for i, s in enumerate(stages) if s != 0), 0)
    last_sleep = next((i for i, s in enumerate(reversed(stages)) if s != 0), 0)
    last_sleep = len(stages) - last_sleep  # convert from reversed index
    trimmed = stages[first_sleep:last_sleep]
    c = Counter(trimmed)
    sleep_epochs = sum(c[s] for s in [1, 2, 3])
    total_h = len(trimmed) * EPOCH_SECONDS / 3600  # time in bed (onset to final awakening)
    sleep_eff = 100 * sleep_epochs / len(trimmed) if trimmed else 0

    # --- Smooth: merge consecutive same-stage epochs into segments ---
    segments = []  # (start_sec, end_sec, stage)
    seg_start = preds[0][0]
    seg_stage = preds[0][1]

    for ts, stage in preds[1:]:
        if stage != seg_stage:
            segments.append((seg_start, ts, seg_stage))
            seg_start = ts
            seg_stage = stage
    segments.append((seg_start, preds[-1][0] + EPOCH_SECONDS, seg_stage))

    # Create figure — dark background
    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw each segment as a colored horizontal line
    for start, end, stage in segments:
        y = STAGE_Y[stage]
        color = STAGE_COLORS[stage]

        # Thick colored line segment
        ax.plot(
            [start, end], [y, y],
            color=color, linewidth=5, solid_capstyle="butt",
        )

    # Y-axis: stage names
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(
        ["Deep", "Light", "REM", "Awake"],
        color="#8888aa", fontsize=11,
    )

    # X-axis: clock time
    total_secs = preds[-1][0] + EPOCH_SECONDS
    tick_positions = []
    tick_labels = []
    for h in range(0, int(total_secs / 3600) + 2):
        tick_time = start_dt + timedelta(seconds=h * 3600)
        tick_positions.append(h * 3600)
        tick_labels.append(tick_time.strftime("%H:%M"))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, color="#8888aa", fontsize=10)

    # Subtle horizontal gridlines
    for y in [0, 1, 2, 3]:
        ax.axhline(y=y, color="#2a2a4e", linewidth=0.5, alpha=0.3)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title
    ax.set_title(
        f"Sleep  ·  {total_h:.1f}h  ·  {sleep_eff:.0f}% efficiency",
        color="white", fontsize=14, pad=15, loc="left",
        fontweight="bold",
    )

    # Legend — below the title to avoid occlusion
    legend_patches = []
    for stage in [2, 1, 3, 0]:  # Deep, Light, REM, Awake
        count = c.get(stage, 0)
        hours = count * EPOCH_SECONDS / 3600
        pct = 100 * count / len(trimmed) if trimmed else 0
        legend_patches.append(Patch(
            color=STAGE_COLORS[stage],
            label=f"{STAGE_NAMES[stage]}  {hours:.1f}h ({pct:.0f}%)",
        ))

    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=4,
        facecolor="#2a2a4e",
        edgecolor="#3a3a5e",
        labelcolor="white",
        fontsize=10,
        framealpha=0.9,
    )

    ax.set_xlim(-60, total_secs + 60)
    ax.set_ylim(-0.2, 3.3)

    plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.22)

    output_path = stages_csv.replace("sleep_stages_", "hypnogram_").replace(".csv", ".png")
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Hypnogram saved: {output_path}")


if __name__ == "__main__":
    main()