"""
vigil — compare wav2sleep predictions against Google sleep stages

Aligns a vigil recording's sleep_stages CSV with the overlapping Google
Health sleep session(s) and prints a confusion matrix, per-stage
precision/recall, overall agreement, and Cohen's kappa. Saves a
per-epoch comparison CSV and a results TXT to the session's analysis/
directory.

The vigil recording start time is taken from the PPG file's first
timestamp_ns (Unix epoch nanoseconds), so correct timestamp alignment
is required — run import-android.py first if recordings were made with
an older app version.

Usage:
    uv run compare_google.py                                 # latest session
    uv run compare_google.py data/20260711_231213/analysis/sleep_stages_20260711_231213.csv
    uv run compare_google.py --all                           # all sessions
    uv run compare_google.py --force                         # re-run even if outputs exist

Output:
    data/<timestamp>/analysis/compare_google_<timestamp>.csv   (per-epoch labels)
    data/<timestamp>/analysis/compare_google_<timestamp>.txt   (confusion + metrics)
"""

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
GOOGLE_DIR = os.path.join(DATA_DIR, "google_sleep")

STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]
EPOCH_SECONDS = 30


def parse_session_timestamp(session_dir):
    """Extract the vigil session timestamp from its directory name."""
    return os.path.basename(session_dir)


def load_google_data():
    """Load and merge all Google sleep CSVs in data/google_sleep/.

    Returns a DataFrame sorted by StartTime with columns
    StartTime, EndTime (UTC tz-aware), StageLabel (0-3).
    """
    files = sorted(glob.glob(os.path.join(GOOGLE_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No Google sleep CSVs found in {GOOGLE_DIR}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True)
        df["EndTime"] = pd.to_datetime(df["EndTime"], utc=True)
        frames.append(df[["StartTime", "EndTime", "StageLabel"]])
    g = pd.concat(frames, ignore_index=True)
    g = g.drop_duplicates(["StartTime", "EndTime", "StageLabel"])
    g = g.sort_values("StartTime").reset_index(drop=True)
    return g


def find_google_overlap(google_df, start_dt, end_dt):
    """Return Google rows whose interval overlaps [start_dt, end_dt]."""
    mask = (google_df["StartTime"] <= end_dt) & (google_df["EndTime"] >= start_dt)
    return google_df[mask].reset_index(drop=True)


def assign_google_labels(google_overlap, epoch_dt):
    """For each epoch timestamp, find the Google stage containing it.

    Returns int array with -1 where no Google interval covers the epoch.
    """
    # Use searchsorted on StartTime for a fast left-bound lookup, then verify
    # the epoch falls within [StartTime, EndTime) of that row.
    starts = google_overlap["StartTime"].values
    ends = google_overlap["EndTime"].values
    labels = google_overlap["StageLabel"].values

    # searchsorted: for each epoch, find the last interval with StartTime <= epoch
    # i.e. insertion point - 1
    epoch_np = pd.DatetimeIndex(epoch_dt).asi8  # int64 ns since epoch
    start_np = pd.DatetimeIndex(starts).asi8
    end_np = pd.DatetimeIndex(ends).asi8

    idx = np.searchsorted(start_np, epoch_np, side="right") - 1
    out = np.full(len(epoch_dt), -1, dtype=int)
    valid = (idx >= 0) & (idx < len(google_overlap))
    valid_idx = idx[valid]
    valid_epoch = epoch_np[valid]
    contained = (valid_epoch >= start_np[valid_idx]) & (valid_epoch < end_np[valid_idx])
    valid_pos = np.where(valid)[0][contained]
    out[valid_pos] = labels[valid_idx[contained]]
    return out


def confusion_matrix(true, pred, num_classes=4):
    """Build a confusion matrix: rows=true, cols=pred."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def cohen_kappa(cm):
    """Cohen's kappa from a confusion matrix."""
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    row_marginal = cm.sum(axis=1) / n
    col_marginal = cm.sum(axis=0) / n
    pe = np.sum(row_marginal * col_marginal)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def format_report(true, pred, cm):
    """Build the text report as a single string."""
    lines = []
    lines.append("=" * 60)
    lines.append("  wav2sleep vs Google — comparison")
    lines.append("=" * 60)

    n = (true >= 0).sum()
    if n == 0:
        lines.append("  No overlapping epochs with Google labels.")
        lines.append("=" * 60)
        return "\n".join(lines)

    agreement = 100 * (true == pred).mean()
    kappa = cohen_kappa(cm)

    lines.append(f"  Epochs compared:    {n}")
    lines.append(f"  Overall agreement: {agreement:.1f}%")
    lines.append(f"  Cohen's kappa:     {kappa:.3f}")
    lines.append("")

    # Confusion matrix header
    lines.append("  Confusion (rows=Google, cols=wav2sleep):")
    header = f"  {'':12s} " + " ".join(f"{n:>7s}" for n in STAGE_NAMES)
    lines.append(header)
    for gt in range(4):
        row = [cm[gt, k] for k in range(4)]
        lines.append(f"  G={STAGE_NAMES[gt]:10s} " + " ".join(f"{v:7d}" for v in row))

    # Per-stage precision/recall/F1
    lines.append("")
    lines.append("  Per-stage metrics (Google = ground truth):")
    lines.append(f"  {'Stage':8s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'N':>5s}")
    for i, name in enumerate(STAGE_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        lines.append(
            f"  {name:8s}  {prec:6.2f}  {rec:6.2f}  {f1:6.2f}  {int(cm[i,:].sum()):5d}"
        )

    # Stage distribution comparison
    lines.append("")
    lines.append("  Stage distribution (Google vs wav2sleep, % of epochs):")
    for i, name in enumerate(STAGE_NAMES):
        g_pct = 100 * cm[i, :].sum() / n
        w_pct = 100 * cm[:, i].sum() / n
        lines.append(f"    {name:8s}  Google {g_pct:5.1f}%  wav2sleep {w_pct:5.1f}%")

    lines.append("=" * 60)
    return "\n".join(lines)


def output_paths_for(stages_csv):
    """Compute the comparison output paths for a sleep_stages CSV."""
    analysis_dir = os.path.dirname(stages_csv)
    basename = os.path.basename(stages_csv)
    timestamp = basename.replace("sleep_stages_", "").replace(".csv", "")
    return (
        analysis_dir,
        timestamp,
        os.path.join(analysis_dir, f"compare_google_{timestamp}.csv"),
        os.path.join(analysis_dir, f"compare_google_{timestamp}.txt"),
    )


def is_analyzed(stages_csv):
    """True if both compare_google outputs already exist."""
    _, _, cmp_csv, cmp_txt = output_paths_for(stages_csv)
    return os.path.exists(cmp_csv) and os.path.exists(cmp_txt)


def find_all_stages_csv():
    """Find all sleep_stages CSVs in data/*/analysis/, sorted oldest-first."""
    return sorted(glob.glob(os.path.join(DATA_DIR, "*", "analysis", "sleep_stages_*.csv")))


def find_latest_stages_csv():
    """Find the most recent sleep_stages CSV."""
    files = find_all_stages_csv()
    return files[-1] if files else None


def compare_one(stages_csv, google_df, force=False):
    """Run the comparison for one session. Returns True on success."""
    if not os.path.exists(stages_csv):
        print(f"Error: {stages_csv} not found")
        return False

    analysis_dir, timestamp, cmp_csv, cmp_txt = output_paths_for(stages_csv)

    if not force and is_analyzed(stages_csv):
        print(f"Already compared: {cmp_txt}")
        print("(use --force to re-run)")
        return True

    # Load vigil predictions
    preds = pd.read_csv(stages_csv)
    pred_labels = preds["Pred"].values.astype(int)
    epoch_sec = preds["Timestamp"].values.astype(float)

    # Absolute recording start from the PPG file's timestamp_ns
    session_dir = os.path.dirname(analysis_dir)
    session_ts = os.path.basename(session_dir)
    ppg_path = os.path.join(session_dir, "input", f"sleep_ppg_{session_ts}.csv")
    if not os.path.exists(ppg_path):
        print(f"Error: PPG file not found: {ppg_path}")
        return False
    ppg = pd.read_csv(ppg_path, usecols=["timestamp_ns"])
    start_ns = ppg["timestamp_ns"].iloc[0]
    start_dt = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    end_dt = start_dt + pd.to_timedelta(epoch_sec[-1] + EPOCH_SECONDS, unit="s")
    epoch_dt = start_dt + pd.to_timedelta(epoch_sec, unit="s")

    # Find overlapping Google data
    g_overlap = find_google_overlap(google_df, start_dt, end_dt)
    if g_overlap.empty:
        print(f"  No Google sleep data overlaps {session_ts}")
        print(f"  (recording: {start_dt} → {end_dt})")
        return False

    g_labels = assign_google_labels(g_overlap, epoch_dt)
    mask = g_labels >= 0
    n_overlap = int(mask.sum())

    if n_overlap == 0:
        print(f"  {session_ts}: no epochs fall inside Google intervals")
        return False

    # Align prediction labels to the same length
    p_aligned = pred_labels[: len(g_labels)]
    true = g_labels[mask]
    pred = p_aligned[mask]
    cm = confusion_matrix(true, pred)

    report = format_report(true, pred, cm)
    print(f"\n  {session_ts}  ({start_dt} → {end_dt})")
    print(f"  Google overlap: {n_overlap}/{len(g_labels)} epochs")
    print()
    print(report)

    # Save per-epoch comparison CSV
    out_df = pd.DataFrame({
        "Timestamp_sec": epoch_sec[mask],
        "DateTime": epoch_dt[mask],
        "Google": true,
        "wav2sleep": pred,
        "match": (true == pred).astype(int),
    })
    os.makedirs(analysis_dir, exist_ok=True)
    out_df.to_csv(cmp_csv, index=False)
    print(f"\n  Per-epoch comparison saved: {cmp_csv}")

    # Save text report
    with open(cmp_txt, "w") as f:
        f.write(f"Session: {session_ts}\n")
        f.write(f"Recording start (UTC): {start_dt}\n")
        f.write(f"Google overlap: {n_overlap}/{len(g_labels)} epochs\n\n")
        f.write(report + "\n")
    print(f"  Report saved: {cmp_txt}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare wav2sleep sleep stages against Google Health data."
    )
    parser.add_argument("stages_csv", nargs="?", default=None,
                        help="sleep_stages_<ts>.csv (default: latest in data/*/analysis/)")
    parser.add_argument("--all", action="store_true",
                        help="Compare all sessions with sleep_stages CSVs")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if comparison outputs already exist")
    args = parser.parse_args()

    if args.stages_csv:
        google_df = load_google_data()
        compare_one(args.stages_csv, google_df, force=args.force)
        return

    if args.all:
        files = find_all_stages_csv()
        if not files:
            print("No sleep_stages CSVs found in data/*/analysis/")
            sys.exit(1)
        google_df = load_google_data()
        print(f"Comparing {len(files)} session(s) against Google data...\n")
        compared = 0
        skipped = 0
        for stages_csv in files:
            _, ts, _, _ = output_paths_for(stages_csv)
            if not args.force and is_analyzed(stages_csv):
                print(f"  ✓ {ts} — already compared, skipping")
                skipped += 1
                continue
            print(f"  → {ts}...")
            if compare_one(stages_csv, google_df, force=args.force):
                compared += 1
            print()
        print("=" * 55)
        print(f"  Compared {compared}, already complete {skipped}")
        print("=" * 55)
        return

    # Default: pick the latest session that's missing a comparison,
    # falling back to the latest session overall.
    files = find_all_stages_csv()
    if not files:
        print("No sleep_stages CSVs found in data/*/analysis/")
        sys.exit(1)
    google_df = load_google_data()
    incomplete = [f for f in files if not is_analyzed(f)]
    if incomplete:
        stages_csv = incomplete[-1]
        print(f"Auto-selected latest un-compared: {stages_csv}")
    else:
        stages_csv = files[-1]
        print(f"Auto-selected latest: {stages_csv}")
    compare_one(stages_csv, google_df, force=args.force)


if __name__ == "__main__":
    main()