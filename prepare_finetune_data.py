"""
vigil — prepare paired data for wav2sleep fine-tuning

Converts each paired night (vigil PPG + Google sleep stages from
compare_google_<ts>.csv) into a wav2sleep-format parquet file with:
  - PPG column: single-channel PPG resampled to 1024 samples per 30 s epoch
    (~34.13 Hz, wav2sleep's expected rate for PPG)
  - Stage column: int8 in 0-4 (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
    Google's 4-class labels are mapped to the 5-class space: Light(1)->N1(1),
    Deep(2)->N3(3), REM(3)->REM(4). The 4-class dataloader merges N1+N2->Light
    automatically via INTEGER_LABEL_MAPS[4]. Unscored epochs -> -1.

Organizes parquets into leave-one-out CV folds under
data/finetune/<run_name>/fold_<n>/{train,val}/<session>.parquet, plus a
folds.json manifest. Each fold holds out one session for validation.

Sessions flagged implausible in hrv_comparison.csv or with <50% Google epoch
overlap (per compare_google_<ts>.csv 'match' column mean) are excluded.

Usage:
    uv run prepare_finetune_data.py                           # run name = timestamp
    uv run prepare_finetune_data.py --run-name myrun           # custom run name
    uv run prepare_finetune_data.py --min-overlap 0.5          # require 50% Google overlap
    uv run prepare_finetune_data.py --dry-run                  # list sessions, write nothing

Output:
    data/finetune/<run_name>/
        fold_0/{train,val}/<session>.parquet
        fold_1/{train,val}/<session>.parquet
        ...
        folds.json
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# wav2sleep's PPG sample rate: 1024 samples per 30 s epoch = 34.13 Hz
SAMPLES_PER_EPOCH = 1024
EPOCH_SECONDS = 30
RAW_PPG_HZ = 55  # vigil Verity Sense PPG sample rate

# Google 4-class label -> 5-class int (wav2sleep's expected encoding)
# 0=Wake->0, 1=Light->N1(1), 2=Deep->N3(3), 3=REM->4
# N2 (2) is never produced by Google — only N1 (1). The 4-class dataloader
# merges N1+N2 -> Light via INTEGER_LABEL_MAPS[4] = {0:0, 1:1, 2:1, 3:2, 4:3}.
GOOGLE_TO_5CLASS = {0: 0, 1: 1, 2: 3, 3: 4}
UNSCORED = -1


def is_timestamp_dir(name):
    return (
        len(name) == 15
        and name[8] == "_"
        and name[:8].isdigit()
        and name[9:].isdigit()
    )


def find_paired_sessions(min_overlap=0.5):
    """Find sessions with both a PPG recording and a Google comparison CSV.

    Returns a list of dicts: {session, ppg_path, compare_path, overlap_frac}.
    Skips sessions missing either file or with overlap below min_overlap.
    """
    sessions = []
    for session_dir in sorted(os.listdir(DATA_DIR)):
        if not is_timestamp_dir(session_dir):
            continue
        ppg_path = os.path.join(DATA_DIR, session_dir, "input", f"sleep_ppg_{session_dir}.csv")
        compare_path = os.path.join(
            DATA_DIR, session_dir, "analysis", f"compare_google_{session_dir}.csv"
        )
        if not os.path.exists(ppg_path) or not os.path.exists(compare_path):
            continue

        # Compute Google epoch overlap fraction
        cmp = pd.read_csv(compare_path)
        if len(cmp) == 0:
            continue
        # 'Google' column has -1 for no-Google-coverage epochs
        overlap_frac = (cmp["Google"] >= 0).mean()

        sessions.append({
            "session": session_dir,
            "ppg_path": ppg_path,
            "compare_path": compare_path,
            "overlap_frac": overlap_frac,
        })

    # Filter by overlap
    included = [s for s in sessions if s["overlap_frac"] >= min_overlap]
    excluded = [s for s in sessions if s["overlap_frac"] < min_overlap]
    return included, excluded


def load_implausible_sessions():
    """Read hrv_comparison.csv and return set of implausible session IDs.

    These sessions have broken peak detection (e.g. RMSSD outside 5-500 ms)
    and should be excluded from fine-tuning. Returns empty set if file missing.
    """
    path = os.path.join(DATA_DIR, "hrv_comparison.csv")
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        if "plausible" not in df.columns or "session" not in df.columns:
            return set()
        return set(df.loc[~df["plausible"], "session"])
    except (OSError, ValueError, KeyError):
        return set()


def resample_ppg_to_epochs(ppg_path):
    """Load vigil PPG (55 Hz, 4 channels) and resample to wav2sleep's format.

    Returns a 1-D numpy array containing the single-channel PPG (mean of
    ch0-ch3) at ~34.13 Hz (1024 samples per 30 s epoch). Linear interpolation
    matches wav2sleep's own resampling. The length is determined by the
    recording duration: floor(recording_seconds * 1024 / 30) samples.
    """
    df = pd.read_csv(ppg_path, usecols=["timestamp_ns", "ch0", "ch1", "ch2", "ch3"])
    ppg = df[["ch0", "ch1", "ch2", "ch3"]].mean(axis=1).values.astype(float)

    start_ns = df["timestamp_ns"].iloc[0]
    t_src = (df["timestamp_ns"].values - start_ns) / 1e9

    # Recording duration in seconds → number of complete 30 s epochs
    duration_s = t_src[-1]
    num_epochs = int(duration_s // EPOCH_SECONDS)
    # Target timestamps: 1024 samples per 30 s epoch, uniformly sampled
    t_target = np.arange(num_epochs * SAMPLES_PER_EPOCH) / SAMPLES_PER_EPOCH * EPOCH_SECONDS
    # Clip to source range to avoid extrapolation beyond the last sample
    t_target = t_target[t_target < t_src[-1]]

    interp = interp1d(t_src, ppg, kind="linear", bounds_error=False, fill_value="extrapolate")
    resampled = interp(t_target)

    return resampled, num_epochs


def build_labels(compare_path, num_epochs):
    """Build per-epoch label array from compare_google CSV.

    Returns int8 array of length num_epochs with values in 0-4 or -1.
    Labels come from the 'Google' column (0-3 = W/L/D/REM), mapped to
    the 5-class space (0/1/3/4). Epochs without Google coverage (-1 in the
    source) or beyond the compare CSV's span become -1 (unscored, ignored
    by the loss via ignore_index=-1).
    """
    labels = np.full(num_epochs, UNSCORED, dtype=np.int8)
    cmp = pd.read_csv(compare_path)
    for _, row in cmp.iterrows():
        epoch_idx = int(row["Timestamp_sec"] / EPOCH_SECONDS)
        if 0 <= epoch_idx < num_epochs:
            g_label = int(row["Google"])
            if g_label >= 0:
                labels[epoch_idx] = GOOGLE_TO_5CLASS.get(g_label, UNSCORED)
    return labels


def write_parquet(ppg_signal, labels, output_path):
    """Write a single parquet file with PPG + Stage columns.

    PPG is stored as a single column of length N*SAMPLES_PER_EPOCH
    (one continuous signal at ~34.13 Hz).

    Stage is stored as a per-sample column (length N*SAMPLES_PER_EPOCH) with
    the label value at the first sample of each epoch and NaN elsewhere.
    The wav2sleep dataloader recovers one-label-per-epoch via dropna(), and
    asserts the label count equals num_epochs (PPG length // SAMPLES_PER_EPOCH).
    So labels must cover every epoch — unscored epochs get -1 (mapped to -1
    by the loss's ignore_index=-1), NOT NaN (which dropna would remove,
    causing a count mismatch).

    The expected num_epochs is len(labels). PPG is padded/truncated to match
    labels * SAMPLES_PER_EPOCH.
    """
    num_epochs = len(labels)
    expected_ppg_len = num_epochs * SAMPLES_PER_EPOCH
    if len(ppg_signal) < expected_ppg_len:
        pad_val = ppg_signal[-1] if len(ppg_signal) else 0.0
        ppg_signal = np.concatenate([
            ppg_signal,
            np.full(expected_ppg_len - len(ppg_signal), pad_val),
        ])
    elif len(ppg_signal) > expected_ppg_len:
        ppg_signal = ppg_signal[:expected_ppg_len]

    # Stage per sample: label at first sample of each epoch, NaN elsewhere.
    # -1 (unscored) is a real value, not NaN, so dropna keeps it → one-per-epoch.
    # The dataloader then maps -1 via INTEGER_LABEL_MAPS (which preserves -1 as
    # NaN → fillna(-1) → ignored by CrossEntropyLoss(ignore_index=-1)).
    stage_per_sample = np.full(len(ppg_signal), np.nan, dtype=np.float32)
    for i, label in enumerate(labels):
        stage_per_sample[i * SAMPLES_PER_EPOCH] = float(label)

    df = pd.DataFrame({"PPG": ppg_signal, "Stage": stage_per_sample})
    df.to_parquet(output_path, index=False)


def validate_parquet(path):
    """Re-read the parquet and assert shape/dtype/label range are correct."""
    df = pd.read_parquet(path)
    assert "PPG" in df.columns, f"Missing PPG column in {path}"
    assert "Stage" in df.columns, f"Missing Stage column in {path}"
    ppg_len = len(df)
    assert ppg_len % SAMPLES_PER_EPOCH == 0, (
        f"PPG length {ppg_len} not divisible by {SAMPLES_PER_EPOCH} in {path}"
    )
    num_epochs = ppg_len // SAMPLES_PER_EPOCH
    labels = df["Stage"].dropna()
    if len(labels) > 0:
        label_vals = labels.unique()
        valid = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}
        bad = set(label_vals) - valid
        assert not bad, f"Unexpected label values {bad} in {path}"
        # One label per epoch (dropna should yield exactly num_epochs)
        assert len(labels) == num_epochs, (
            f"Label count {len(labels)} != num_epochs {num_epochs} in {path}"
        )
    return num_epochs


def build_folds(sessions, run_dir, dry_run=False):
    """Build leave-one-out CV fold directories.

    For N sessions, creates N folds. Each fold has:
      train/<other_session>.parquet  (N-1 files)
      val/<held_out_session>.parquet (1 file)
    Returns a list of fold dicts for the manifest.
    """
    folds = []
    n = len(sessions)
    for i in range(n):
        held_out = sessions[i]
        train_sessions = [s for j, s in enumerate(sessions) if j != i]

        fold_dir = os.path.join(run_dir, f"fold_{i}")
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")

        fold = {
            "fold": i,
            "held_out_session": held_out["session"],
            "train_sessions": [s["session"] for s in train_sessions],
            "fold_dir": os.path.relpath(fold_dir, DATA_DIR),
        }

        if dry_run:
            folds.append(fold)
            continue

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Write training parquets
        for s in train_sessions:
            out = os.path.join(train_dir, f"{s['session']}.parquet")
            write_session_parquet(s, out)
            print(f"    train/{s['session']}.parquet")

        # Write held-out val parquet
        out = os.path.join(val_dir, f"{held_out['session']}.parquet")
        write_session_parquet(held_out, out)
        print(f"    val/{held_out['session']}.parquet (held out)")

        folds.append(fold)

    return folds


def write_session_parquet(session, output_path):
    """Build and write one session's parquet from PPG + compare_google CSV."""
    # Resample PPG to wav2sleep's 1024-samples-per-epoch format; num_epochs
    # is derived from the PPG recording duration (not the compare CSV, which
    # may cover fewer epochs than the recording).
    ppg_signal, num_epochs = resample_ppg_to_epochs(session["ppg_path"])

    # Build per-epoch labels (length num_epochs; -1 for uncovered epochs)
    labels = build_labels(session["compare_path"], num_epochs)

    # Write parquet (labels drive the epoch count; PPG is padded/truncated
    # to match labels * SAMPLES_PER_EPOCH)
    write_parquet(ppg_signal, labels, output_path)

    # Validate
    actual_epochs = validate_parquet(output_path)
    assert actual_epochs == num_epochs, (
        f"Epoch mismatch in {output_path}: expected {num_epochs}, got {actual_epochs}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare paired vigil+Google data for wav2sleep fine-tuning."
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this preparation run (default: timestamp)")
    parser.add_argument("--min-overlap", type=float, default=0.5,
                        help="Minimum Google epoch overlap fraction (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List sessions and fold plan without writing parquets")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(DATA_DIR, "finetune", run_name)

    print("╔══════════════════════════════════════════╗")
    print("║  vigil — prepare fine-tuning data         ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Run name:    {run_name}")
    print(f"  Output dir:  {run_dir}")
    print(f"  Min overlap: {args.min_overlap:.0%}")
    print()

    # Find paired sessions
    sessions, excluded = find_paired_sessions(min_overlap=args.min_overlap)
    implausible = load_implausible_sessions()
    sessions = [s for s in sessions if s["session"] not in implausible]
    implausible_excluded = [s for s in sessions if s["session"] in implausible]

    if not sessions:
        print("  No paired sessions found.")
        print("  Run import-android.py + fetch_google_sleep.py + compare_google.py first.")
        sys.exit(1)

    print(f"  Paired sessions: {len(sessions)}")
    for s in sessions:
        print(f"    {s['session']}  (Google overlap: {s['overlap_frac']:.0%})")

    if excluded:
        print(f"\n  Excluded (overlap < {args.min_overlap:.0%}):")
        for s in excluded:
            print(f"    {s['session']}  (overlap: {s['overlap_frac']:.0%})")

    if implausible_excluded:
        print(f"\n  Excluded (implausible per hrv_comparison.csv):")
        for s in implausible_excluded:
            print(f"    {s['session']}")

    print(f"\n  Building {len(sessions)} leave-one-out folds...")

    folds = build_folds(sessions, run_dir, dry_run=args.dry_run)

    # Write manifest
    if not args.dry_run:
        manifest_path = os.path.join(run_dir, "folds.json")
        with open(manifest_path, "w") as f:
            json.dump({
                "run_name": run_name,
                "created": datetime.now().isoformat(),
                "num_sessions": len(sessions),
                "num_folds": len(folds),
                "min_overlap": args.min_overlap,
                "sessions": [s["session"] for s in sessions],
                "folds": folds,
            }, f, indent=2)
        print(f"\n  Manifest: {manifest_path}")
        print(f"  Parquets written: {len(sessions)} sessions x LOO = {len(sessions)} val + {len(sessions) * (len(sessions) - 1)} train")
    else:
        print(f"\n  --dry-run: no files written. Planned folds:")
        for fold in folds:
            print(f"    fold {fold['fold']}: held_out={fold['held_out_session']}, "
                  f"train={len(fold['train_sessions'])} sessions")


if __name__ == "__main__":
    main()