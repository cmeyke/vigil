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

Each session's parquet is written ONCE to data/finetune/<run>/sessions/,
then folds are built via symlinks (no duplicated data — at 100 nights this
saves ~69 GB vs. one-copy-per-fold).

Cross-validation strategy:
  - N < 30:  leave-one-out (N folds, each trains on N-1)
  - N >= 30: k-fold (10 folds by default, each trains on N*(k-1)/k)
  Override with --cv {loo,kfold} and --folds N.

Sessions flagged implausible in hrv_comparison.csv or with <50% Google epoch
overlap (per compare_google_<ts>.csv 'match' column mean) are excluded.

Usage:
    uv run prepare_finetune_data.py                           # run name = timestamp
    uv run prepare_finetune_data.py --run-name myrun           # custom run name
    uv run prepare_finetune_data.py --min-overlap 0.5          # require 50% Google overlap
    uv run prepare_finetune_data.py --cv loo                   # force leave-one-out
    uv run prepare_finetune_data.py --cv kfold --folds 5       # force 5-fold
    uv run prepare_finetune_data.py --dry-run                  # list sessions, write nothing

Output:
    data/finetune/<run_name>/
        sessions/<session>.parquet             (one per night, written once)
        fold_0/{train,val}/<session>.parquet   (symlinks to sessions/)
        fold_1/{train,val}/<session>.parquet
        ...
        folds.json
"""

import argparse
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

# Google 4-class label -> 5-class int (wav2sleep's expected encoding)
# 0=Wake->0, 1=Light->N1(1), 2=Deep->N3(3), 3=REM->4
# N2 (2) is never produced by Google — only N1 (1). The 4-class dataloader
# merges N1+N2 -> Light via INTEGER_LABEL_MAPS[4] = {0:0, 1:1, 2:1, 3:2, 4:3}.
GOOGLE_TO_5CLASS = {0: 0, 1: 1, 2: 3, 3: 4}
UNSCORED = -1

# CV strategy auto-selection threshold (see AGENTS.md / README)
LOO_MAX_N = 30
DEFAULT_KFOLD_K = 10


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

        cmp = pd.read_csv(compare_path)
        if len(cmp) == 0:
            continue
        overlap_frac = (cmp["Google"] >= 0).mean()

        sessions.append({
            "session": session_dir,
            "ppg_path": ppg_path,
            "compare_path": compare_path,
            "overlap_frac": overlap_frac,
        })

    included = [s for s in sessions if s["overlap_frac"] >= min_overlap]
    excluded = [s for s in sessions if s["overlap_frac"] < min_overlap]
    return included, excluded


def load_implausible_sessions():
    """Read hrv_comparison.csv and return set of implausible session IDs."""
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

    Returns (resampled_signal, num_epochs). The signal is at ~34.13 Hz
    (1024 samples per 30 s epoch). Linear interpolation matches wav2sleep's
    own resampling. Length is floor(recording_seconds * 1024 / 30) samples.
    """
    df = pd.read_csv(ppg_path, usecols=["timestamp_ns", "ch0", "ch1", "ch2", "ch3"])
    ppg = df[["ch0", "ch1", "ch2", "ch3"]].mean(axis=1).values.astype(float)

    start_ns = df["timestamp_ns"].iloc[0]
    t_src = (df["timestamp_ns"].values - start_ns) / 1e9

    duration_s = t_src[-1]
    num_epochs = int(duration_s // EPOCH_SECONDS)
    t_target = np.arange(num_epochs * SAMPLES_PER_EPOCH) / SAMPLES_PER_EPOCH * EPOCH_SECONDS
    t_target = t_target[t_target < t_src[-1]]

    interp = interp1d(t_src, ppg, kind="linear", bounds_error=False, fill_value="extrapolate")
    resampled = interp(t_target)

    return resampled, num_epochs


def build_labels(compare_path, num_epochs):
    """Build per-epoch label array from compare_google CSV.

    Returns int8 array of length num_epochs with values in 0-4 or -1.
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

    PPG is stored as a continuous signal of length N*SAMPLES_PER_EPOCH.
    Stage is stored per-sample: label at the first sample of each epoch,
    NaN elsewhere. The wav2sleep dataloader recovers one-label-per-epoch
    via dropna() and asserts the count equals num_epochs. Unscored epochs
    get -1 (a real value, not NaN, so dropna keeps them).
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
        assert len(labels) == num_epochs, (
            f"Label count {len(labels)} != num_epochs {num_epochs} in {path}"
        )
    return num_epochs


def write_session_parquet(session, output_path):
    """Build and write one session's parquet from PPG + compare_google CSV."""
    ppg_signal, num_epochs = resample_ppg_to_epochs(session["ppg_path"])
    labels = build_labels(session["compare_path"], num_epochs)
    write_parquet(ppg_signal, labels, output_path)
    actual_epochs = validate_parquet(output_path)
    assert actual_epochs == num_epochs, (
        f"Epoch mismatch in {output_path}: expected {num_epochs}, got {actual_epochs}"
    )


def write_all_session_parquets(sessions, sessions_dir, dry_run=False):
    """Write each session's parquet ONCE to sessions_dir.

    Returns a dict mapping session_id -> absolute path to its parquet.
    Skips sessions that already have a valid parquet (idempotent).
    """
    session_paths = {}
    if dry_run:
        for s in sessions:
            session_paths[s["session"]] = os.path.join(sessions_dir, f"{s['session']}.parquet")
        return session_paths

    os.makedirs(sessions_dir, exist_ok=True)
    for s in sessions:
        out = os.path.join(sessions_dir, f"{s['session']}.parquet")
        if os.path.exists(out):
            print(f"    {s['session']}.parquet (exists, skipping)")
            session_paths[s["session"]] = out
            continue
        write_session_parquet(s, out)
        print(f"    {s['session']}.parquet")
        session_paths[s["session"]] = out
    return session_paths


def make_symlink(src, dst):
    """Create a symlink dst -> src, removing dst if it already exists."""
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def build_loo_folds(sessions, session_paths, run_dir, dry_run=False):
    """Build leave-one-out CV fold directories using symlinks.

    For N sessions, creates N folds. Each fold's train/ and val/ contain
    symlinks to the session parquets in sessions_dir — no duplicated data.
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
            "held_out_sessions": [held_out["session"]],
            "train_sessions": [s["session"] for s in train_sessions],
            "fold_dir": os.path.relpath(fold_dir, DATA_DIR),
        }

        if dry_run:
            folds.append(fold)
            continue

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for s in train_sessions:
            make_symlink(
                session_paths[s["session"]],
                os.path.join(train_dir, f"{s['session']}.parquet"),
            )
        make_symlink(
            session_paths[held_out["session"]],
            os.path.join(val_dir, f"{held_out['session']}.parquet"),
        )
        print(f"    fold_{i}: train={len(train_sessions)}, val=1 (held out: {held_out['session']})")
        folds.append(fold)

    return folds


def build_kfold_folds(sessions, session_paths, run_dir, k, dry_run=False):
    """Build k-fold CV fold directories using symlinks.

    Splits sessions into k contiguous groups (preserving chronological order
    since sessions are sorted by timestamp). Each fold holds out one group
    for validation and trains on the rest.
    """
    n = len(sessions)
    fold_size = n // k
    remainder = n % k

    folds = []
    idx = 0
    groups = []
    for i in range(k):
        size = fold_size + (1 if i < remainder else 0)
        groups.append(sessions[idx:idx + size])
        idx += size

    for i in range(k):
        val_sessions = groups[i]
        train_sessions = [s for j, g in enumerate(groups) if j != i for s in g]

        fold_dir = os.path.join(run_dir, f"fold_{i}")
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")

        fold = {
            "fold": i,
            "held_out_sessions": [s["session"] for s in val_sessions],
            "train_sessions": [s["session"] for s in train_sessions],
            "fold_dir": os.path.relpath(fold_dir, DATA_DIR),
        }

        if dry_run:
            folds.append(fold)
            continue

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for s in train_sessions:
            make_symlink(
                session_paths[s["session"]],
                os.path.join(train_dir, f"{s['session']}.parquet"),
            )
        for s in val_sessions:
            make_symlink(
                session_paths[s["session"]],
                os.path.join(val_dir, f"{s['session']}.parquet"),
            )
        print(f"    fold_{i}: train={len(train_sessions)}, val={len(val_sessions)} "
              f"(held out: {', '.join(s['session'] for s in val_sessions)})")
        folds.append(fold)

    return folds


def select_cv_strategy(n, cv_arg, folds_arg):
    """Determine CV strategy and number of folds.

    Returns (strategy, num_folds) where strategy is 'loo' or 'kfold'.
    """
    if cv_arg == "loo":
        return "loo", n
    if cv_arg == "kfold":
        k = folds_arg if folds_arg and folds_arg > 0 else DEFAULT_KFOLD_K
        k = min(k, n)
        return "kfold", k
    # Auto: LOO for small N, k-fold for large N
    if n < LOO_MAX_N:
        return "loo", n
    k = folds_arg if folds_arg and folds_arg > 0 else DEFAULT_KFOLD_K
    k = min(k, n)
    return "kfold", k


def main():
    parser = argparse.ArgumentParser(
        description="Prepare paired vigil+Google data for wav2sleep fine-tuning."
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this preparation run (default: timestamp)")
    parser.add_argument("--min-overlap", type=float, default=0.5,
                        help="Minimum Google epoch overlap fraction (default: 0.5)")
    parser.add_argument("--cv", choices=["loo", "kfold", "auto"], default="auto",
                        help="Cross-validation strategy: loo (leave-one-out), "
                             "kfold (k-fold), or auto (loo if N<30, kfold otherwise)")
    parser.add_argument("--folds", type=int, default=None,
                        help=f"Number of folds for kfold CV (default: {DEFAULT_KFOLD_K}). "
                             "Ignored for loo.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List sessions and fold plan without writing parquets")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(DATA_DIR, "finetune", run_name)
    sessions_dir = os.path.join(run_dir, "sessions")

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
    implausible_excluded = [s for s in sessions if s["session"] in implausible]
    sessions = [s for s in sessions if s["session"] not in implausible]

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

    # Select CV strategy
    n = len(sessions)
    strategy, num_folds = select_cv_strategy(n, args.cv, args.folds)
    print(f"\n  CV strategy: {strategy} ({num_folds} folds)")
    if args.cv == "auto" and strategy == "kfold":
        print(f"  (auto-selected kfold: N={n} >= {LOO_MAX_N})")
    elif args.cv == "auto" and strategy == "loo":
        print(f"  (auto-selected loo: N={n} < {LOO_MAX_N})")

    # Write session parquets (once, not per-fold)
    print(f"\n  Writing {n} session parquet(s)...")
    session_paths = write_all_session_parquets(sessions, sessions_dir, dry_run=args.dry_run)

    # Build folds via symlinks
    print(f"\n  Building {num_folds} {strategy} fold(s)...")
    if strategy == "loo":
        folds = build_loo_folds(sessions, session_paths, run_dir, dry_run=args.dry_run)
    else:
        folds = build_kfold_folds(sessions, session_paths, run_dir, num_folds, dry_run=args.dry_run)

    # Write manifest
    if not args.dry_run:
        manifest_path = os.path.join(run_dir, "folds.json")
        with open(manifest_path, "w") as f:
            json.dump({
                "run_name": run_name,
                "created": datetime.now().isoformat(),
                "num_sessions": n,
                "num_folds": len(folds),
                "cv_strategy": strategy,
                "min_overlap": args.min_overlap,
                "sessions": [s["session"] for s in sessions],
                "folds": folds,
            }, f, indent=2)
        print(f"\n  Manifest: {manifest_path}")
        print(f"  Parquets: {n} session files (symlinked into {len(folds)} folds)")
    else:
        print(f"\n  --dry-run: no files written. Planned folds:")
        for fold in folds:
            held = ", ".join(fold["held_out_sessions"])
            print(f"    fold {fold['fold']}: held_out={held}, "
                  f"train={len(fold['train_sessions'])} sessions")


if __name__ == "__main__":
    main()