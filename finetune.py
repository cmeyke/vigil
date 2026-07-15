#!/usr/bin/env python3
"""
vigil — fine-tune wav2sleep on paired vigil + Google nights

Orchestrates leave-one-out cross-validation fine-tuning of the released
wav2sleep model (hf://joncarter/wav2sleep) on paired vigil PPG + Google
sleep stage labels. Conservative strategy: freeze signal_encoders +
epoch_mixer, train only sequence_mixer + classifier at low LR.

This script runs in the wav2sleep virtualenv (not vigil's venv) because it
imports torch, lightning, and wav2sleep's trainer module. It self-re-execs
into the wav2sleep env if invoked from vigil's `uv run`.

Usage:
    uv run finetune.py                              # default: needs 10 paired nights
    uv run finetune.py --min-nights 5                # enable "meaningful" band
    uv run finetune.py --min-nights 4               # enable testing (current data)
    uv run finetune.py --min-nights 0               # disable gate entirely
    uv run finetune.py --run-name myrun              # use a specific prepared run
    uv run finetune.py --resume                      # skip completed folds

Requires:
    - prepare_finetune_data.py must have been run to produce
      data/finetune/<run_name>/fold_*/{train,val}/*.parquet + folds.json
    - wav2sleep venv at ~/code/python/ai/wav2sleep-env with lightning + hydra
      installed (the script will check and print install instructions if not)

Output:
    data/models/vigil_finetuned_<run_name>_fold<n>/{config.yaml,state_dict.pth}
    data/models/vigil_finetuned_<run_name>_best/  (best by mean kappa)
    data/finetune/<run_name>/results.csv           (per-fold held-out metrics)
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FINETUNE_DIR = os.path.join(DATA_DIR, "finetune")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# wav2sleep venv paths
WAV2SLEEP_VENV = os.path.expanduser("~/code/python/ai/wav2sleep-env")
WAV2SLEEP_PYTHON = os.path.join(WAV2SLEEP_VENV, ".venv", "bin", "python")

# Gating thresholds (see README "Fine-tuning" section)
MIN_NIGHTS_IDEAL = 10        # default minimum — the "ideal" threshold
MIN_NIGHTS_MEANINGFUL = 5   # warning band starts here


def reexec_in_wav2sleep_env():
    """Re-exec this script under the wav2sleep venv's Python if not already there."""
    current_python = sys.executable
    if os.path.abspath(current_python) == os.path.abspath(WAV2SLEEP_PYTHON):
        return  # already in wav2sleep env
    if not os.path.exists(WAV2SLEEP_PYTHON):
        print(f"Error: wav2sleep venv not found at {WAV2SLEEP_PYTHON}")
        print("Create it with:")
        print("  cd ~/code/python/ai")
        print("  uv venv wav2sleep-env --python 3.12")
        print("  uv pip install --python wav2sleep-env/.venv/bin/python \\")
        print("      'git+https://github.com/joncarter1/wav2sleep.git' \\")
        print("      lightning hydra-core mlflow")
        sys.exit(1)
    # Re-exec with the same args
    os.execv(WAV2SLEEP_PYTHON, [WAV2SLEEP_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])


def check_dependencies():
    """Verify torch, lightning, hydra, wav2sleep are importable. Exit with instructions if not."""
    missing = []
    for mod in ["torch", "lightning", "hydra", "wav2sleep"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"Error: missing dependencies in wav2sleep env: {missing}")
        print(f"Install with:")
        print(f"  uv pip install --python {WAV2SLEEP_VENV}/.venv/bin/python \\")
        print(f"      {' '.join(missing)}")
        sys.exit(1)


def check_cuda():
    """Verify CUDA is available. Fine-tuning on CPU is impractical."""
    import torch
    if not torch.cuda.is_available():
        print("Error: CUDA not available. Fine-tuning requires a GPU.")
        print("If you have one, ensure the wav2sleep env's torch was installed with CUDA support.")
        print("The data-prep path (prepare_finetune_data.py) does not need GPU — only this script.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")


def load_run_manifest(run_name):
    """Load folds.json for the given run. Returns the manifest dict."""
    manifest_path = os.path.join(FINETUNE_DIR, run_name, "folds.json")
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found.")
        print(f"Run prepare_finetune_data.py --run-name {run_name} first.")
        sys.exit(1)
    with open(manifest_path) as f:
        return json.load(f)


def get_completed_folds(run_dir):
    """Return set of fold indices that already have a saved model."""
    completed = set()
    pattern = os.path.join(run_dir, "fold_*", "config.yaml")
    for path in glob.glob(pattern):
        fold_dir = os.path.dirname(path)
        sd = os.path.join(fold_dir, "state_dict.pth")
        if os.path.exists(sd):
            fold_idx = int(os.path.basename(fold_dir).split("_")[1])
            completed.add(fold_idx)
    return completed


def load_pretrained_model():
    """Load the released wav2sleep model from HF Hub. Returns a Wav2Sleep instance on CPU."""
    from wav2sleep.api import load_model
    print("  Loading pretrained model from hf://joncarter/wav2sleep...")
    model = load_model("hf://joncarter/wav2sleep", device="cpu")
    print(f"    ✓ {type(model).__name__}, num_classes={model.num_classes}, "
          f"signals={model.valid_signals}")
    return model


def build_lightning_module(model, freeze_encoders=True):
    """Wrap a Wav2Sleep model in SleepLightningModule, optionally freezing encoders.

    Conservative freeze (default): signal_encoders + epoch_mixer frozen,
    only sequence_mixer + classifier train (~823K params).
    """
    from wav2sleep.trainer import SleepLightningModule
    import torch.nn as nn

    if freeze_encoders:
        frozen = []
        trainable = []
        for name, param in model.named_parameters():
            if name.startswith("signal_encoders.") or name.startswith("epoch_mixer."):
                param.requires_grad_(False)
                frozen.append(name)
            else:
                trainable.append(name)
        print(f"    Frozen {len(frozen)} param tensors (signal_encoders + epoch_mixer)")
        print(f"    Trainable {len(trainable)} param tensors (sequence_mixer + classifier)")
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"    Trainable: {n_trainable:,} / {n_total:,} params ({100*n_trainable/n_total:.1f}%)")

    # Optimizer: AdamW with low LR for fine-tuning (1e-4, vs 1e-3 for from-scratch)
    def optimizer_fn(params):
        import torch.optim as optim
        return optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # Only pass trainable params to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = lambda _ignored: optimizer_fn(trainable_params)

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)

    pl_model = SleepLightningModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_classes=model.num_classes,
        flip_polarity=True,
        on_step=False,
        on_epoch=True,
        debug_level=1,
    )
    return pl_model


def build_datamodule(fold_dir, num_classes=4, batch_size=4):
    """Build a SleepDataModule for one fold's train/val parquets.

    The fold_dir must contain train/*.parquet and val/*.parquet.
    We treat the fold_dir itself as the data_location and 'data' as the
    dataset name, with train/ and val/ subdirectories — matching the layout
    SleepDataModule expects: {data_location}/{dataset}/{train,val}/*.parquet.
    """
    from wav2sleep.data.datamodule import SleepDataModule

    # SleepDataModule globs {data_location}/{dataset}/{train,val}/*.parquet
    # Our fold_dir has train/ and val/ directly. Use fold_dir's parent as
    # data_location and fold_dir's basename as the dataset name.
    data_location = os.path.dirname(fold_dir)
    dataset_name = os.path.basename(fold_dir)

    dm = SleepDataModule(
        columns=["PPG"],
        num_classes=num_classes,
        data_location=data_location,
        train_datasets=[dataset_name],
        val_datasets=[dataset_name],
        test=False,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    return dm


def save_model(pl_model, model_cfg_dict, output_dir):
    """Save the fine-tuned model in HF-folder format (config.yaml + state_dict.pth)."""
    import torch
    import yaml
    os.makedirs(output_dir, exist_ok=True)
    # Save inner Wav2Sleep state_dict (not the Lightning wrapper)
    sd_path = os.path.join(output_dir, "state_dict.pth")
    torch.save(pl_model.model.state_dict(), sd_path)
    # Save model config (copied from the pretrained model's config.yaml)
    cfg_path = os.path.join(output_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(model_cfg_dict, f, sort_keys=False)
    return sd_path, cfg_path


def get_model_config_dict():
    """Read the config.yaml from the cached hf://joncarter/wav2sleep download."""
    from wav2sleep.hub import download_from_hub
    folder = download_from_hub("hf://joncarter/wav2sleep")
    cfg_path = os.path.join(folder, "config.yaml")
    import yaml
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def run_fold(fold, manifest, run_name, args):
    """Run one CV fold: train on the train set, save model, eval on held-out."""
    import torch
    from wav2sleep.api import load_model

    fold_idx = fold["fold"]
    held_out_sessions = fold["held_out_sessions"]  # list (1 for LOO, possibly >1 for kfold)
    fold_dir = os.path.join(FINETUNE_DIR, run_name, fold["fold_dir"])
    model_dir = os.path.join(MODELS_DIR, f"vigil_finetuned_{run_name}_fold{fold_idx}")

    held_str = ", ".join(held_out_sessions)
    print(f"\n  Fold {fold_idx}/{len(manifest['folds'])-1}  (held out: {held_str})")
    print(f"  Fold dir: {fold_dir}")
    print(f"  Model dir: {model_dir}")

    if os.path.exists(os.path.join(model_dir, "state_dict.pth")) and not args.force:
        print(f"  ✓ Already done — skipping (use --force to re-run)")
        return model_dir

    # Load fresh pretrained model for each fold (independent weights)
    pretrained = load_pretrained_model()
    model_cfg = get_model_config_dict()

    # Wrap in Lightning module with conservative freeze
    pl_model = build_lightning_module(pretrained, freeze_encoders=True)

    # Build datamodule for this fold
    dm = build_datamodule(fold_dir, num_classes=pretrained.num_classes, batch_size=args.batch_size)
    print(f"  Train sessions: {len(fold['train_sessions'])}")
    print(f"  Val (held out): {held_str}")

    # Build trainer
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_last=True,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        gradient_clip_val=1.0,
        precision="32-true",
        enable_checkpointing=True,
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train (no ckpt_path — fresh fine-tune from HF weights, not a Lightning resume)
    print(f"  Training (max {args.epochs} epochs, lr=1e-4, batch_size={args.batch_size})...")
    trainer.fit(pl_model, datamodule=dm)

    # Save the fine-tuned model in HF-folder format
    sd_path, cfg_path = save_model(pl_model, model_cfg, model_dir)
    print(f"  Saved: {sd_path}")
    print(f"        {cfg_path}")

    # Clean up Lightning checkpoints to save disk
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    return model_dir


def evaluate_held_out(model_dir, held_out_sessions, run_name, fold_idx, force_eval=False):
    """Run sleep_staging.py + compare_google.py on each held-out session.

    For LOO there's one session; for kfold there may be several. Metrics
    are averaged across held-out sessions in this fold.

    Returns a dict of averaged metrics, or None if eval was skipped/failed.
    """
    all_metrics = []
    for session in held_out_sessions:
        ft_cmp_txt = os.path.join(
            DATA_DIR, session, "analysis",
            f"compare_google_{session}_ft_fold{fold_idx}.txt"
        )
        if os.path.exists(ft_cmp_txt) and not force_eval:
            print(f"  {session}: eval already done — skipping")
            all_metrics.append(parse_metrics(ft_cmp_txt))
            continue

        stages_csv = os.path.join(
            DATA_DIR, session, "analysis", f"sleep_stages_{session}.csv"
        )
        cmp_csv = os.path.join(
            DATA_DIR, session, "analysis", f"compare_google_{session}.csv"
        )
        cmp_txt = os.path.join(
            DATA_DIR, session, "analysis", f"compare_google_{session}.txt"
        )

        # Back up baseline comparison files so we can restore after
        backups = {}
        for p in [stages_csv, cmp_csv, cmp_txt]:
            if os.path.exists(p):
                backups[p] = p + ".baseline_backup"
                shutil.copy2(p, backups[p])

        try:
            ppg_path = os.path.join(
                DATA_DIR, session, "input", f"sleep_ppg_{session}.csv"
            )
            print(f"  → sleep_staging.py --model-folder {model_dir} {ppg_path}")
            ret = subprocess.run(
                ["uv", "run", "sleep_staging.py", "--model-folder", model_dir, ppg_path],
                cwd=SCRIPT_DIR,
                capture_output=True, text=True,
            )
            if ret.returncode != 0:
                print(f"  sleep_staging.py failed for {session}: {ret.stderr[:500]}")
                continue

            print(f"  → compare_google.py {stages_csv}")
            ret = subprocess.run(
                ["uv", "run", "compare_google.py", "--force", stages_csv],
                cwd=SCRIPT_DIR,
                capture_output=True, text=True,
            )
            if ret.returncode != 0:
                print(f"  compare_google.py failed for {session}: {ret.stderr[:500]}")
                continue

            # Save the fine-tuned comparison with fold suffix
            ft_cmp_csv = os.path.join(
                DATA_DIR, session, "analysis",
                f"compare_google_{session}_ft_fold{fold_idx}.csv"
            )
            ft_cmp_txt = os.path.join(
                DATA_DIR, session, "analysis",
                f"compare_google_{session}_ft_fold{fold_idx}.txt"
            )
            if os.path.exists(cmp_csv):
                shutil.copy2(cmp_csv, ft_cmp_csv)
            if os.path.exists(cmp_txt):
                shutil.copy2(cmp_txt, ft_cmp_txt)

            metrics = parse_metrics(ft_cmp_txt)
            if metrics:
                print(f"  {session}: agreement={metrics.get('agreement','?')}%  "
                      f"kappa={metrics.get('kappa','?')}")
            all_metrics.append(metrics)

        finally:
            for p, backup in backups.items():
                if os.path.exists(backup):
                    shutil.copy2(backup, p)
                    os.unlink(backup)

    # Aggregate metrics across held-out sessions (average)
    if not all_metrics or all(m is None for m in all_metrics):
        return None
    valid = [m for m in all_metrics if m]
    if not valid:
        return None
    avg = {}
    for key in ["epochs", "agreement", "kappa"]:
        vals = [m[key] for m in valid if key in m]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return avg


def parse_metrics(txt_path):
    """Extract agreement and kappa from a compare_google_*.txt report."""
    if not os.path.exists(txt_path):
        return None
    metrics = {}
    with open(txt_path) as f:
        for line in f:
            if "Overall agreement:" in line:
                try:
                    metrics["agreement"] = float(line.split(":")[1].strip().rstrip("%"))
                except (ValueError, IndexError):
                    pass
            elif "Cohen's kappa:" in line:
                try:
                    metrics["kappa"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "Epochs compared:" in line:
                try:
                    metrics["epochs"] = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return metrics if metrics else None


def aggregate_results(run_name, fold_results):
    """Write per-fold results CSV and pick the best model by mean kappa."""
    import csv
    results_csv = os.path.join(FINETUNE_DIR, run_name, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "held_out_session", "epochs", "agreement", "kappa", "model_dir"])
        for r in fold_results:
            writer.writerow([
                r["fold"], r["held_out_session"],
                r.get("epochs", ""), r.get("agreement", ""), r.get("kappa", ""),
                r["model_dir"],
            ])

    kappas = [r["kappa"] for r in fold_results if r.get("kappa") is not None]
    agreements = [r["agreement"] for r in fold_results if r.get("agreement") is not None]

    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete — {len(fold_results)} folds")
    print(f"{'='*60}")
    if kappas:
        import statistics
        mean_k = statistics.mean(kappas)
        std_k = statistics.stdev(kappas) if len(kappas) > 1 else 0.0
        mean_a = statistics.mean(agreements)
        std_a = statistics.stdev(agreements) if len(agreements) > 1 else 0.0
        print(f"  Mean kappa:     {mean_k:.3f} ± {std_k:.3f}")
        print(f"  Mean agreement: {mean_a:.1f}% ± {std_a:.1f}%")
        print(f"  Results: {results_csv}")

        # Pick best by kappa
        best_fold = max(fold_results, key=lambda r: r.get("kappa", -999))
        best_dir = best_fold["model_dir"]
        best_link = os.path.join(MODELS_DIR, f"vigil_finetuned_{run_name}_best")
        if os.path.islink(best_link):
            os.unlink(best_link)
        elif os.path.exists(best_link):
            shutil.rmtree(best_link)
        os.symlink(os.path.abspath(best_dir), best_link)
        print(f"  Best model (fold {best_fold['fold']}, kappa={best_fold['kappa']:.3f}):")
        print(f"    {best_dir}")
        print(f"    → symlinked as {best_link}")

        # Update the canonical 'vigil_finetuned_best' pointer so import-android.py
        # picks up this run automatically. Only do this if the canonical link
        # doesn't exist or already points at a previous run (don't clobber a
        # user's explicit choice silently — print what we did).
        canonical = os.path.join(MODELS_DIR, "vigil_finetuned_best")
        prev_target = os.path.realpath(canonical) if os.path.islink(canonical) else None
        if prev_target != os.path.abspath(best_dir):
            if os.path.islink(canonical):
                os.unlink(canonical)
            elif os.path.exists(canonical):
                shutil.rmtree(canonical)
            os.symlink(os.path.abspath(best_dir), canonical)
            if prev_target:
                print(f"  Updated active model: {prev_target} → {best_dir}")
            else:
                print(f"  Set as active model: {canonical} → {best_dir}")
            print(f"  (import-android.py and sleep_staging.py will now use this model)")
        if mean_k < 0:
            print(f"\n  Warning: mean kappa < 0 — fine-tuning did not help.")
            print(f"  Consider collecting more paired nights (target: 10+).")
    else:
        print(f"  No valid metrics — check eval output for errors.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune wav2sleep on paired vigil + Google nights (LOO-CV)."
    )
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name of the prepared run (from prepare_finetune_data.py)")
    parser.add_argument("--min-nights", type=int, default=MIN_NIGHTS_IDEAL,
                        help=f"Minimum paired nights required (default: {MIN_NIGHTS_IDEAL}; "
                             f"5 for meaningful band, 0 to disable gate)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max training epochs per fold (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4 — small for few nights)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run folds even if model already exists")
    parser.add_argument("--force-eval", action="store_true",
                        help="Re-run held-out evaluation even if results exist")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed folds (alias for not passing --force)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and print what would run, but don't train")
    args = parser.parse_args()

    # Self-re-exec into wav2sleep env if needed
    reexec_in_wav2sleep_env()

    print("╔══════════════════════════════════════════╗")
    print("║  vigil — wav2sleep fine-tuning            ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Run name:  {args.run_name}")
    print(f"  Python:    {sys.executable}")

    check_dependencies()
    check_cuda()

    # Load manifest
    manifest = load_run_manifest(args.run_name)
    n_nights = manifest["num_sessions"]
    cv_strategy = manifest.get("cv_strategy", "loo")
    n_folds = manifest["num_folds"]
    print(f"\n  Paired nights: {n_nights}")
    print(f"  CV strategy:   {cv_strategy} ({n_folds} folds)")

    # Gating
    if args.min_nights > 0 and n_nights < args.min_nights:
        print(f"\n  Error: only {n_nights} paired nights available, minimum is {args.min_nights}.")
        if args.min_nights > MIN_NIGHTS_MEANINGFUL:
            print(f"  - {MIN_NIGHTS_IDEAL}+ nights is the 'ideal' default (see README)")
            print(f"  - {MIN_NIGHTS_MEANINGFUL}+ nights can be enabled with: --min-nights {MIN_NIGHTS_MEANINGFUL}")
        print(f"  - Fewer nights for testing: --min-nights N (N>=1)")
        print(f"  - Disable gate: --min-nights 0")
        print(f"  - Collect more via: import-android.py + fetch_google_sleep.py + compare_google.py")
        sys.exit(1)

    if MIN_NIGHTS_MEANINGFUL <= n_nights < MIN_NIGHTS_IDEAL:
        print(f"\n  Note: {n_nights} paired nights — fine-tuning may run but {MIN_NIGHTS_IDEAL}+ is ideal.")
        print(f"  Results may not generalize; treat as experimental until {MIN_NIGHTS_IDEAL}+ are collected.")

    if n_nights < MIN_NIGHTS_MEANINGFUL:
        train_per_fold = len(manifest["folds"][0]["train_sessions"]) if manifest["folds"] else n_nights - 1
        print(f"\n  Warning: {cv_strategy.upper()} with N={n_nights} — each fold trains on {train_per_fold} nights.")
        print(f"  Per-fold metrics will be highly noisy; treat the mean as indicative only.")

    # Plan folds
    run_dir = os.path.join(FINETUNE_DIR, args.run_name)
    completed = get_completed_folds(run_dir) if args.resume or not args.force else set()
    if args.resume and completed:
        print(f"\n  Resuming — {len(completed)} fold(s) already complete: {sorted(completed)}")

    folds_to_run = [f for f in manifest["folds"] if f["fold"] not in completed]

    if args.dry_run:
        print(f"\n  --dry-run: would run {len(folds_to_run)} fold(s):")
        for f in folds_to_run:
            held = ", ".join(f["held_out_sessions"])
            print(f"    fold {f['fold']}: held_out={held}, "
                  f"train={len(f['train_sessions'])} sessions")
        return

    if not folds_to_run:
        print(f"\n  All folds already complete. Use --force to re-run.")
        # Still aggregate existing results
        fold_results = collect_existing_results(manifest, args.run_name)
        if fold_results:
            aggregate_results(args.run_name, fold_results)
        return

    # Run each fold
    fold_results = []
    for fold in folds_to_run:
        model_dir = run_fold(fold, manifest, args.run_name, args)
        # Evaluate on held-out session(s)
        metrics = evaluate_held_out(
            model_dir, fold["held_out_sessions"], args.run_name, fold["fold"],
            force_eval=args.force_eval,
        )
        fold_results.append({
            "fold": fold["fold"],
            "held_out_session": ", ".join(fold["held_out_sessions"]),
            "epochs": metrics.get("epochs") if metrics else None,
            "agreement": metrics.get("agreement") if metrics else None,
            "kappa": metrics.get("kappa") if metrics else None,
            "model_dir": model_dir,
        })

    # Also include previously-completed folds (from --resume) in aggregation
    if completed:
        prior = collect_existing_results(manifest, args.run_name, exclude=set(f["fold"] for f in folds_to_run))
        fold_results = prior + fold_results

    # Aggregate
    aggregate_results(args.run_name, fold_results)


def collect_existing_results(manifest, run_name, exclude=None):
    """Read metrics from existing _ft_fold<n>.txt files for completed folds."""
    results = []
    exclude = exclude or set()
    for fold in manifest["folds"]:
        if fold["fold"] in exclude:
            continue
        # For kfold, a fold may hold out multiple sessions — aggregate metrics
        # from all held-out sessions' _ft_fold<n>.txt files.
        all_metrics = []
        for session in fold["held_out_sessions"]:
            ft_txt = os.path.join(
                DATA_DIR, session, "analysis",
                f"compare_google_{session}_ft_fold{fold['fold']}.txt"
            )
            m = parse_metrics(ft_txt)
            if m:
                all_metrics.append(m)
        if all_metrics:
            avg = {}
            for key in ["epochs", "agreement", "kappa"]:
                vals = [m[key] for m in all_metrics if key in m]
                if vals:
                    avg[key] = sum(vals) / len(vals)
            metrics = avg
        else:
            metrics = None
        model_dir = os.path.join(MODELS_DIR, f"vigil_finetuned_{run_name}_fold{fold['fold']}")
        results.append({
            "fold": fold["fold"],
            "held_out_session": ", ".join(fold["held_out_sessions"]),
            "epochs": metrics.get("epochs") if metrics else None,
            "agreement": metrics.get("agreement") if metrics else None,
            "kappa": metrics.get("kappa") if metrics else None,
            "model_dir": model_dir,
        })
    return results


if __name__ == "__main__":
    main()