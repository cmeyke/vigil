# vigil

Self-hosted, cloud-free sleep tracking using the Polar Verity Sense optical
heart rate sensor. Records raw PPG (55 Hz) + accelerometer (52 Hz) overnight,
classifies sleep stages (Wake/Light/Deep/REM) using open-source ML, and
generates a hypnogram plot — all on your own machine.

No cloud, no account, no subscription. The Polar Flow app is only needed for
initial sensor setup and firmware updates.

## Hardware

- **Polar Verity Sense** (optical HR armband, BLE)
- Worn on forearm for sleep, upper arm for gym
- Firmware 3.0.16+

## Two recording methods

### Method 1: Android app (recommended)

The companion Android app ([vigil-android](https://github.com/cmeyke/vigil-android))
streams PPG + ACC directly to CSV files on the phone. No computer needed
overnight — the phone is the nightstand recorder.

1. Install the vigil APK on your phone
2. Pair the Verity Sense in the app
3. Press **Start sleep recording** before bed
4. Press **Stop** in the morning
5. Import the recordings to your computer:

```bash
# Preview new sessions on a connected phone:
uv run import-android.py --dry-run

# Import new sessions into data/<timestamp>/input/:
uv run import-android.py
```

The script uses `adb` to scan `Documents/vigil/<timestamp>/` on the phone,
copies any sessions not already present locally, and prompts to run
`sleep_staging.py` + `plot_hypnogram.py` on the freshly imported recording.
Set the adb path via `ADB=/path/to/adb` or `--adb PATH` if it's not on `PATH`.

### Method 2: Python live streaming

A computer within BLE range (~10 m) streams PPG + ACC overnight via Python.

```bash
# Pair the sensor first (one-time, Linux only):
bluetoothctl pair <MAC_ADDRESS>
bluetoothctl trust <MAC_ADDRESS>

# Record overnight (8 hours):
uv run record_sleep.py --hours 8

# Or stop early with Ctrl+C
```

Files are saved to `data/<timestamp>/input/`:
- `sleep_ppg_<timestamp>.csv` — 55 Hz PPG, 4 channels (ch0–ch3), timestamp in ns
- `sleep_acc_<timestamp>.csv` — 52 Hz ACC, 3 axes (x_mg, y_mg, z_mg), timestamp in ns

## Sleep analysis workflow

### Step 1: Run sleep staging

```bash
uv run sleep_staging.py data/<timestamp>/input/sleep_ppg_<timestamp>.csv
```

This converts vigil's 4-channel PPG to wav2sleep's format, runs inference, and
saves predictions to `data/<timestamp>/analysis/sleep_stages_<timestamp>.csv`.

Output: per-epoch (30 s) predictions with columns `Timestamp` (seconds from
start) and `Pred` (0=Wake, 1=Light, 2=Deep, 3=REM).

### Step 2: Generate hypnogram plot

```bash
uv run plot_hypnogram.py data/<timestamp>/analysis/sleep_stages_<timestamp>.csv
```

Creates a dark-themed hypnogram PNG at `data/<timestamp>/analysis/hypnogram_<timestamp>.png`
with:
- Stepped horizontal line segments per sleep stage (no vertical connectors)
- Clock time on x-axis (parsed from the timestamp in the filename)
- Title with total sleep time and sleep efficiency
- Legend with per-stage hours and percentages (excludes leading/trailing
  Awake from statistics)

### Step 3 (optional): Analyze HR/HRV

```bash
uv run analyze_ppg.py
```

Auto-finds the latest recording in `data/*/input/sleep_ppg_*.csv` and produces:
- Heart rate (BPM) with motion-aware peak detection
- HRV metrics (SDNN, RMSSD, pNN50) with RR outlier rejection
- 4-panel plot saved to `data/<timestamp>/analysis/sleep_analysis_<timestamp>.png`
- RR intervals saved to `data/<timestamp>/analysis/sleep_rr_<timestamp>.csv`
- Results summary saved to `data/<timestamp>/analysis/sleep_results_<timestamp>.txt`

Skips recordings that already have all three outputs; pass `--force` to re-run.

## Google Health integration

Fetch sleep stages from Google's Pixel Watch / Fitbit data for comparison
and fine-tuning ground truth labels.

### Setup (one-time)

1. Create a Google Cloud project and enable the **Google Health API**
2. Create OAuth 2.0 credentials (Desktop app type)
3. Download `client_secret_*.json` and save as `google_credentials.json`
4. Add yourself as a test user on the OAuth consent screen
5. Add the `googlehealth.sleep.readonly` scope

### Fetch sleep data

```bash
# Last 7 days:
uv run fetch_google_sleep.py --days 7

# Last 90 days:
uv run fetch_google_sleep.py --days 90

# Since a specific date:
uv run fetch_google_sleep.py --since 2026-07-01
```

First run opens a browser for OAuth authorization. Subsequent runs use the
cached token (`google_token.json`). Token expires after 7 days in testing
mode — re-run to re-authorize.

Output: `data/google_sleep/google_sleep_<start>_to_<end>.csv` with per-stage
segments (variable length, not fixed 30 s epochs). Google's API exposes
`STAGES` type only (AWAKE/LIGHT/DEEP/REM), not the RESTLESS label visible
in the app.

### Compare wav2sleep against Google

```bash
# Latest session with a sleep_stages CSV:
uv run compare_google.py

# Specific session:
uv run compare_google.py data/20260711_231213/analysis/sleep_stages_20260711_231213.csv

# All sessions:
uv run compare_google.py --all
```

Aligns each 30 s vigil epoch (by absolute UTC time, using the recording's
`timestamp_ns`) with the overlapping Google interval and reports:

- Confusion matrix (rows=Google, cols=wav2sleep)
- Per-stage precision / recall / F1
- Overall agreement and Cohen's kappa
- Stage distribution comparison (Google vs wav2sleep)

Output: `data/<timestamp>/analysis/compare_google_<timestamp>.{csv,txt}`.
The CSV has one row per epoch with both labels; the TXT is the printed
report. Recordings without Google overlap are skipped. Requires the
timestamp fix from `import-android.py` (recordings made with old app
versions are auto-normalized on import).

## Fine-tuning

wav2sleep over-predicts Wake (~20% vs Google's ~7%) because it was trained
on clinical PSG data, not Verity Sense PPG. Fine-tuning on paired nights
(vigil PPG + Google labels) is the fix.

- Each paired night = one training sample (PPG input + Google's stage labels)
- Fine-tuning requires **10+ paired nights by default** (the "ideal" threshold)
- 5+ nights can be enabled with `--min-nights 5` (the "meaningful" threshold)
- Fewer nights can be used for pipeline testing with `--min-nights N`
- Strategy: freeze signal encoders + epoch mixer, train only sequence mixer
  + classifier at low LR (conservative — adapts the head to PPG-domain
  features without destroying learned representations)
- Evaluation: cross-validation. Strategy auto-selected by night count:
  - **N < 30**: leave-one-out (N folds, each trains on N-1, eval on 1)
  - **N ≥ 30**: 10-fold (each trains on 90% of nights, eval on 10%)
  - Override with `--cv {loo,kfold}` and `--folds N`
- wav2sleep's `SleepLightningModule` is used directly from a checkout of the
  [wav2sleep repo](https://github.com/joncarter1/wav2sleep) — vigil stays
  pure-Python with no torch dependency in its main venv
- Currently collecting paired nights

### Prerequisites (one-time)

```bash
# Create the wav2sleep venv (separate from vigil's venv due to numpy conflict)
cd ~/code/python/ai
uv venv wav2sleep-env --python 3.12
uv pip install --python wav2sleep-env/.venv/bin/python \
    "git+https://github.com/joncarter1/wav2sleep.git" \
    lightning hydra-core mlflow
```

A CUDA-capable GPU is required for fine-tuning. The data-prep step
(`prepare_finetune_data.py`) does not need GPU and runs in vigil's venv.

### Workflow

```bash
# 1. Collect paired nights: import vigil PPG + fetch Google labels
uv run import-android.py
uv run fetch_google_sleep.py --days 30
uv run compare_google.py --all          # produces compare_google_<ts>.csv per night

# 2. Prepare parquet training data (CV folds — auto: LOO if N<30, 10-fold otherwise)
uv run prepare_finetune_data.py --run-name myrun
# Override CV strategy:
uv run prepare_finetune_data.py --run-name myrun --cv kfold --folds 5

# 3. Fine-tune (default needs 10+ nights; override for testing)
uv run finetune.py --run-name myrun                    # default gate: 10+ nights
uv run finetune.py --run-name myrun --min-nights 5     # meaningful band
uv run finetune.py --run-name myrun --min-nights 3     # testing only
uv run finetune.py --run-name myrun --dry-run          # plan without training
uv run finetune.py --run-name myrun --resume           # skip completed folds

# 4. Use the fine-tuned model for future sleep staging
uv run sleep_staging.py data/<ts>/input/sleep_ppg_<ts>.csv \
    --model-folder data/models/vigil_finetuned_myrun_best
```

The fine-tuned model is saved as `data/models/vigil_finetuned_<run>_best/`
(a `config.yaml` + `state_dict.pth` folder, same format as wav2sleep's HF
release). `import-android.py` auto-detects it and uses it for the post-import
sleep staging prompt if present.

Per-fold held-out metrics are saved to `data/finetune/<run>/results.csv`.
The best model (highest mean Cohen's kappa across folds) is symlinked as
`data/models/vigil_finetuned_<run>_best/`. `finetune.py` also updates a
canonical `data/models/vigil_finetuned_best` symlink to point at the most
recently trained run — this is what `import-android.py` uses by default.

### Selecting which fine-tuned model is active

If you have multiple runs (`v1`, `v2`, ...) and want to switch which one
`import-android.py` uses, repoint the canonical symlink:

```bash
# See available runs:
ls -l data/models/vigil_finetuned_*_best/

# Point the active-model symlink at a different run:
ln -sfn vigil_finetuned_v2_best data/models/vigil_finetuned_best

# Or use a specific run directly without touching the symlink:
uv run sleep_staging.py data/<ts>/input/sleep_ppg_<ts>.csv \
    --model-folder data/models/vigil_finetuned_v1_best

# Or use the original wav2sleep model (overrides any fine-tuned symlink):
uv run import-android.py --base-model
```

## Data directory structure

```
data/
├── <timestamp>/
│   ├── input/                              ← raw data (don't modify)
│   │   ├── sleep_ppg_<timestamp>.csv       (55 Hz PPG, 4ch, ~59 MB for 7h)
│   │   └── sleep_acc_<timestamp>.csv       (52 Hz ACC, 3-axis, ~41 MB for 7h)
│   └── analysis/                           ← everything derived
│       ├── sleep_stages_<timestamp>.csv    (wav2sleep predictions)
│       ├── hypnogram_<timestamp>.png        (hypnogram plot)
│       ├── sleep_analysis_<timestamp>.png   (HR/HRV plot)
│       ├── sleep_results_<timestamp>.txt    (HR/HRV results summary)
│       ├── sleep_rr_<timestamp>.csv         (RR intervals)
│       ├── compare_google_<timestamp>.csv   (per-epoch Google vs wav2sleep)
│       └── compare_google_<timestamp>.txt   (confusion matrix + metrics)
├── finetune/
│   └── <run_name>/
│       ├── sessions/*.parquet                # one per night (written once)
│       ├── fold_0/{train,val}/*.parquet       # symlinks to sessions/
│       ├── fold_1/{train,val}/*.parquet
│       └── folds.json                         # manifest: CV strategy + fold membership
├── models/
│   └── vigil_finetuned_<run_name>_best/      (fine-tuned model: config.yaml + state_dict.pth)
└── google_sleep/
    └── google_sleep_<date>_to_<date>.csv    (Google Health API data)
```

The `data/` directory is gitignored (contains personal health data).

## Scripts

| Script | Description |
|---|---|
| `record_sleep.py` | Overnight BLE recording (PPG + ACC → CSV) with auto-reconnect, `--hours N` auto-stop |
| `import-android.py` | Pull new sleep recordings from a connected Android phone via `adb` (skips already-imported sessions) |
| `sleep_staging.py` | PPG → wav2sleep inference → sleep stage predictions CSV. `--model-folder` selects base vs fine-tuned model |
| `plot_hypnogram.py` | Sleep stages CSV → dark-themed hypnogram PNG |
| `analyze_ppg.py` | PPG → heart rate + HRV metrics + 4-panel plot |
| `fetch_google_sleep.py` | Google Health API → sleep stages + nightly HRV CSV (for comparison/ground truth) |
| `compare_google.py` | Align wav2sleep predictions with Google sleep stages → confusion matrix, per-stage P/R/F1, kappa |
| `compare-hrv.py` | Match vigil RMSSD against Google nightly HRV → side-by-side table, Pearson r, scatter plot |
| `prepare_finetune_data.py` | Convert paired nights to wav2sleep parquet format (LOO-CV folds) |
| `finetune.py` | Fine-tune wav2sleep on paired nights (LOO-CV, conservative freeze). Runs in wav2sleep venv |
| `live_hr.py` | Real-time HR + HRV terminal display with rolling buffer and sparkline |
| `record_test.py` | 30-second test recording (for verifying sensor connectivity) |
| `scan.py` | BLE scanner — find Polar devices by name |
| `sensor.py` | Shared sensor discovery module (finds by name, not MAC — cross-platform) |
| `stream_live.py` | Live PPG + ACC + PPI + HR streaming (debugging/testing) |

## Tech stack

- Python 3.12, [uv](https://docs.astral.sh/uv/) (package manager)
- [polar-python](https://github.com/polarofficial/polar-python-sdk) (BLE communication)
- [wav2sleep](https://github.com/joncarter1/wav2sleep) (sleep staging model)
- numpy, scipy, pandas (signal processing)
- matplotlib (visualization)
- Google Health API v4 (sleep data import)

## Architecture

```
Polar Verity Sense (forearm)
  ↓ live BLE streaming (PPG 55Hz + ACC 52Hz)
  ↓
Android app or Python record_sleep.py
  ↓
data/<timestamp>/input/*.csv (raw PPG + ACC)
  ↓
sleep_staging.py → wav2sleep → sleep stages CSV
  ↓
plot_hypnogram.py → hypnogram PNG
analyze_ppg.py → HR/HRV metrics + plot
  ↓
fetch_google_sleep.py → Google ground-truth labels (for fine-tuning)
  ↓
All data stays local — your data, your machine
```

## Getting started

```bash
git clone https://github.com/cmeyke/vigil.git
cd vigil
uv sync

# Optional: install wav2sleep in a separate venv (numpy version conflict)
cd ~/code/python/ai
uv venv wav2sleep-env --python 3.12
cd wav2sleep-env
uv pip install "git+https://github.com/joncarter1/wav2sleep.git" --python .venv/bin/python

# Find your sensor:
cd ../vigil
uv run scan.py

# Record overnight:
uv run record_sleep.py --hours 8

# Analyze:
uv run sleep_staging.py data/<timestamp>/input/sleep_ppg_<timestamp>.csv
uv run plot_hypnogram.py data/<timestamp>/analysis/sleep_stages_<timestamp>.csv
```

## License

vigil is licensed under the [GNU Affero General Public License v3.0](LICENSE).

For commercial use without copyleft obligations (e.g., embedding vigil in a
proprietary product or SaaS without open-sourcing your modifications),
contact the author for a commercial license.