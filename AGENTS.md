# AGENTS.md — vigil project guide for opencode sessions

## Quick orientation

vigil is a self-hosted sleep tracking pipeline. Recordings are PPG + ACC CSV
files from a Polar Verity Sense armband, organized under `data/<timestamp>/`.
The analysis pipeline produces sleep stage predictions, hypnograms, and
HR/HRV metrics. All `data/` is gitignored (personal health data).

## Common commands

### Lint / typecheck
vigil has no configured linter or typechecker. Verify a script parses with:
```bash
uv run python -c "import ast; ast.parse(open('<script>.py').read()); print('OK')"
```

### Recording → analysis (one session)
```bash
uv run import-android.py            # pulls new sessions from phone, may auto-run analysis
uv run sleep_staging.py data/<ts>/input/sleep_ppg_<ts>.csv
uv run plot_hypnogram.py data/<ts>/analysis/sleep_stages_<ts>.csv
uv run analyze_ppg.py               # auto-picks latest un-analyzed session
```

### Backfill missing analysis across all sessions
```bash
uv run analyze_ppg.py --all
uv run compare_google.py --all
```

### Google Health comparison
```bash
uv run fetch_google_sleep.py --days 30      # fetches sleep stages + nightly HRV
uv run compare_google.py                   # compares latest un-compared vs Google
uv run compare-hrv.py                       # vigil RMSSD vs Google nightly HRV
```

### Fine-tuning pipeline
```bash
uv run prepare_finetune_data.py --run-name myrun      # vigil's venv
uv run finetune.py --run-name myrun --min-nights 4    # self-re-execs into wav2sleep venv
```

## Two Python environments

vigil uses **two separate venvs** because of a numpy version conflict:

- **vigil venv** (`.venv/`): the main venv. Used for everything except
  wav2sleep inference and fine-tuning. Run scripts with `uv run <script>.py`.
- **wav2sleep venv** (`~/code/python/ai/wav2sleep-env/.venv/`): contains
  torch, lightning, wav2sleep. Used by:
  - `sleep_staging.py` — shells out via subprocess to `WAV2SLEEP_PYTHON`
    (defined at the top of `sleep_staging.py`)
  - `finetune.py` — self-re-execs into this venv if invoked from vigil's

LSP "could not be resolved" errors for `pandas`, `scipy`, `torch`,
`lightning`, `wav2sleep` are expected — the LSP only sees vigil's venv but
the scripts run correctly via `uv run`.

## Fine-tuning gate thresholds

`finetune.py` enforces a three-tier paired-nights gate:

| Tier | Paired nights N | Default behavior | Override |
|---|---|---|---|
| Optimal | N ≥ 10 | runs | (default) |
| Meaningful | 5 ≤ N < 10 | hard error | `--min-nights 5` |
| Experimental | 1 ≤ N < 5 | hard error | `--min-nights N` (N≥1) |
| Disabled | any | n/a | `--min-nights 0` |

Defaults: `MIN_NIGHTS_IDEAL = 10`, `MIN_NIGHTS_MEANINGFUL = 5` (defined in
`finetune.py`). The README's "## Fine-tuning" section documents these.

## Fine-tuning CV strategy

`prepare_finetune_data.py` auto-selects the cross-validation strategy:

| Paired nights N | CV strategy | Folds | Train per fold | Eval per fold |
|---|---|---|---|---|
| N < 30 | leave-one-out (LOO) | N | N-1 | 1 |
| N ≥ 30 | k-fold | 10 | ~N×9/10 | ~N/10 |

Override with `--cv {loo,kfold}` and `--folds N`. Session parquets are
written ONCE to `data/finetune/<run>/sessions/` and symlinked into fold
`train/`/`val/` dirs — no duplicated data (at 100 nights this saves ~69 GB).

## Data directory layout

```
data/
├── <timestamp>/                  # one recording session (YYYYMMDD_HHMMSS, local time)
│   ├── input/                     # raw PPG + ACC CSVs (don't modify)
│   └── analysis/                  # derived: sleep_stages, hypnogram, HR/HRV, comparisons
├── google_sleep/                  # Google Health API exports
├── finetune/<run_name>/
│   ├── sessions/*.parquet         # one per night (written once)
│   ├── fold_n/{train,val}/*.parquet  # symlinks to sessions/
│   └── folds.json                 # manifest with CV strategy + fold membership
├── models/                        # fine-tuned model checkpoints
├── hrv_comparison.csv             # vigil vs Google HRV comparison (from compare-hrv.py)
└── ...
```

## Polar timestamp convention

Polar sensors report nanoseconds since **2000-01-01** (Polar epoch), not
1970-01-01 (Unix epoch). The offset is `946684800000000000` ns.

- The **Python** `polar-python` SDK adds the offset internally —
  `record_sleep.py` writes correct Unix-epoch timestamps.
- The **Android** SDK does NOT — older vigil-android versions wrote raw
  Polar-epoch timestamps (~30 years in the past). `import-android.py`
  auto-detects and fixes these on import (see `needs_epoch_fix` / `fix_epoch`).

Reference: https://github.com/polarofficial/polar-ble-sdk/blob/master/documentation/TimeSystemExplained.md

## Conventions

- No comments in code unless explicitly requested by the user.
- Match existing code style (argparse, `uv run` invocations, emoji-free).
- `data/` is gitignored — never commit recordings or analysis outputs.
- After editing scripts, verify they parse with the ast check above.
- The user prefers to commit changes explicitly — never commit unless asked.