# vigil

Self-hosted, cloud-free sleep and strength training analysis pipeline using the
Polar Verity Sense optical heart rate sensor.

## What it does

- **Sleep analysis**: Records raw PPG (55 Hz) + accelerometer (52 Hz) from the
  Polar Verity Sense overnight, then classifies sleep stages (Wake/Light/Deep/REM)
  using open-source ML models.
- **Strength training**: Records HR zones, rep counts (from accelerometer), and
  HR recovery between sets during barbell training sessions.
- **No cloud, no account, no subscription**: All data stays on your machine.
  The Polar Flow app is only used for initial sensor setup and firmware updates.

## Hardware

- Polar Verity Sense (optical HR armband, BLE + ANT+)
- Worn on forearm (sleep) or upper arm (gym)
- Firmware 3.0.16 (SDK mode supported)

## Architecture

```
Verity Sense (upper arm / forearm)
  ↓ offline recording (PPG 55Hz + ACC 52Hz)
  ↓
Fetch via BLE → Python (vigil)
  ↓
Sleep: sleep staging model → stages, HRV, breathing rate
Gym: rep count + exercise classify + HR zones + recovery
  ↓
Local storage (SQLite/CSV) — your data, your machine
  ↓
AI/LLM for interpretation, trends, recommendations
```

## Tech stack

- Python 3.12, uv
- bleak + bleakheart (BLE communication)
- numpy, scipy, pandas (signal processing & analysis)
- matplotlib (visualization)
- Open-source sleep staging models (wav2sleep, SleepPPGNet — to be integrated)

## Getting started

```bash
cd ~/code/python/ai/vigil
uv sync
uv run main
```

## License

vigil is licensed under the [GNU Affero General Public License v3.0](LICENSE).

For commercial use without copyleft obligations (e.g., embedding vigil in a
proprietary product or SaaS without open-sourcing your modifications),
contact the author for a commercial license.