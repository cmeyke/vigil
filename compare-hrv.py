"""
vigil — compare vigil RMSSD against Google Health nightly HRV

For every vigil recording with a sleep_results_<ts>.txt (or sleep_rr_<ts>.csv),
match it against the corresponding night in the Google HRV data by date, and
report side-by-side RMSSD values. Computes Pearson correlation across all
paired nights and saves a scatter plot + CSV.

The vigil session timestamp (directory name YYYYMMDD_HHMMSS) is interpreted
as the local start time of the recording; the matching Google HRV row is the
one whose Date equals the recording's date. (Google's HRV row for date D
reflects the sleep of the night beginning on D — which matches a vigil
recording started on the evening of D.)

Usage:
    uv run compare-hrv.py                          # compare all paired nights
    uv run compare-hrv.py --force                  # re-run even if outputs exist

Output:
    data/hrv_comparison.csv   (one row per paired night)
    data/hrv_comparison.png  (scatter: vigil RMSSD vs Google RMSSD)
"""

import argparse
import glob
import os
import re
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
GOOGLE_DIR = os.path.join(DATA_DIR, "google_sleep")

OUTPUT_CSV = os.path.join(DATA_DIR, "hrv_comparison.csv")
OUTPUT_PNG = os.path.join(DATA_DIR, "hrv_comparison.png")


def is_timestamp_dir(name):
    """True if name looks like a vigil session timestamp (YYYYMMDD_HHMMSS)."""
    return (
        len(name) == 15
        and name[8] == "_"
        and name[:8].isdigit()
        and name[9:].isdigit()
    )


def session_date(session):
    """Return the date (YYYY-MM-DD string) a vigil recording belongs to.

    The session dir is named YYYYMMDD_HHMMSS in local time. We treat the
    recording as belonging to the date it started in (the evening). Google's
    HRV row for date D describes the night beginning on D, so the two align.
    """
    return f"{session[:4]}-{session[4:6]}-{session[6:8]}"


def load_google_hrv():
    """Load all google_hrv_*.csv files in data/google_sleep/.

    Returns a DataFrame indexed by Date (string) with columns
    RMSSD_ms, DeepSleep_RMSSD_ms, NonREM_HR_bpm, Entropy.
    """
    files = sorted(glob.glob(os.path.join(GOOGLE_DIR, "google_hrv_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No google_hrv_*.csv found in {GOOGLE_DIR}. "
            f"Run: uv run fetch_google_sleep.py"
        )
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    g = pd.concat(frames, ignore_index=True)
    g = g.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return g.set_index("Date")


def parse_vigil_rmssd_from_txt(txt_path):
    """Extract RMSSD (ms) from a sleep_results_<ts>.txt file. Returns None if missing/unparsed."""
    if not os.path.exists(txt_path):
        return None
    with open(txt_path) as f:
        for line in f:
            m = re.match(r"\s*RMSSD:\s*([\d.]+)\s*ms", line)
            if m:
                return float(m.group(1))
    return None


def compute_rmssd_from_rr_csv(rr_path):
    """Compute RMSSD = sqrt(mean(diff(rr)^2)) from a sleep_rr_<ts>.csv file."""
    if not os.path.exists(rr_path):
        return None
    try:
        df = pd.read_csv(rr_path)
        if "rr_ms" not in df.columns or len(df) < 2:
            return None
        rr = df["rr_ms"].values
        diff = np.diff(rr)
        return float(np.sqrt(np.mean(diff ** 2)))
    except (OSError, ValueError, KeyError):
        return None


def collect_vigil_rmssd():
    """Walk data/<session>/analysis/ for sleep_results / sleep_rr files.

    Returns a list of dicts: {session, date, rmssd_ms, source}.
    Prefers the RMSSD printed in sleep_results_*.txt; falls back to computing
    from sleep_rr_*.csv directly.
    """
    out = []
    for session_dir in sorted(os.listdir(DATA_DIR)):
        if not is_timestamp_dir(session_dir):
            continue
        analysis_dir = os.path.join(DATA_DIR, session_dir, "analysis")
        if not os.path.isdir(analysis_dir):
            continue
        txt_path = os.path.join(analysis_dir, f"sleep_results_{session_dir}.txt")
        rr_path = os.path.join(analysis_dir, f"sleep_rr_{session_dir}.csv")

        rmssd = parse_vigil_rmssd_from_txt(txt_path)
        source = "sleep_results.txt"
        if rmssd is None:
            rmssd = compute_rmssd_from_rr_csv(rr_path)
            source = "sleep_rr.csv (computed)"
        if rmssd is None:
            continue
        out.append({
            "session": session_dir,
            "date": session_date(session_dir),
            "rmssd_ms": rmssd,
            "source": source,
        })
    return out


def correlation(x, y):
    """Pearson correlation, returns (r, p) — NaN-safe for n<2."""
    if len(x) < 2:
        return float("nan"), float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    # Avoid scipy dependency for p-value; report r only
    return r, float("nan")


def build_report(paired, r_pearson):
    lines = []
    lines.append("=" * 60)
    lines.append("  vigil vs Google — nightly HRV (RMSSD)")
    lines.append("=" * 60)
    lines.append(f"  Paired nights:    {len(paired)}  (implausible vigil values excluded)")
    if not paired.empty:
        lines.append(f"  Pearson r:        {r_pearson:.3f}")
        lines.append("")
        lines.append(f"  {'Date':12s}  {'vigil':>8s}  {'Google':>8s}  {'Δ':>8s}")
        for _, row in paired.iterrows():
            delta = row["vigil_rmssd_ms"] - row["google_rmssd_ms"]
            lines.append(
                f"  {row['date']:12s}  {row['vigil_rmssd_ms']:8.1f}  "
                f"{row['google_rmssd_ms']:8.1f}  {delta:8.1f}"
            )
        # Averages
        v_mean = paired["vigil_rmssd_ms"].mean()
        g_mean = paired["google_rmssd_ms"].mean()
        lines.append("")
        lines.append(f"  {'Mean':12s}  {v_mean:8.1f}  {g_mean:8.1f}  {v_mean - g_mean:8.1f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def plot_scatter(paired, output_path):
    """Scatter vigil RMSSD vs Google RMSSD with identity line."""
    fig, ax = plt.subplots(figsize=(7, 7))

    if not paired.empty:
        ax.scatter(
            paired["google_rmssd_ms"],
            paired["vigil_rmssd_ms"],
            s=80,
            color="#4A9EFF",
            edgecolor="white",
            zorder=3,
        )
        # Label each point with its date
        for _, row in paired.iterrows():
            ax.annotate(
                row["date"][5:],  # MM-DD
                (row["google_rmssd_ms"], row["vigil_rmssd_ms"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color="#888",
            )

    # Identity line spanning the union of both axes
    all_vals = []
    if not paired.empty:
        all_vals = pd.concat(
            [paired["google_rmssd_ms"], paired["vigil_rmssd_ms"]]
        )
    if len(all_vals):
        lo = float(all_vals.min()) * 0.9
        hi = float(all_vals.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "--", color="#FF3B6B", alpha=0.6,
                label="identity (vigil = Google)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("Google RMSSD (ms)")
    ax.set_ylabel("vigil RMSSD (ms)")
    ax.set_title("Nightly HRV — vigil (Verity Sense) vs Google (Pixel Watch)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare vigil RMSSD against Google Health nightly HRV."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if data/hrv_comparison.csv exists")
    args = parser.parse_args()

    if not args.force and os.path.exists(OUTPUT_CSV):
        print(f"Already compared: {OUTPUT_CSV}")
        print("(use --force to re-run)")
        return

    # Load Google HRV (one row per night, indexed by Date)
    try:
        google = load_google_hrv()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Collect vigil RMSSD per session
    vigil = collect_vigil_rmssd()
    if not vigil:
        print("No vigil RMSSD found. Run analyze_ppg.py on at least one session first:")
        print("  uv run analyze_ppg.py --all")
        sys.exit(1)

    print(f"  vigil sessions with RMSSD:    {len(vigil)}")
    print(f"  Google nights with HRV:        {len(google)}")

    # Match by date
    rows = []
    for v in vigil:
        if v["date"] in google.index:
            g = google.loc[v["date"]]
            rows.append({
                "date": v["date"],
                "session": v["session"],
                "vigil_rmssd_ms": v["rmssd_ms"],
                "google_rmssd_ms": float(g["RMSSD_ms"]) if pd.notna(g["RMSSD_ms"]) else float("nan"),
                "google_deep_rmssd_ms": float(g["DeepSleep_RMSSD_ms"]) if pd.notna(g.get("DeepSleep_RMSSD_ms")) else float("nan"),
                "google_nonrem_hr_bpm": g.get("NonREM_HR_bpm"),
                "google_entropy": g.get("Entropy"),
                "source": v["source"],
            })
        else:
            print(f"  {v['date']} ({v['session']}) — no matching Google HRV row")

    if not rows:
        print("\nNo paired nights found.")
        print("Ensure fetch_google_sleep.py has been run for the relevant dates.")
        sys.exit(0)

    paired = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Flag implausible vigil RMSSD values (likely from broken peak detection —
    # a healthy overnight RMSSD is in the 10-200 ms range). These are kept in
    # the CSV for traceability but excluded from the correlation and plot.
    RMSSD_PLAUSIBLE_MIN = 5.0
    RMSSD_PLAUSIBLE_MAX = 500.0
    paired["plausible"] = (
        (paired["vigil_rmssd_ms"] >= RMSSD_PLAUSIBLE_MIN)
        & (paired["vigil_rmssd_ms"] <= RMSSD_PLAUSIBLE_MAX)
    )
    outliers = paired[~paired["plausible"]]
    for _, row in outliers.iterrows():
        print(f"  ! {row['date']} vigil RMSSD {row['vigil_rmssd_ms']:.1f} ms is implausible "
              f"(outside {RMSSD_PLAUSIBLE_MIN:.0f}-{RMSSD_PLAUSIBLE_MAX:.0f} ms) — "
              f"excluded from correlation/plot")
    paired_plausible = paired[paired["plausible"]].reset_index(drop=True)

    # Pearson correlation across plausible paired nights
    r, _ = correlation(
        paired_plausible["vigil_rmssd_ms"].values,
        paired_plausible["google_rmssd_ms"].values,
    )

    report = build_report(paired_plausible, r)
    print()
    print(report)

    # Save CSV (all rows including outliers, with plausibility flag)
    paired.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved: {OUTPUT_CSV}")

    # Scatter plot (plausible only)
    plot_scatter(paired_plausible, OUTPUT_PNG)
    print(f"  Plot:  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()