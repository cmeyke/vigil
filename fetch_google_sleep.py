"""
vigil — fetch Google Health sleep data

Fetches sleep sessions with per-epoch stages (including RESTLESS) from the
Google Health API v4. Saves stages as CSV matching vigil's format for
direct comparison with wav2sleep predictions.

Requires: google_credentials.json (OAuth Desktop App credentials from Google Cloud)
          google_token.json (auto-created on first run via browser OAuth flow)

Usage:
    uv run fetch_google_sleep.py                          # last 7 days
    uv run fetch_google_sleep.py --days 30                # last 30 days
    uv run fetch_google_sleep.py --since 2026-07-01       # since a specific date

Output:
    data/google_sleep/google_sleep_<date>.csv
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta, timezone

import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleAuthRequest

# Google Health API
HEALTH_API_BASE = "https://health.googleapis.com/v4"
SCOPE = "https://www.googleapis.com/auth/googlehealth.sleep.readonly"

# Paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(SCRIPT_DIR, "google_credentials.json")
TOKEN_FILE = os.path.join(SCRIPT_DIR, "google_token.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "google_sleep")

# Google Health sleep stage types → vigil labels
# https://developers.google.com/health/reference/rest/v4/users.dataTypes.dataPoints#Sleep.SleepStageType
STAGE_MAP = {
    "SLEEP_STAGE_TYPE_UNSPECIFIED": -1,  # Unknown
    "AWAKE":   0,   # Wake
    "LIGHT":   1,   # Light
    "DEEP":    2,   # Deep
    "REM":     3,   # REM
    "ASLEEP":  1,   # Generic sleep → treat as Light
    "RESTLESS": 4,  # New label — maps to 4 for now (not in wav2sleep)
}
STAGE_NAMES = {0: "Awake", 1: "Light", 2: "Deep", 3: "REM", 4: "Restless", -1: "Unknown"}


def authenticate():
    """OAuth2 flow with token caching. Manual flow — no PKCE."""
    # Allow http://localhost redirect (oauthlib requires https by default)
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    creds = None

    # Load cached token if exists
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            creds_data = json.load(f)
        # Reconstruct credentials object
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_info(creds_data, [SCOPE])

    # Refresh or run full OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleAuthRequest())
        else:
            # Load client config from credentials file
            with open(CREDENTIALS_FILE) as f:
                client_config = json.load(f)["installed"]

            # Build auth URL manually (no PKCE)
            import urllib.parse
            import secrets
            state = secrets.token_urlsafe(16)
            params = {
                "response_type": "code",
                "client_id": client_config["client_id"],
                "redirect_uri": "http://localhost",
                "scope": SCOPE,
                "state": state,
                "prompt": "consent",
                "access_type": "offline",
            }
            auth_url = client_config["auth_uri"] + "?" + urllib.parse.urlencode(params)

            print("\n" + "=" * 60)
            print("  Open this URL in your browser to authorize:")
            print("=" * 60)
            print(f"\n{auth_url}\n")
            print("After authorizing, you'll be redirected to localhost.")
            print("The page will show an error (can't connect) — that's fine.")
            print("Copy the FULL URL from the browser's address bar")
            print("(it will look like http://localhost/?code=...)")
            print("and paste it below:\n")

            response = input("Paste redirect URL here: ").strip()

            # Parse the code from the redirect URL
            parsed = urllib.parse.urlparse(response)
            query_params = urllib.parse.parse_qs(parsed.query)
            code = query_params.get("code", [None])[0]

            if not code:
                print("Error: no code found in the redirect URL")
                sys.exit(1)

            # Exchange code for tokens manually (no PKCE verifier)
            token_resp = requests.post(
                client_config["token_uri"],
                data={
                    "client_id": client_config["client_id"],
                    "client_secret": client_config["client_secret"],
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": "http://localhost",
                },
                timeout=30,
            )

            if token_resp.status_code != 200:
                print(f"Token exchange error: {token_resp.text}")
                sys.exit(1)

            token_data = token_resp.json()

            # Build credentials object
            from google.oauth2.credentials import Credentials
            creds = Credentials(
                token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                token_uri=client_config["token_uri"],
                client_id=client_config["client_id"],
                client_secret=client_config["client_secret"],
                scopes=[SCOPE],
            )

        # Save token for future runs
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return creds


def fetch_sleep_sessions(creds, start_dt, end_dt):
    """Fetch sleep sessions from Google Health API."""
    headers = {"Authorization": f"Bearer {creds.token}"}

    # List sleep data points — sleep uses interval.end_time filter
    url = f"{HEALTH_API_BASE}/users/me/dataTypes/sleep/dataPoints"

    # AIP-160 filter: sleep sessions ending within our date range
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    filter_expr = f'sleep.interval.end_time >= "{start_str}" AND sleep.interval.end_time < "{end_str}"'

    params = {
        "filter": filter_expr,
        "pageSize": 25,  # max for sleep
    }

    all_datapoints = []
    next_page_token = None

    while True:
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(url, headers=headers, params=params, timeout=30)

        if resp.status_code != 200:
            print(f"API error {resp.status_code}: {resp.text[:500]}")
            sys.exit(1)

        data = resp.json()
        all_datapoints.extend(data.get("dataPoints", []))

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return all_datapoints


def parse_sleep_stages(datapoints):
    """Extract sleep stages from API response into flat epoch list."""
    sessions = []

    for dp in datapoints:
        sleep = dp.get("sleep", {})
        if not sleep:
            continue

        session_interval = sleep.get("interval", {})
        session_start = session_interval.get("startTime", "")
        session_type = sleep.get("type", "")  # CLASSIC or STAGES
        stages = sleep.get("stages", [])

        session_info = {
            "start": session_start,
            "type": session_type,
            "num_stages": len(stages),
        }

        for stage in stages:
            start_str = stage.get("startTime", "")
            end_str = stage.get("endTime", "")
            stage_type = stage.get("type", "SLEEP_STAGE_TYPE_UNSPECIFIED")

            # Parse times
            try:
                stage_start = datetime.fromisoformat(
                    start_str.replace("Z", "+00:00")
                )
                stage_end = datetime.fromisoformat(
                    end_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                continue

            sessions.append({
                "session_start": session_start,
                "stage_start": stage_start,
                "stage_end": stage_end,
                "stage_type": stage_type,
                "sleep_type": session_type,
            })

    return sessions


def save_stages_csv(stages, output_path):
    """Save stages to CSV in a format comparable with vigil's sleep_stages CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "StageType", "StageLabel",
            "StartTime", "EndTime", "Duration_s", "SleepType",
        ])

        for s in stages:
            duration = (s["stage_end"] - s["stage_start"]).total_seconds()
            label = STAGE_MAP.get(s["stage_type"], -1)
            writer.writerow([
                s["stage_start"].strftime("%Y-%m-%dT%H:%M:%S%z"),
                s["stage_type"],
                label,
                s["stage_start"].strftime("%Y-%m-%dT%H:%M:%S%z"),
                s["stage_end"].strftime("%Y-%m-%dT%H:%M:%S%z"),
                f"{duration:.0f}",
                s["sleep_type"],
            ])


def main():
    parser = argparse.ArgumentParser(description="Fetch Google Health sleep data")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days to fetch (default: 7)")
    parser.add_argument("--since", type=str, default=None,
                        help="Fetch since this date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not os.path.exists(CREDENTIALS_FILE):
        print(f"Error: {CREDENTIALS_FILE} not found")
        print("Download OAuth credentials from Google Cloud Console")
        sys.exit(1)

    # Determine date range
    if args.since:
        start_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = datetime.now(timezone.utc) - timedelta(days=args.days)
    end_dt = datetime.now(timezone.utc)

    print(f"Authenticating with Google Health API...")
    creds = authenticate()
    print("  ✓ Authenticated")

    print(f"Fetching sleep sessions from {start_dt.date()} to {end_dt.date()}...")
    datapoints = fetch_sleep_sessions(creds, start_dt, end_dt)
    print(f"  ✓ {len(datapoints)} sleep sessions found")

    if not datapoints:
        print("No sleep data found in this date range")
        sys.exit(0)

    stages = parse_sleep_stages(datapoints)

    if not stages:
        print("No sleep stages found in sessions")
        sys.exit(0)

    # Print summary
    from collections import Counter
    stage_counts = Counter(s["stage_type"] for s in stages)
    sleep_types = Counter(s["sleep_type"] for s in stages)

    print()
    print("=" * 50)
    print("  GOOGLE HEALTH SLEEP DATA")
    print("=" * 50)
    print(f"  Sessions: {len(datapoints)}")
    print(f"  Total stages: {len(stages)}")
    print(f"  Sleep types: {dict(sleep_types)}")
    print()
    print("  Stage breakdown:")
    for stage_type in ["AWAKE", "LIGHT", "DEEP", "REM", "RESTLESS", "ASLEEP"]:
        count = stage_counts.get(stage_type, 0)
        if count > 0:
            total_s = sum(
                (s["stage_end"] - s["stage_start"]).total_seconds()
                for s in stages if s["stage_type"] == stage_type
            )
            hours = total_s / 3600
            label = STAGE_NAMES.get(STAGE_MAP.get(stage_type, -1), stage_type)
            print(f"    {stage_type:12s} ({label:10s}): {count:3d} segments, {hours:.1f}h")
    print("=" * 50)

    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"google_sleep_{start_dt.date()}_to_{end_dt.date()}.csv")
    save_stages_csv(stages, output_path)
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()