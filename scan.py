"""
vigil — scan for Polar Verity Sense via BLE

Run: uv run scan.py

Turn on your Verity Sense (press the button), then run this script.
It will scan for nearby BLE devices and list any Polar sensors found.
"""

import asyncio
from bleak import BleakScanner


async def scan_for_polar():
    print("Scanning for BLE devices (10 seconds)...")
    print("Make sure your Verity Sense is turned on (press the button)\n")

    devices = await BleakScanner.discover(duration=10.0)

    polar_devices = []
    for d in devices:
        # Polar devices typically advertise with "Polar" in the name
        name = d.name or ""
        rssi = "?"
        if hasattr(d, "rssi") and d.rssi is not None:
            rssi = d.rssi
        elif hasattr(d, "details"):
            details = d.details
            if isinstance(details, dict):
                rssi = details.get("rssi", "?")
            elif isinstance(details, (tuple, list)) and len(details) > 0:
                rssi = details[0]
        if "polar" in name.lower() or "verity" in name.lower():
            polar_devices.append(d)
            print(f"  ✅ POLAR DEVICE FOUND")
            print(f"     Name:    {d.name}")
            print(f"     Address: {d.address}")
            print(f"     RSSI:    {rssi} dBm")
            print()
        elif name:
            print(f"  {name:40s} {d.address}  RSSI: {rssi}")

    if not polar_devices:
        print("\n⚠️  No Polar devices found.")
        print("   Make sure the Verity Sense is turned on and not connected")
        print("   to the Polar Flow app (disconnect from Flow first).")
    else:
        print(f"\n✅ Found {len(polar_devices)} Polar device(s).")
        print(f"\nNext step: note the address above, we'll use it to connect.\n")

    # Also show all devices for debugging
    print(f"\nTotal BLE devices found: {len(devices)}")


if __name__ == "__main__":
    asyncio.run(scan_for_polar())