"""
vigil — shared sensor discovery

Finds the Polar Verity Sense by name (cross-platform: works on Linux with MAC
addresses and macOS with UUID addresses).
"""

from bleak import BleakScanner
import asyncio


async def find_sensor(timeout=15.0):
    """Find the Polar Verity Sense by name. Works cross-platform.

    Returns a BLEDevice or None.
    """
    print(f"Looking for Polar sensor...")
    device = await BleakScanner.find_device_by_filter(
        lambda dev, adv: dev.name and "polar sense" in dev.name.lower(),
        timeout=timeout,
    )
    if device:
        print(f"✅ Found {device.name} ({device.address})")
    else:
        print("❌ Sensor not found. Turn it on and make sure Flow app is closed.")
    return device