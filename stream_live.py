"""
vigil — live stream PPG + ACC from Polar Verity Sense

Run: uv run stream_live.py

Connects to the Verity Sense, streams live PPG (55 Hz) + accelerometer (52 Hz)
+ PPI + HR for 15 seconds, prints sample counts and first few values.

Make sure the sensor is turned on and NOT connected to the Polar Flow app.
"""

import asyncio
from bleak import BleakScanner
from polar_python import PolarDevice
from polar_python.models import PPGData, ACCData, PPIData, HRData

# Sensor address from scan.py
SENSOR_ADDRESS = "24:AC:AC:1F:72:FF"

# Counters
ppg_count = 0
acc_count = 0
ppi_count = 0
hr_count = 0


async def main():
    global ppg_count, acc_count, ppi_count, hr_count

    print(f"Looking for Polar sensor at {SENSOR_ADDRESS}...")
    device = await BleakScanner.find_device_by_address(SENSOR_ADDRESS, timeout=10.0)
    if not device:
        print("❌ Sensor not found. Make sure it's turned on and not connected to Flow app.")
        return

    print(f"✅ Found {device.name} ({device.address})")
    print("Connecting...")

    polar_device = PolarDevice(device)
    # Retry connection — BlueZ on Linux sometimes drops on first attempt
    connected = False
    for attempt in range(3):
        try:
            print(f"Connecting (attempt {attempt+1}/3)...")
            await polar_device.connect()
            connected = True
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2)
    if not connected:
        print("❌ Could not connect after 3 attempts.")
        print("   Make sure the Polar Flow app is CLOSED on your phone")
        print("   (it may be holding the BLE connection).")
        return
    print("Connected! Starting streams...\n")

    # Callbacks
    def ppg_callback(data: PPGData):
        global ppg_count
        ppg_count += 1
        if ppg_count <= 3:
            n_samples = len(data.samples)
            print(f"  PPG frame {ppg_count}: {n_samples} samples, "
                  f"ch0 first 5: {data.samples[0][:5] if data.samples else 'empty'}")

    def acc_callback(data: ACCData):
        global acc_count
        acc_count += 1
        if acc_count <= 3:
            n_samples = len(data.data)
            print(f"  ACC frame {acc_count}: {n_samples} samples, "
                  f"first: {data.data[0] if data.data else 'empty'}")

    def ppi_callback(data: PPIData):
        global ppi_count
        ppi_count += 1
        if ppi_count <= 5:
            for s in data.samples:
                print(f"  PPI frame {ppi_count}: hr={s.hr}, ppi={s.ppi}ms, "
                      f"err={s.pp_error_estimate}")

    def hr_callback(data: HRData):
        global hr_count
        hr_count += 1
        if hr_count <= 5:
            print(f"  HR  frame {hr_count}: hr={data.heartrate}, "
                  f"rr={data.rr_intervals}")

    # PPG: 55 Hz, 22-bit, 4 channels (Verity Sense defaults)
    await polar_device.start_ppg_stream(
        ppg_callback=ppg_callback, sample_rate=55, resolution=22, channels=4
    )
    print("  PPG stream started (55 Hz, 4ch)")

    # ACC: 52 Hz, 16-bit, ±8G, 3 channels (Verity Sense defaults)
    await polar_device.start_acc_stream(
        acc_callback=acc_callback, sample_rate=52, resolution=16, range=8, channels=3
    )
    print("  ACC stream started (52 Hz, ±8G)")

    # PPI: no config needed
    await polar_device.start_ppi_stream(ppi_callback=ppi_callback)
    print("  PPI stream started")

    # HR: standard BLE HR service
    await polar_device.start_hr_stream(hr_callback=hr_callback)
    print("  HR  stream started")

    print(f"\nStreaming for 15 seconds... keep the sensor on your skin\n")

    await asyncio.sleep(15)

    # Stop streams
    await polar_device.stop_ppg_stream()
    await polar_device.stop_acc_stream()
    await polar_device.stop_ppi_stream()
    await polar_device.stop_hr_stream()

    await polar_device.disconnect()

    # Summary
    print(f"\n{'='*50}")
    print(f"Stream summary (15 seconds):")
    print(f"  PPG frames received: {ppg_count}")
    print(f"  ACC frames received: {acc_count}")
    print(f"  PPI frames received: {ppi_count}")
    print(f"  HR  frames received: {hr_count}")
    print(f"{'='*50}")

    if ppg_count > 0 and acc_count > 0:
        print(f"\n✅ All streams working! Ready to build the recording pipeline.")
    elif ppg_count == 0 and acc_count == 0:
        print(f"\n⚠️  No data received. Check that the sensor is on your skin")
        print(f"   and not connected to the Polar Flow app.")
    else:
        print(f"\n⚠️  Partial data. Some streams may need SDK mode.")


if __name__ == "__main__":
    asyncio.run(main())