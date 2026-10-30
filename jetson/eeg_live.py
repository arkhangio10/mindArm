#!/usr/bin/env python3
"""Streams one EEG channel (ADS1115, I2C bus 0) to stdout as JSON lines every 50 ms, for the laptop to read over SSH."""
import json
import signal
import sys
import time

import smbus

BUS = 0
RATE = 250.0
LSB_V = 4.096 / 32768
# Continuous, AIN0 vs GND, PGA ±4.096 V, 475 SPS (faster than the sampling, so every read gets a fresh conversion).
CONFIG_RUN = [0x42, 0xC3]
CONFIG_IDLE = [0x85, 0x83]


def main():
    channel = 2 if len(sys.argv) > 1 and sys.argv[1] == "2" else 1
    addr = 0x49 if channel == 2 else 0x48
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(0))
    bus = smbus.SMBus(BUS)
    bus.write_i2c_block_data(addr, 0x01, CONFIG_RUN)
    time.sleep(0.01)
    batch, errors, count, rate, i = [], 0, 0, 0.0, 0
    start = mark = flushed = time.perf_counter()
    try:
        while True:
            target = start + i / RATE
            now = time.perf_counter()
            if now < target:
                time.sleep(max(0.0, target - now - 0.0005))
                continue
            if now - target > 0.1:
                start = now - i / RATE  # Fell behind: resynchronize instead of reading in a burst.
            i += 1
            try:
                hi, lo = bus.read_i2c_block_data(addr, 0x00, 2)
            except OSError:
                errors += 1
                continue
            raw = (hi << 8) | lo
            batch.append(round((raw - 65536 if raw & 0x8000 else raw) * LSB_V, 6))
            count += 1
            if count % int(RATE) == 0:
                rate, mark = RATE / (now - mark), now
            if now - flushed >= 0.05:
                print(json.dumps({"s": batch, "rate": round(rate, 1), "err": errors, "ch": channel}), flush=True)
                batch, flushed = [], now
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        bus.write_i2c_block_data(addr, 0x01, CONFIG_IDLE)


if __name__ == "__main__":
    main()
