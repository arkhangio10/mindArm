#!/usr/bin/env python3
"""MindArm arm service for the Jetson: JSON lines on stdin/stdout, ramped PCA9685 servo moves.

Runs on the system Python 3.6 with python3-smbus only; the laptop starts it over SSH. On end of
input, SIGHUP, or SIGTERM it stops and exits without releasing, so the arm holds its posture.
"""
import json
import os
import select
import signal
import sys
import time

BUS, ADDR = 1, 0x40
MODE1, LED0, ALL_LED_OFF_H, PRESCALE = 0x00, 0x06, 0xFD, 0xFE
SLEEP, AUTO_INCREMENT, FULL_OFF = 0x10, 0x20, 0x10
PRESCALE_50HZ = 121  # 25 MHz / (4096 * 50 Hz) - 1
PERIOD_US = 20000
STEP_S = 0.02
MAX_SPEED_US_S = 1000.0
MAX_DURATION_S = 10.0
MAX_STAGGER_S = 2.0
JOINTS = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]  # index = channel
# Hard envelope from the 2026-10-11 calibration; the laptop configuration must stay inside it.
LIMITS_US = {"base": (1000, 2000), "shoulder": (1000, 2000), "elbow": (1000, 2000),
             "wrist_pitch": (1180, 1980), "wrist_roll": (1000, 2000), "gripper": (1130, 1950)}


class PCA9685(object):
    """Minimal PCA9685 access that adopts a chip already running at 50 Hz instead of resetting it."""

    def __init__(self, bus):
        self.bus = bus
        mode = bus.read_byte_data(ADDR, MODE1)
        if mode & SLEEP or bus.read_byte_data(ADDR, PRESCALE) != PRESCALE_50HZ:
            bus.write_byte_data(ADDR, ALL_LED_OFF_H, FULL_OFF)
            # The prescaler can only be written while the oscillator sleeps.
            bus.write_byte_data(ADDR, MODE1, (mode & 0x7F) | SLEEP)
            bus.write_byte_data(ADDR, PRESCALE, PRESCALE_50HZ)
            bus.write_byte_data(ADDR, MODE1, (mode & 0x6F) | AUTO_INCREMENT)
            time.sleep(0.001)
        elif not mode & AUTO_INCREMENT:
            bus.write_byte_data(ADDR, MODE1, (mode & 0x7F) | AUTO_INCREMENT)

    def read_us(self, channel):
        """Return the last commanded pulse in microseconds, or None if the channel is released."""
        base = LED0 + 4 * channel
        on_l, on_h, off_l, off_h = [self.bus.read_byte_data(ADDR, base + i) for i in range(4)]
        if off_h & FULL_OFF or on_h & FULL_OFF:
            return None
        count = (((off_h & 0x0F) << 8 | off_l) - ((on_h & 0x0F) << 8 | on_l)) % 4096
        return int(round(count * PERIOD_US / 4096.0))

    def write_us(self, channel, us):
        count = int(round(us * 4096.0 / PERIOD_US))
        # One transaction: the chip latches all four bytes together, never an intermediate pulse.
        self.bus.write_i2c_block_data(ADDR, LED0 + 4 * channel, [0, 0, count & 0xFF, count >> 8])


class Lines(object):
    """Unbuffered stdin line reader with a timeout, so a ramp notices input at once."""

    def __init__(self, fd):
        self.fd, self.buffer, self.closed = fd, b"", False

    def next(self, timeout):
        """Return the next line, or None after the timeout or once input has closed."""
        while b"\n" not in self.buffer and not self.closed:
            if not select.select([self.fd], [], [], timeout)[0]:
                return None
            chunk = os.read(self.fd, 4096)
            self.closed = not chunk
            self.buffer += chunk
        if b"\n" not in self.buffer:
            return None
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("utf-8")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Arm(object):
    """Check commands against the hard envelope and ramp the channels; never release them."""

    def __init__(self, pca, lines):
        self.pca, self.lines = pca, lines
        self.positions = {joint: pca.read_us(channel) for channel, joint in enumerate(JOINTS)}

    def reply(self, request_id, ok, error=None, stopped=False):
        return {"id": request_id, "ok": ok, "stopped": stopped, "positions": dict(self.positions), "error": error}

    def check(self, targets, duration, energize):
        if not isinstance(targets, dict) or set(targets) != set(JOINTS):
            raise ValueError("targets must name exactly: " + ", ".join(JOINTS))
        if not is_number(duration) or not 0 < duration <= MAX_DURATION_S:
            raise ValueError("duration must be in (0, {}] s".format(MAX_DURATION_S))
        for joint in JOINTS:
            value, (low, high), start = targets[joint], LIMITS_US[joint], self.positions[joint]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("{}: target must be an integer of us".format(joint))
            if not low <= value <= high:
                raise ValueError("{}: {} us outside [{}, {}]".format(joint, value, low, high))
            if start is None and not energize:
                raise ValueError("{} is released; send start first".format(joint))
            if start is not None and abs(value - start) / float(duration) > MAX_SPEED_US_S:
                raise ValueError("{}: speed exceeds {} us/s".format(joint, MAX_SPEED_US_S))

    def interrupted(self, timeout):
        return self.lines.next(timeout) is not None or self.lines.closed

    def ramp(self, targets, duration):
        """Move linearly; return False if a received line or end of input stopped the move."""
        start = dict(self.positions)
        moving = [joint for joint in JOINTS if targets[joint] != start[joint]]
        began = time.monotonic()
        while moving:
            fraction = min(1.0, (time.monotonic() - began) / duration)
            for joint in moving:
                value = int(round(start[joint] + (targets[joint] - start[joint]) * fraction))
                if value != self.positions[joint]:
                    self.pca.write_us(JOINTS.index(joint), value)
                    self.positions[joint] = value
            if fraction >= 1.0:
                break
            if self.interrupted(STEP_S):
                return False
        return True

    def start(self, targets, duration, stagger):
        """Energize released channels one at a time at their targets, then ramp the others."""
        for channel, joint in enumerate(JOINTS):
            if self.positions[joint] is None:
                self.pca.write_us(channel, targets[joint])
                self.positions[joint] = targets[joint]
                if self.interrupted(stagger):
                    return False
        return self.ramp(targets, duration)

    def handle(self, request):
        """Run one command and return its reply, or None for a stop that arrives while idle."""
        if not isinstance(request, dict):
            return self.reply(None, False, "request must be a JSON object")
        command, request_id = request.get("cmd"), request.get("id")
        if command == "stop":
            return None
        try:
            if command not in ("start", "move"):
                raise ValueError("unknown command: {}".format(command))
            targets, duration = request.get("targets"), request.get("duration")
            self.check(targets, duration, energize=command == "start")
            if command == "start":
                stagger = request.get("stagger")
                if not is_number(stagger) or not 0 <= stagger <= MAX_STAGGER_S:
                    raise ValueError("stagger must be in [0, {}] s".format(MAX_STAGGER_S))
                done = self.start(targets, duration, stagger)
            else:
                done = self.ramp(targets, duration)
        except ValueError as error:
            return self.reply(request_id, False, str(error))
        if not done:
            return self.reply(request_id, False, "Stopped before the target", stopped=True)
        return self.reply(request_id, True)


def write(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    import smbus

    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(0))
    lines = Lines(sys.stdin.fileno())
    arm = Arm(PCA9685(smbus.SMBus(BUS)), lines)
    write(arm.reply(0, True))
    while True:
        line = lines.next(None)
        if line is None:
            return  # End of input: stop and hold; the PCA9685 keeps the last pulses.
        try:
            request = json.loads(line)
        except ValueError as error:
            write(arm.reply(None, False, "invalid JSON: {}".format(error)))
            continue
        response = arm.handle(request)
        if response is not None:
            write(response)


if __name__ == "__main__":
    main()
