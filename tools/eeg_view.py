"""Operator console server: streams the Jetson's EEG and camera to the browser and runs MindArm for the page."""
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import dotenv_values

from mindarm.teach import CAMERA_COMMAND, CameraStream

SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
RATE = 250
PORT = 8765
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONSOLE = HERE / "console.html"
MINDARM_CONFIG = REPO / "config" / "mindarm_live.yaml"
CALIBRATIONS = REPO / "recordings"
LOCAL_ORIGINS = {"http://localhost:{}".format(PORT), "http://127.0.0.1:{}".format(PORT)}
# 60 Hz mains: avoids banding in the image. The rest sets color and sharpness for the table's light.
CAMERA_SETUP = ("v4l2-ctl -d /dev/video0 -c power_line_frequency=2,sharpness=3,saturation=84,contrast=34,"
                "gamma=100,backlight_compensation=2,exposure_auto_priority=0; ")


class Feed(threading.Thread):
    """Starts the streamer on the Jetson over SSH and keeps the last 10 s that arrive."""

    def __init__(self, channel):
        super().__init__(daemon=True)
        command = "python3 -u ~/eeg_live.py {}".format(channel)
        self.proc = subprocess.Popen([SSH, "-o", "BatchMode=yes", "jetson-usb", command],
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.buffer = deque(maxlen=RATE * 10)
        self.count = 0
        self.meta = {"rate": 0.0, "err": 0, "ch": channel}
        self.lock = threading.Lock()

    def run(self):
        for line in self.proc.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            with self.lock:
                self.buffer.extend(message["s"])
                self.count += len(message["s"])
                self.meta = {key: message[key] for key in ("rate", "err", "ch")}

    def since(self, last):
        with self.lock:
            new = self.count - last
            items = list(self.buffer)[-new:] if new > 0 else []
            return items, self.count, dict(self.meta)


class MindArm:
    """Runs the real MindArm for the console: forwards its keys and publishes its states, options, and calls."""

    STATES = {"IDLE", "PROPOSING", "SELECTING", "PLANNING", "EXECUTING"}

    def __init__(self, config=MINDARM_CONFIG):
        self.config = config
        self.proc = None
        self.events = []
        self.queued = ""
        self.lock = threading.Lock()

    def publish(self, **event):
        with self.lock:
            self.events.append(event)

    def since(self, last):
        with self.lock:
            return self.events[last:], len(self.events)

    def key(self, key):
        """The first e starts MindArm, which energizes the arm, and is delivered once it is ready."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                if key != "e":
                    return
                self.queued = key
                env = {**os.environ, **{k: v for k, v in dotenv_values(REPO / ".env").items() if v}}
                self.proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", "mindarm", "--config", str(self.config), "--once"],
                    cwd=str(REPO), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, encoding="utf-8", errors="replace", bufsize=1)
                threading.Thread(target=self.pump, args=(self.proc,), daemon=True).start()
                self.events.append({"kind": "starting"})
                return
            if self.queued and key == "e":
                return
            self.proc.stdin.write(key)
            self.proc.stdin.flush()

    def pump(self, proc):
        for line in proc.stdout:
            self.parse(proc, line.rstrip())
        self.publish(kind="exit", code=proc.wait())

    def parse(self, proc, line):
        option = re.match(r"^([1-4])\. (.*?): (.*)$", line)
        selected = re.match(r"^Gesture selected ([1-4])\. ", line)
        if line.startswith("Session log: "):
            threading.Thread(target=self.tail, args=(proc, Path(line[13:])), daemon=True).start()
        elif line.startswith("IDLE - e:"):
            with self.lock:
                queued, self.queued = self.queued, ""
            if queued:
                proc.stdin.write(queued)
                proc.stdin.flush()
        elif line in self.STATES:
            self.publish(kind="state", state=line)
        elif option:
            self.publish(kind="option", index=int(option.group(1)) - 1, label=option.group(2), description=option.group(3))
        elif selected:
            self.publish(kind="selected", index=int(selected.group(1)) - 1)
        elif "Double blink detected: " in line:
            self.publish(kind="blink", text=line.split("detected: ", 1)[1])
        elif line.startswith("Executed "):
            self.publish(kind="executed", action=line[9:])
        elif line.startswith(("Error:", "HOME failed:", "MindArm:", "Traceback")):
            self.publish(kind="error", text=line)
        elif line and not line.startswith("INFO"):
            self.publish(kind="message", text=line)

    def tail(self, proc, path):
        partial = ""
        with path.open(encoding="utf-8") as log:
            while True:
                partial += log.readline()
                if not partial.endswith("\n"):
                    if proc.poll() is not None:
                        return
                    time.sleep(0.1)
                    continue
                record, partial = json.loads(partial), ""
                if record["type"] == "planner_call":
                    usage = record.get("usage") or {}
                    self.publish(kind="planner", operation=record["operation"], ms=record["latency_ms"], status=record["status"],
                                 p=usage.get("prompt_tokens", 0), c=usage.get("completion_tokens", 0))
                elif record["type"] == "cycle":
                    plan = record.get("plan") or {}
                    self.publish(kind="cycle", plan=plan.get("primitives", []), status=record.get("status"),
                                 executed=(record.get("result") or {}).get("executed_primitives", 0))


class Handler(BaseHTTPRequestHandler):
    """Serves the page, the camera, the MindArm events, and the new samples every 50 ms."""

    feed = None
    camera = None
    mindarm = None

    def log_message(self, *args):
        pass

    def send_page(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/mindarm/key":
            key = self.rfile.read(int(self.headers["Content-Length"])).decode("ascii", "ignore")
            if self.headers.get("Origin") not in LOCAL_ORIGINS or key not in ("e", "q", "1", "2", "3", "4"):
                self.send_error(403)
                return
            self.mindarm.key(key)
            self.send_page(b"{}", "application/json")
            return
        if self.path != "/calibration":
            self.send_error(404)
            return
        name = time.strftime("eeg_calibration_%Y%m%d_%H%M%S.json")
        CALIBRATIONS.mkdir(exist_ok=True)
        (CALIBRATIONS / name).write_bytes(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_page(json.dumps({"file": name}).encode("utf-8"), "application/json")

    def do_GET(self):
        if self.path == "/console":
            self.send_page(CONSOLE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/camera":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    frame = self.camera.next_frame(2.0)
                    if frame is None:
                        return
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n"
                                     % (len(frame), frame))
            except OSError:
                return
        elif self.path == "/mindarm/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            _, last = self.mindarm.since(0)
            try:
                while True:
                    items, last = self.mindarm.since(last)
                    for item in items:
                        self.wfile.write(("data: " + json.dumps(item) + "\n\n").encode("utf-8"))
                    self.wfile.write(b":\n\n")
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = max(0, self.feed.count - RATE * 10)
            try:
                while True:
                    items, last, meta = self.feed.since(last)
                    self.wfile.write(("data: " + json.dumps({"s": items, **meta}) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
        else:
            self.send_error(404)


def main():
    channel = "2" if len(sys.argv) > 1 and sys.argv[1] == "2" else "1"
    feed = Feed(channel)
    Handler.feed = feed
    Handler.mindarm = MindArm()
    Handler.camera = CameraStream([SSH, "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=2",
                                   "-o", "ServerAliveCountMax=3", "jetson-usb",
                                   CAMERA_SETUP + CAMERA_COMMAND.replace("width=640,height=480", "width=1280,height=720")
                                   .replace("--set-parm=15", "--set-parm=30")])
    feed.start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("Channel {} live: open http://localhost:{}/console. Ctrl+C to quit.".format(channel, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.mindarm.key("q")
        feed.proc.terminate()
        Handler.camera.close()
        server.server_close()


if __name__ == "__main__":
    main()
