"""Exercise the actual module entry point and keyboard-driven subprocess loop."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import yaml

from conftest import ROOT
from mindarm.config import Settings


def environment() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "NEBIUS_API_KEY": ""}


def test_cli_help() -> None:
    result = subprocess.run([sys.executable, "-m", "mindarm", "--help"], env=environment(),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "--planner" in result.stdout and "--once" in result.stdout


def test_cli_full_offline_cycle(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json")), encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-m", "mindarm", "--config", str(path), "--once"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=environment(), cwd=tmp_path)
    lines: Queue[str] = Queue()
    output: list[str] = []

    def read_lines() -> None:
        for line in process.stdout:
            lines.put(line)

    reader = Thread(target=read_lines, daemon=True)
    reader.start()

    def wait_for(text: str) -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=0.1)
                output.append(line)
                if text in line:
                    return
            except Empty:
                pass
        raise AssertionError("Missing output: " + text + "\n" + "".join(output))

    try:
        wait_for("IDLE -")
        process.stdin.write("e\n")
        process.stdin.flush()
        wait_for("4. Home")
        process.stdin.write("2\n")
        process.stdin.flush()
        wait_for("Executed RELEASE")
        assert process.wait(timeout=10) == 0
        session = next(settings.logs_dir.glob("*.jsonl"))
        records = [json.loads(line) for line in session.read_text(encoding="utf-8").splitlines()]
        cycle = next(record for record in records if record["type"] == "cycle")
        assert cycle["status"] == "completed" and cycle["selection"] == 1
        assert records[-1]["type"] == "session_end"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdin.close()
        reader.join(timeout=2)
        process.stdout.close()


def test_missing_key_is_actionable(settings: Settings, tmp_path: Path) -> None:
    settings.planner = "nemotron"
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json")), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "mindarm", "--config", str(path)],
                            env=environment(), capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "NEBIUS_API_KEY" in result.stderr and "--planner fake" in result.stderr
