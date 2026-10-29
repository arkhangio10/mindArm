"""Validated configuration, independent of the caller's working directory."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Select implementations and configure only walking-skeleton features."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    eeg: Literal["fake", "replay"] = "fake"
    decoder: Literal["keyboard"] = "keyboard"
    perception: Literal["fake"] = "fake"
    planner: Literal["nemotron", "fake"] = "nemotron"
    arm: Literal["simulated", "jetson"] = "simulated"
    scene_path: Path
    joint_limits_path: Path
    jetson_arm_path: Path
    jetson_command: list[str] = Field(min_length=1)
    replay_path: Path
    logs_dir: Path
    recordings_dir: Path
    sample_rate: float = Field(default=250.0, gt=0)
    n_channels: int = Field(default=1, ge=1)
    frequency: float = Field(default=12.0, gt=0)
    noise_std: float = Field(default=0.2, ge=0)
    alpha_bursts: bool = False
    seed: int | None = 7
    replay_loop: bool = True
    record_eeg: bool = True
    eeg_chunk_samples: int = Field(default=25, ge=1)
    poll_interval: float = Field(default=0.02, gt=0, le=1)
    planner_timeout: float = Field(default=30.0, gt=0, le=120)
    planner_max_tokens: int = Field(default=4096, ge=256, le=16384)


def default_config_path() -> Path:
    """Find configuration beside this editable source checkout."""
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def load_settings(path: Path) -> Settings:
    """Resolve configured paths relative to the YAML file, never to cwd."""
    path = path.resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = Settings.model_validate(data)
    for field in ("scene_path", "joint_limits_path", "jetson_arm_path", "replay_path", "logs_dir", "recordings_dir"):
        value = getattr(settings, field)
        setattr(settings, field, (path.parent / value).resolve())
    return settings
