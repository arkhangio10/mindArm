"""Configuration defaults, implementation selection, and stable path resolution."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mindarm.config import Settings, default_config_path, load_settings
from mindarm.factory import create_components
from mindarm.interfaces import ArmController, Decoder, EEGSource, Perception, Planner
from mindarm.planner.fake import FakePlanner
from mindarm.session_log import SessionLog


def test_paths_resolve_from_config_not_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = default_config_path()
    monkeypatch.chdir(tmp_path)
    settings = load_settings(path)
    assert settings.planner == "nemotron"
    assert settings.scene_path.is_file() and settings.joint_limits_path.is_file()


def test_factory_conforms_to_protocols(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import ScriptedDecoder

    monkeypatch.setattr("mindarm.factory.KeyboardDecoder", ScriptedDecoder)
    components = create_components(settings, SessionLog(settings.logs_dir))
    for instance, protocol in zip(
        [components.eeg, components.decoder, components.perception, components.planner, components.arm],
        [EEGSource, Decoder, Perception, Planner, ArmController],
    ):
        assert isinstance(instance, protocol)
    assert isinstance(components.planner, FakePlanner)


def test_unknown_implementation_and_fields_are_rejected(settings: Settings) -> None:
    data = settings.model_dump()
    with pytest.raises(ValidationError):
        Settings.model_validate({**data, "planner": "anything"})
    with pytest.raises(ValidationError):
        Settings.model_validate({**data, "typo": True})
