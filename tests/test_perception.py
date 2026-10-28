"""Fixture contents and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mindarm.interfaces import Scene
from mindarm.perception.fake import FakePerception


def test_fixture_scene(scene: Scene) -> None:
    assert [obj.label for obj in scene.objects] == ["cup", "bottle", "sports ball"]


def test_bad_fixture(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"objects":[{"label":"cup","x":2,"y":0,"confidence":1}]}', encoding="utf-8")
    with pytest.raises(ValidationError):
        FakePerception(path).detect()
