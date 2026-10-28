"""Fixture-backed perception."""

from pathlib import Path

from mindarm.interfaces import Scene


class FakePerception:
    """Load and validate the fixture afresh on every detection."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def detect(self) -> Scene:
        return Scene.model_validate_json(self._path.read_text(encoding="utf-8"))
