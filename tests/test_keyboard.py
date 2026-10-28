"""Keyboard mapping and redirected-input lifecycle."""

import io
import time

import pytest

from mindarm.decoder.keyboard import KeyboardDecoder


@pytest.mark.parametrize("key,index", [("1", 0), ("2", 1), ("3", 2), ("4", 3)])
def test_select_keys(key: str, index: int) -> None:
    event = KeyboardDecoder.decode_key(key)
    assert event is not None and event.kind == "SELECT" and event.index == index


def test_control_keys() -> None:
    assert KeyboardDecoder.decode_key("e").kind == "TOGGLE_ENGAGE"
    assert KeyboardDecoder.decode_key("q").kind == "QUIT"
    assert KeyboardDecoder.decode_key("\n") is None
    assert KeyboardDecoder.decode_key("x") is None


def test_pipe_reaches_quit_on_eof() -> None:
    decoder = KeyboardDecoder(io.StringIO("e2"))
    events = []
    deadline = time.monotonic() + 2
    while len(events) < 3 and time.monotonic() < deadline:
        event = decoder.poll()
        if event:
            events.append(event.kind)
        time.sleep(0.001)
    decoder.close()
    assert events == ["TOGGLE_ENGAGE", "SELECT", "QUIT"]
