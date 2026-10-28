"""Nonblocking keyboard input on Windows, POSIX terminals, and pipes."""

from __future__ import annotations

import os
import sys
from queue import Empty, Queue
from threading import Thread
from typing import Any, TextIO

from mindarm.interfaces import DecoderEvent


class KeyboardDecoder:
    """Map 1-4 to SELECT, e to toggle, and q/EOF to QUIT; restore terminal on close."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._queue: Queue[str] = Queue()
        self._terminal = self._stream.isatty()
        self._original: list[Any] | None = None
        if self._terminal and os.name != "nt":
            import termios
            import tty

            self._original = termios.tcgetattr(self._stream.fileno())
            tty.setcbreak(self._stream.fileno())
        elif not self._terminal:
            Thread(target=self._read_pipe, name="mindarm-keyboard", daemon=True).start()

    def _read_pipe(self) -> None:
        try:
            fd: int | None = self._stream.fileno()
        except (OSError, ValueError):
            fd = None
        try:
            while True:
                # Raw reads avoid holding stdin's buffered lock during interpreter exit.
                char = (os.read(fd, 1).decode("utf-8", errors="replace")
                        if fd is not None else self._stream.read(1))
                self._queue.put(char)
                if not char:
                    return
        except (OSError, ValueError):
            self._queue.put("")

    @staticmethod
    def decode_key(key: str) -> DecoderEvent | None:
        """Translate one key without terminal I/O."""
        if key in ("1", "2", "3", "4"):
            return DecoderEvent(kind="SELECT", index=int(key) - 1)
        if key.lower() == "e":
            return DecoderEvent(kind="TOGGLE_ENGAGE")
        if key.lower() == "q" or key in ("", "\x03", "\x04"):
            return DecoderEvent(kind="QUIT")
        return None

    def poll(self) -> DecoderEvent | None:
        if self._terminal and os.name == "nt":
            import msvcrt

            if not msvcrt.kbhit():
                return None
            char = msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                msvcrt.getwch()  # Ignore Windows extended-key scan codes.
                return None
        elif self._terminal:
            import select

            if not select.select([self._stream], [], [], 0)[0]:
                return None
            char = os.read(self._stream.fileno(), 1).decode("utf-8", errors="ignore")
        else:
            try:
                char = self._queue.get_nowait()
            except Empty:
                return None
        return self.decode_key(char)

    def close(self) -> None:
        """Restore POSIX terminal settings; no-op for Windows and redirected input."""
        if self._original is not None:
            import termios

            termios.tcsetattr(self._stream.fileno(), termios.TCSADRAIN, self._original)
            self._original = None
