"""Capture stdout/stderr and broadcast to WebSocket clients."""

from __future__ import annotations

import asyncio
import io
import sys
import threading
from typing import Any, TextIO


class StreamCapture(io.TextIOBase):
    """Tee a stream: writes go to the original *and* a callback."""

    def __init__(self, original: TextIO, callback: Any, stream_name: str) -> None:
        self._original = original
        self._callback = callback
        self._stream_name = stream_name

    def write(self, text: str) -> int:
        if text:
            self._original.write(text)
            self._original.flush()
            self._callback(self._stream_name, text)
        return len(text) if text else 0

    def flush(self) -> None:
        self._original.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return False


class LogBroadcaster:
    """Thread-safe log capture → asyncio WebSocket broadcast.

    Usage:
        broadcaster = LogBroadcaster(loop)
        broadcaster.install()   # patches sys.stdout / sys.stderr
        ...
        broadcaster.uninstall() # restores originals
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=4096)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_capture: StreamCapture | None = None
        self._stderr_capture: StreamCapture | None = None
        self._lock = threading.Lock()
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._stdout_capture = StreamCapture(
            self._original_stdout, self._on_write, "stdout"
        )
        self._stderr_capture = StreamCapture(
            self._original_stderr, self._on_write, "stderr"
        )
        sys.stdout = self._stdout_capture  # type: ignore[assignment]
        sys.stderr = self._stderr_capture  # type: ignore[assignment]
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._installed = False

    def _on_write(self, stream: str, text: str) -> None:
        msg = {"type": "log", "stream": stream, "text": text}
        try:
            asyncio.run_coroutine_threadsafe(self._queue.put(msg), self._loop)
        except RuntimeError:
            pass  # loop closed — ignore

    async def drain(self, ws_clients: list) -> None:
        """Continuously drain the queue and broadcast to all WS clients.

        Run this as a long-lived asyncio task.
        """
        while True:
            msg = await self._queue.get()
            await _broadcast(ws_clients, msg)


async def _broadcast(ws_clients: list, msg: dict) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    import json

    payload = json.dumps(msg)
    stale: list[int] = []
    for i, ws in enumerate(ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(i)
    for i in reversed(stale):
        ws_clients.pop(i)
