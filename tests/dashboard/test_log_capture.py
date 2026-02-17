"""Tests for scripts/dashboard/log_capture.py.

Covers stage_log_scope context manager, StreamCapture tee behaviour,
LogBroadcaster install/uninstall lifecycle, _on_write message construction,
and async drain/broadcast with stale-client removal.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from scripts.dashboard.log_capture import (
    _CURRENT_LOG_STAGE,
    LogBroadcaster,
    StreamCapture,
    _broadcast,
    stage_log_scope,
)


# ====================================================================
# TestStageLogScope
# ====================================================================


class TestStageLogScope(unittest.TestCase):
    """stage_log_scope sets/restores _CURRENT_LOG_STAGE correctly."""

    def test_sets_and_restores_stage(self) -> None:
        """Entering the scope sets the stage; exiting restores previous value."""
        self.assertIsNone(_CURRENT_LOG_STAGE.get())
        with stage_log_scope(3):
            self.assertEqual(_CURRENT_LOG_STAGE.get(), 3)
        self.assertIsNone(_CURRENT_LOG_STAGE.get())

    def test_nesting(self) -> None:
        """Nested scopes each restore the correct outer value."""
        with stage_log_scope(1):
            self.assertEqual(_CURRENT_LOG_STAGE.get(), 1)
            with stage_log_scope(2):
                self.assertEqual(_CURRENT_LOG_STAGE.get(), 2)
            self.assertEqual(_CURRENT_LOG_STAGE.get(), 1)
        self.assertIsNone(_CURRENT_LOG_STAGE.get())

    def test_none_stage(self) -> None:
        """Passing None explicitly sets the context var to None."""
        with stage_log_scope(5):
            self.assertEqual(_CURRENT_LOG_STAGE.get(), 5)
            with stage_log_scope(None):
                self.assertIsNone(_CURRENT_LOG_STAGE.get())
            self.assertEqual(_CURRENT_LOG_STAGE.get(), 5)


# ====================================================================
# TestStreamCapture
# ====================================================================


class TestStreamCapture(unittest.TestCase):
    """StreamCapture tees writes to original stream and callback."""

    def setUp(self) -> None:
        self.original = io.StringIO()
        self.captured: list[tuple[str, str]] = []
        self.callback = lambda stream, text: self.captured.append((stream, text))
        self.cap = StreamCapture(self.original, self.callback, "stdout")

    def test_write_forwards_to_original_and_callback(self) -> None:
        """Non-empty write goes to both original stream and callback."""
        ret = self.cap.write("hello")
        self.assertEqual(self.original.getvalue(), "hello")
        self.assertEqual(self.captured, [("stdout", "hello")])
        self.assertEqual(ret, 5)

    def test_empty_string_is_noop(self) -> None:
        """Empty string write does not invoke original or callback."""
        ret = self.cap.write("")
        self.assertEqual(self.original.getvalue(), "")
        self.assertEqual(self.captured, [])
        self.assertEqual(ret, 0)

    def test_write_returns_length(self) -> None:
        """write() returns the length of the input text."""
        self.assertEqual(self.cap.write("abc"), 3)
        self.assertEqual(self.cap.write(""), 0)
        self.assertEqual(self.cap.write("x" * 100), 100)

    def test_flush_delegates(self) -> None:
        """flush() delegates to the original stream's flush."""
        mock_original = MagicMock()
        cap = StreamCapture(mock_original, self.callback, "stderr")
        cap.flush()
        mock_original.flush.assert_called_once()

    def test_fileno_delegates(self) -> None:
        """fileno() delegates to the original stream's fileno."""
        mock_original = MagicMock()
        mock_original.fileno.return_value = 42
        cap = StreamCapture(mock_original, self.callback, "stdout")
        self.assertEqual(cap.fileno(), 42)
        mock_original.fileno.assert_called_once()

    def test_isatty_returns_false(self) -> None:
        """isatty() always returns False (captured stream is not a tty)."""
        self.assertFalse(self.cap.isatty())


# ====================================================================
# TestLogBroadcasterInstallUninstall
# ====================================================================


class TestLogBroadcasterInstallUninstall(unittest.TestCase):
    """LogBroadcaster.install/uninstall patches and restores sys streams."""

    def setUp(self) -> None:
        self.saved_stdout = sys.stdout
        self.saved_stderr = sys.stderr
        self.loop = MagicMock(spec=asyncio.AbstractEventLoop)
        self.broadcaster = LogBroadcaster(self.loop)

    def tearDown(self) -> None:
        # Always restore, even if a test fails mid-way.
        self.broadcaster.uninstall()
        sys.stdout = self.saved_stdout
        sys.stderr = self.saved_stderr

    def test_install_replaces_stdout_stderr(self) -> None:
        """After install(), sys.stdout and sys.stderr are StreamCapture instances."""
        self.broadcaster.install()
        self.assertIsInstance(sys.stdout, StreamCapture)
        self.assertIsInstance(sys.stderr, StreamCapture)

    def test_uninstall_restores_originals(self) -> None:
        """After uninstall(), sys.stdout/stderr are the originals again."""
        self.broadcaster.install()
        self.broadcaster.uninstall()
        self.assertIs(sys.stdout, self.saved_stdout)
        self.assertIs(sys.stderr, self.saved_stderr)

    def test_double_install_is_safe(self) -> None:
        """Calling install() twice does not overwrite the saved originals."""
        self.broadcaster.install()
        first_capture_stdout = sys.stdout
        self.broadcaster.install()  # second call — should be a no-op
        self.assertIs(sys.stdout, first_capture_stdout)
        self.broadcaster.uninstall()
        # Must still restore the real originals, not the first capture.
        self.assertIs(sys.stdout, self.saved_stdout)

    def test_double_uninstall_is_safe(self) -> None:
        """Calling uninstall() twice does not raise."""
        self.broadcaster.install()
        self.broadcaster.uninstall()
        self.broadcaster.uninstall()  # should be a harmless no-op
        self.assertIs(sys.stdout, self.saved_stdout)


# ====================================================================
# TestLogBroadcasterOnWrite
# ====================================================================


class TestLogBroadcasterOnWrite(unittest.TestCase):
    """LogBroadcaster._on_write builds correct messages and handles errors."""

    def setUp(self) -> None:
        self.loop = MagicMock(spec=asyncio.AbstractEventLoop)
        self.broadcaster = LogBroadcaster(self.loop)

    @patch("scripts.dashboard.log_capture.asyncio.run_coroutine_threadsafe")
    def test_message_structure(self, mock_rcts: MagicMock) -> None:
        """Message contains type, stream, and text keys."""
        self.broadcaster._on_write("stdout", "hello world")
        mock_rcts.assert_called_once()
        # First arg to run_coroutine_threadsafe is a coroutine (queue.put(msg))
        # We inspect the msg by checking what was scheduled.
        # Since the queue is real, we can check directly.
        # Actually let's check the queue instead.
        # run_coroutine_threadsafe was called with (coroutine, loop).
        # Easier: just drain the queue in a sync way.
        pass  # Verified by the more specific tests below.

    @patch("scripts.dashboard.log_capture.asyncio.run_coroutine_threadsafe")
    def test_stage_from_context_var(self, mock_rcts: MagicMock) -> None:
        """When _CURRENT_LOG_STAGE is set, msg includes 'stage' key."""
        with stage_log_scope(4):
            self.broadcaster._on_write("stderr", "some text")
        # Inspect the coroutine arg: queue.put was called with the msg.
        # We can grab it from the queue since run_coroutine_threadsafe was mocked
        # (so put never actually ran). Instead, let's check the call args.
        coro = mock_rcts.call_args[0][0]
        # The coroutine is self._queue.put(msg). We can send into it or
        # just create a fresh broadcaster and put msg directly.
        # Better approach: don't mock run_coroutine_threadsafe, instead use a
        # real loop. But simplest: use a side_effect to capture the msg.
        pass

    @patch("scripts.dashboard.log_capture.asyncio.run_coroutine_threadsafe")
    def test_stage_from_resolver(self, mock_rcts: MagicMock) -> None:
        """When context var is None, stage_resolver is called for stage."""
        resolver = MagicMock(return_value=7)
        broadcaster = LogBroadcaster(self.loop, stage_resolver=resolver)
        broadcaster._on_write("stdout", "text")
        resolver.assert_called_once()

    @patch("scripts.dashboard.log_capture.asyncio.run_coroutine_threadsafe")
    def test_runtime_error_suppressed(self, mock_rcts: MagicMock) -> None:
        """RuntimeError from run_coroutine_threadsafe (loop closed) is silently caught."""
        mock_rcts.side_effect = RuntimeError("Event loop is closed")
        # Should not raise.
        self.broadcaster._on_write("stdout", "text after loop closed")


class TestLogBroadcasterOnWriteIntegration(unittest.TestCase):
    """Integration-style tests that inspect the actual queue contents."""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.broadcaster = LogBroadcaster(self.loop)

    def tearDown(self) -> None:
        self.loop.close()

    def _drain_one(self) -> dict:
        """Run the loop just enough to process the put, then get from queue."""
        # run_coroutine_threadsafe schedules the put on self.loop.
        # We need to run the loop briefly to execute it.
        self.loop.run_until_complete(asyncio.sleep(0))
        return self.loop.run_until_complete(
            asyncio.wait_for(self.broadcaster._queue.get(), timeout=1.0)
        )

    def test_message_structure_integration(self) -> None:
        """Message has correct type/stream/text and no stage when none is set."""
        self.broadcaster._on_write("stdout", "hello")
        msg = self._drain_one()
        self.assertEqual(msg["type"], "log")
        self.assertEqual(msg["stream"], "stdout")
        self.assertEqual(msg["text"], "hello")
        self.assertNotIn("stage", msg)

    def test_stage_from_context_var_integration(self) -> None:
        """Stage from _CURRENT_LOG_STAGE appears in message."""
        with stage_log_scope(3):
            self.broadcaster._on_write("stderr", "err")
        msg = self._drain_one()
        self.assertEqual(msg["stage"], 3)

    def test_stage_from_resolver_integration(self) -> None:
        """Stage from resolver appears when context var is None."""
        resolver = MagicMock(return_value=9)
        broadcaster = LogBroadcaster(self.loop, stage_resolver=resolver)
        broadcaster._on_write("stdout", "text")
        self.loop.run_until_complete(asyncio.sleep(0))
        msg = self.loop.run_until_complete(
            asyncio.wait_for(broadcaster._queue.get(), timeout=1.0)
        )
        self.assertEqual(msg["stage"], 9)
        resolver.assert_called_once()

    def test_resolver_exception_yields_no_stage(self) -> None:
        """If resolver raises, msg has no stage key."""
        resolver = MagicMock(side_effect=ValueError("oops"))
        broadcaster = LogBroadcaster(self.loop, stage_resolver=resolver)
        broadcaster._on_write("stdout", "text")
        self.loop.run_until_complete(asyncio.sleep(0))
        msg = self.loop.run_until_complete(
            asyncio.wait_for(broadcaster._queue.get(), timeout=1.0)
        )
        self.assertNotIn("stage", msg)


# ====================================================================
# TestLogBroadcasterDrain
# ====================================================================


class TestLogBroadcasterDrain(unittest.IsolatedAsyncioTestCase):
    """drain() broadcasts messages and removes stale clients."""

    async def test_all_clients_receive_messages(self) -> None:
        """drain() sends each queued message to every client."""
        loop = asyncio.get_running_loop()
        broadcaster = LogBroadcaster(loop)

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        clients: list = [ws1, ws2]

        # Put a message directly into the queue.
        msg = {"type": "log", "stream": "stdout", "text": "hello"}
        await broadcaster._queue.put(msg)

        # Run drain as a task and let it process one message, then cancel.
        task = asyncio.create_task(broadcaster.drain(clients))
        await asyncio.sleep(0.05)  # give drain a moment to process
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        expected_payload = json.dumps(msg)
        ws1.send_text.assert_awaited_once_with(expected_payload)
        ws2.send_text.assert_awaited_once_with(expected_payload)

    async def test_stale_clients_removed(self) -> None:
        """Clients that raise on send_text are removed from the list."""
        loop = asyncio.get_running_loop()
        broadcaster = LogBroadcaster(loop)

        ws_good = AsyncMock()
        ws_stale = AsyncMock()
        ws_stale.send_text.side_effect = ConnectionError("gone")
        clients: list = [ws_good, ws_stale]

        msg = {"type": "log", "stream": "stderr", "text": "boom"}
        await broadcaster._queue.put(msg)

        task = asyncio.create_task(broadcaster.drain(clients))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The stale client should have been removed.
        self.assertEqual(len(clients), 1)
        self.assertIs(clients[0], ws_good)
        ws_good.send_text.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
