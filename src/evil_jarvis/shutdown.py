"""Ctrl-C that always works, even while a native call (microphone, model) blocks."""

from __future__ import annotations

import os
import select
import signal
import sys
import threading
import time

FORCE_QUIT_PRESSES = 2  # a second Ctrl-C quits at once


def install_interrupt_watchdog(grace_seconds: float = 2.0) -> None:
    """Make Ctrl-C always work.

    Python only reacts to Ctrl-C between bytecodes, so a native call that blocks
    (a stalled microphone read, a model call) can swallow it. The OS-level
    signal handler still writes to the wakeup fd, so a watcher thread can see
    the press and force-exit if a clean shutdown has not finished in time.
    A second Ctrl-C exits immediately.
    """
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)

    def watch() -> None:
        presses = 0
        deadline = None
        while True:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready = wait_readable(read_fd, timeout)
            if ready:
                data = os.read(read_fd, 64)
                presses += data.count(bytes([signal.SIGINT]))
                if presses >= FORCE_QUIT_PRESSES:
                    force_exit("\nForce quitting.\n")
                if presses == 1 and deadline is None:
                    deadline = time.monotonic() + grace_seconds
            elif deadline is not None:
                force_exit("\nShutdown was stuck; force quitting.\n")

    threading.Thread(target=watch, name="ctrl-c-watchdog", daemon=True).start()


def wait_readable(fd: int, timeout: float | None) -> bool:

    ready, _, _ = select.select([fd], [], [], timeout)
    return bool(ready)


def force_exit(message: str) -> None:
    os.write(sys.stderr.fileno(), message.encode())
    os._exit(130)
