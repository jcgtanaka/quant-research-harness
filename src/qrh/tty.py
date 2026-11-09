"""The controlling terminal: the only place a passphrase is read from.

``has_terminal``, ``write`` and ``read_secret`` are the seam tests replace (monkeypatch); shipped code has no
other input path for a passphrase. There is deliberately no fallback to stdin: a process without a controlling
terminal (an agent's shell, a pipe, CI) cannot approve.
"""

from __future__ import annotations

import os

TTY = "/dev/tty"


def has_terminal() -> bool:
    try:
        fd = os.open(TTY, os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return False
    os.close(fd)
    return True


def write(text: str) -> None:
    with open(TTY, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()


def _read_line(fd: int) -> str:
    """Read bytes until a newline or end of input. Works on a terminal, which cannot seek."""
    chunks = bytearray()
    while True:
        byte = os.read(fd, 1)
        if not byte or byte == b"\n":
            break
        chunks += byte
    return chunks.decode("utf-8")


def read_secret(prompt: str) -> str:
    """Read one line from the terminal with echo off. Raises OSError without a terminal."""
    import termios

    fd = os.open(TTY, os.O_RDWR | os.O_NOCTTY)
    try:
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~termios.ECHO
        os.write(fd, prompt.encode("utf-8"))
        try:
            termios.tcsetattr(fd, termios.TCSAFLUSH, new)
            line = _read_line(fd)
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
            os.write(fd, b"\n")
    finally:
        os.close(fd)
    return line.rstrip("\n").rstrip("\r")
