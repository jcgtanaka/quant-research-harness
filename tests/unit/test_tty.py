"""read_secret reads one line from a real (pseudo) terminal with echo off, and restores echo afterwards."""

from __future__ import annotations

import os
import pty
import termios
import threading
import time

import pytest

from qrh import tty


@pytest.fixture
def terminal(monkeypatch):
    master, slave = pty.openpty()
    monkeypatch.setattr(tty, "TTY", os.ttyname(slave))
    yield master, slave
    os.close(master)
    os.close(slave)


def _type_later(master: int, text: bytes) -> threading.Thread:
    def feed() -> None:
        time.sleep(0.2)
        os.write(master, text)

    thread = threading.Thread(target=feed)
    thread.start()
    return thread


def test_read_secret_returns_the_typed_line_from_a_terminal(terminal):
    master, _slave = terminal
    feeder = _type_later(master, b"correct horse battery\n")
    assert tty.read_secret("passphrase: ") == "correct horse battery"
    feeder.join()


def test_read_secret_restores_echo_after_reading(terminal):
    master, slave = terminal
    feeder = _type_later(master, b"another long passphrase\n")
    tty.read_secret("passphrase: ")
    feeder.join()
    assert termios.tcgetattr(slave)[3] & termios.ECHO


def test_read_secret_strips_carriage_return(terminal):
    master, _slave = terminal
    feeder = _type_later(master, b"with carriage return\r\n")
    assert tty.read_secret("passphrase: ") == "with carriage return"
    feeder.join()
