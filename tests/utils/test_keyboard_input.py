#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the display-independent keyboard input helpers.

These cover the parts most likely to regress: the environment-detection decision
table (the heart of the Wayland/headless fix), the macOS trust probe, the control
mapping, the terminal escape-sequence parsing, and backend selection. They require
neither ``pynput`` nor a real terminal.
"""

import io
import os
import platform
import pty
import sys
import threading
from types import SimpleNamespace

import pytest

import lerobot.utils.keyboard_input as ki
from lerobot.utils.keyboard_input import (
    TerminalKeyListener,
    apply_recording_control,
    create_key_listener,
    init_keyboard_listener,
    is_headless,
    is_wayland,
    key_listener_is_alive,
    pynput_can_capture,
    pynput_listener_is_trusted,
)


@pytest.fixture(autouse=True)
def _clear_detection_caches():
    """The detection helpers are ``@cache``-decorated; clear around each test."""
    for fn in (is_headless, is_wayland, pynput_can_capture):
        fn.cache_clear()
    yield
    for fn in (is_headless, is_wayland, pynput_can_capture):
        fn.cache_clear()


def _set_platform(monkeypatch, name):
    monkeypatch.setattr(platform, "system", lambda: name)


def _set_tty(monkeypatch, is_tty):
    stdin = io.StringIO("")
    stdin.isatty = lambda: is_tty
    monkeypatch.setattr(sys, "stdin", stdin)


# --- Environment detection (the core of the fix) ---------------------------
@pytest.mark.parametrize(
    ("system", "env", "expected"),
    [
        ("Linux", {}, True),  # no display server
        ("Linux", {"DISPLAY": ":0"}, False),  # X11
        ("Linux", {"WAYLAND_DISPLAY": "wayland-0"}, False),  # Wayland
        ("Darwin", {}, False),  # display always assumed present
    ],
)
def test_is_headless(monkeypatch, system, env, expected):
    _set_platform(monkeypatch, system)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert is_headless() is expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"XDG_SESSION_TYPE": "wayland"}, True),
        ({"WAYLAND_DISPLAY": "wayland-0"}, True),
        ({"XDG_SESSION_TYPE": "x11"}, False),
        ({}, False),
    ],
)
def test_is_wayland(monkeypatch, env, expected):
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert is_wayland() is expected


@pytest.mark.parametrize(
    ("system", "env", "pynput_available", "expected"),
    [
        ("Linux", {"DISPLAY": ":0"}, True, True),  # X11
        ("Linux", {"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0"}, True, False),  # Wayland
        ("Linux", {}, True, False),  # headless
        ("Darwin", {}, True, True),
        ("Linux", {"DISPLAY": ":0"}, False, False),  # pynput not installed
    ],
)
def test_pynput_can_capture(monkeypatch, system, env, pynput_available, expected):
    _set_platform(monkeypatch, system)
    monkeypatch.setattr(ki, "_pynput_available", pynput_available)
    monkeypatch.setattr(ki, "_x11_record_available", lambda: True)
    for var in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert pynput_can_capture() is expected


# --- macOS trust probe ------------------------------------------------------
class _FakeListener:
    def __init__(self, is_trusted):
        self.IS_TRUSTED = is_trusted


def test_pynput_listener_is_trusted(monkeypatch):
    _set_platform(monkeypatch, "Linux")
    assert pynput_listener_is_trusted(_FakeListener(False)) is True  # non-macOS: always assumed ok
    _set_platform(monkeypatch, "Darwin")
    assert pynput_listener_is_trusted(_FakeListener(False), timeout_s=0.05) is False


# --- Control mapping --------------------------------------------------------
def test_apply_recording_control():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    apply_recording_control("left", events)
    assert events == {"exit_early": True, "rerecord_episode": True, "stop_recording": False}
    apply_recording_control("esc", events)
    assert events["stop_recording"] is True
    apply_recording_control("up", events)  # unknown control -> no-op (no error)


# --- Terminal escape-sequence parsing (the tricky bit) ----------------------
def _drive(listener, byte_seq):
    """Run the listener's read loop over a scripted list of bytes (no real terminal)."""
    script = list(byte_seq)

    def fake_read(timeout):
        if script:
            return script.pop(0)
        listener._running = False
        return None

    listener._read_char = fake_read
    listener._running = True
    listener._run()


@pytest.mark.parametrize(
    ("byte_seq", "expected"),
    [
        (["\x1b", "[", "C"], ["right"]),  # CSI arrow
        (["\x1b", "O", "D"], ["left"]),  # SS3 arrow (e.g. over SSH/tmux)
        (["\x1b"], ["esc"]),  # bare ESC
        (["\x1b", "[", "A"], ["up"]),  # decoded even though the record handler ignores it
        (["n"], ["n"]),  # letter passthrough
    ],
)
def test_terminal_parsing(byte_seq, expected):
    collected = []
    _drive(TerminalKeyListener(collected.append), byte_seq)
    assert collected == expected


# --- Backend selection ------------------------------------------------------
def test_init_selects_terminal_when_pynput_cannot_capture(monkeypatch):
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)  # avoid touching termios
    listener, _ = init_keyboard_listener()
    assert isinstance(listener, TerminalKeyListener)


def test_init_returns_none_without_tty(monkeypatch):
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=False)
    listener, _ = init_keyboard_listener()
    assert listener is None


@pytest.mark.parametrize(
    ("key", "flag"),
    [("right", "exit_early"), ("r", "rerecord_episode"), ("q", "stop_recording")],
)
def test_init_terminal_key_routing(monkeypatch, key, flag):
    """Arrows and their letter equivalents drive the same events (terminal backend)."""
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)
    listener, events = init_keyboard_listener()
    listener._on_key(key)
    assert events[flag] is True


# --- Shared factory + pynput key resolver -----------------------------------
def test_resolve_pynput_key_char_fallback():
    """Unmapped keys fall back to ``.char`` (and yield None when there is none)."""
    assert ki._resolve_pynput_key(type("K", (), {"char": "s"})()) == "s"
    assert ki._resolve_pynput_key(type("K", (), {"char": None})()) is None
    assert ki._resolve_pynput_key(type("K", (), {"char": ""})()) is None  # empty char -> no key


def test_create_key_listener_routes_to_dispatch(monkeypatch):
    """The terminal backend forwards canonical key names straight to ``dispatch``."""
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)
    seen = []
    listener = create_key_listener(seen.append, controls_help="save='s'")
    assert isinstance(listener, TerminalKeyListener)
    listener._on_key("space")
    assert seen == ["space"]


def test_create_key_listener_none_without_tty(monkeypatch):
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=False)
    assert create_key_listener(lambda name: None) is None


def test_create_key_listener_prefers_terminal_even_when_display_is_available(monkeypatch):
    """Interactive rollouts must not select a fragile container X11 hook over their TTY."""
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: True)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)

    def unexpected_listener(*, on_press):
        raise AssertionError("pynput must not be constructed when terminal input is preferred")

    monkeypatch.setattr(ki, "keyboard", SimpleNamespace(Listener=unexpected_listener))

    listener = create_key_listener(lambda name: None, prefer_terminal=True)

    assert isinstance(listener, TerminalKeyListener)


def test_linux_without_xrecord_falls_back_to_terminal(monkeypatch):
    """A reachable DISPLAY alone is insufficient: pynput also requires X RECORD."""
    _set_platform(monkeypatch, "Linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.setattr(ki, "_pynput_available", True)
    monkeypatch.setattr(ki, "_x11_record_available", lambda: False)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)

    def unexpected_listener(*, on_press):
        raise AssertionError("pynput must not start without the X RECORD extension")

    monkeypatch.setattr(ki, "keyboard", SimpleNamespace(Listener=unexpected_listener))

    listener = create_key_listener(lambda name: None)

    assert isinstance(listener, TerminalKeyListener)


class _StartupDeadPynputListener:
    """Match pynput after record_create_context fails: running stays true, thread is dead."""

    running = True

    def __init__(self):
        self.start_called = False
        self.stop_called = False

    def start(self):
        self.start_called = True

    def is_alive(self):
        return False

    def stop(self):
        self.stop_called = True
        raise AssertionError("stopping this pynput state would block forever in listener.wait()")


def _install_startup_dead_pynput(monkeypatch):
    dead = _StartupDeadPynputListener()
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: True)
    monkeypatch.setattr(ki, "_x11_record_available", lambda: True)
    monkeypatch.setattr(ki, "pynput_listener_is_trusted", lambda listener: True)
    monkeypatch.setattr(ki, "keyboard", SimpleNamespace(Listener=lambda **kwargs: dead))
    return dead


def test_startup_dead_pynput_falls_back_to_terminal(monkeypatch):
    dead = _install_startup_dead_pynput(monkeypatch)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)

    listener = create_key_listener(lambda name: None)

    assert dead.start_called is True
    assert dead.stop_called is False
    assert isinstance(listener, TerminalKeyListener)


def test_startup_dead_pynput_returns_none_without_terminal(monkeypatch):
    dead = _install_startup_dead_pynput(monkeypatch)
    _set_tty(monkeypatch, is_tty=False)

    listener = create_key_listener(lambda name: None)

    assert dead.start_called is True
    assert dead.stop_called is False
    assert listener is None


def test_key_listener_liveness_uses_thread_state_not_stale_running_flag():
    dead = _StartupDeadPynputListener()

    assert dead.running is True
    assert key_listener_is_alive(dead) is False
    assert key_listener_is_alive(None) is False


@pytest.mark.skipif(not ki._TERMIOS_AVAILABLE, reason="requires a POSIX pseudo-terminal")
def test_terminal_listener_reads_space_and_tab_from_real_pty(monkeypatch):
    """Exercise fileno/termios/select/os.read/thread dispatch instead of calling _on_key directly."""
    master_fd, slave_fd = pty.openpty()
    slave = os.fdopen(slave_fd, "r", encoding="utf-8", buffering=1)
    monkeypatch.setattr(sys, "stdin", slave)
    received = []
    complete = threading.Event()

    def on_key(name):
        received.append(name)
        if received == ["space", "tab"]:
            complete.set()

    listener = TerminalKeyListener(on_key)
    try:
        listener.start()
        assert key_listener_is_alive(listener) is True
        os.write(master_fd, b" \t")
        assert complete.wait(timeout=1.0), received
    finally:
        listener.stop()
        slave.close()
        os.close(master_fd)

    assert received == ["space", "tab"]
    assert key_listener_is_alive(listener) is False
