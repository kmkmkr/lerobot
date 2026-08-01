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

from __future__ import annotations

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout.strategies.dagger_web_ui import DAggerWebUI


def _make_ui(events: list[str], *, preview_enabled: bool = True) -> DAggerWebUI:
    def record(name: str):
        def handler() -> bool:
            events.append(name)
            return True

        return handler

    return DAggerWebUI(
        port=0,
        auto_port=False,
        preview_enabled=preview_enabled,
        preview_fps=30.0,
        jpeg_quality=80,
        command_handlers={
            "pause": record("pause"),
            "resume": record("resume"),
            "start-correction": record("start-correction"),
            "finish-correction": record("finish-correction"),
            "discard-correction": record("discard-correction"),
            "upload": record("upload"),
            "stop": record("stop"),
        },
        task="pick and place",
        target_episodes=2,
        discard_enabled=True,
    )


def _json(url: str) -> dict:
    with urlopen(url, timeout=2.0) as response:
        return json.loads(response.read())


def _post(ui: DAggerWebUI, command: str, *, include_token: bool = True) -> int:
    headers = {"Origin": ui.url}
    if include_token:
        headers["X-DAgger-CSRF"] = ui._csrf_token
    request = Request(
        f"{ui.url}/api/command/{command}",
        data=b"",
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=2.0) as response:
        return response.status


def test_dagger_web_ui_serves_state_and_routes_only_tokened_commands() -> None:
    events: list[str] = []
    ui = _make_ui(events, preview_enabled=False)
    ui.start()
    try:
        with urlopen(ui.url, timeout=2.0) as response:
            html = response.read().decode()
        assert "LeRobot DAgger operator UI" in html
        assert "Discard correction" in html
        assert ui.url.startswith("http://127.0.0.1:")

        state = _json(f"{ui.url}/api/state")
        assert state["phase"] == "autonomous"
        assert state["target_episodes"] == 2
        assert state["preview_enabled"] is False
        assert state["discard_enabled"] is True

        with pytest.raises(HTTPError) as exc_info:
            _post(ui, "pause", include_token=False)
        assert exc_info.value.code == 403
        assert events == []

        assert _post(ui, "pause") == 202
        assert events == ["pause"]
        assert _json(f"{ui.url}/api/state")["pending_command"] == "pause"

        with pytest.raises(HTTPError) as exc_info:
            _post(ui, "resume")
        assert exc_info.value.code == 409
        assert events == ["pause"]

        ui.publish_status(phase="paused", recorded_episodes=1, save_pending=True)
        ui.acknowledge_transition()
        state = _json(f"{ui.url}/api/state")
        assert state["phase"] == "paused"
        assert state["recorded_episodes"] == 1
        assert state["save_pending"] is True
        assert state["pending_command"] is None

        assert _post(ui, "start-correction") == 202
        ui.publish_status(phase="correcting", recorded_episodes=1, save_pending=False)
        ui.acknowledge_transition()
        assert _post(ui, "discard-correction") == 202
        assert events[-2:] == ["start-correction", "discard-correction"]
    finally:
        ui.stop()


def test_dagger_web_ui_encodes_latest_observation_without_opening_a_camera() -> None:
    ui = _make_ui([])
    ui.start()
    try:
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        ui.submit_observation({"front": image, "joint_1.pos": 0.0})

        deadline = time.monotonic() + 2.0
        camera = None
        while time.monotonic() < deadline:
            camera = ui.snapshot()["cameras"].get("front")
            if camera is not None:
                break
            time.sleep(0.01)
        assert camera is not None
        assert camera["width"] == 32
        assert camera["height"] == 24

        with urlopen(f"{ui.url}/api/camera/front", timeout=2.0) as response:
            encoded = np.frombuffer(response.read(), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        assert decoded.shape == (24, 32, 3)
    finally:
        ui.stop()


def test_dagger_web_ui_rejects_invalid_transition_callback() -> None:
    ui = DAggerWebUI(
        port=0,
        auto_port=False,
        preview_enabled=False,
        preview_fps=5.0,
        jpeg_quality=80,
        command_handlers={
            "pause": lambda: False,
            "resume": lambda: False,
            "start-correction": lambda: False,
            "finish-correction": lambda: False,
            "discard-correction": lambda: False,
            "upload": lambda: True,
            "stop": lambda: True,
        },
        task="task",
        target_episodes=1,
        discard_enabled=True,
    )
    ui.start()
    try:
        with pytest.raises(HTTPError) as exc_info:
            _post(ui, "pause")
        assert exc_info.value.code == 409
        assert ui.snapshot()["pending_command"] is None
    finally:
        ui.stop()
