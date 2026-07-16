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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.openarm_leader import OpenArmLeader, OpenArmLeaderConfig

_MODULE = "lerobot.teleoperators.openarm_leader.openarm_leader"


def _make_leader(tmp_path) -> tuple[OpenArmLeader, MagicMock]:
    bus = MagicMock(name="DamiaoMotorsBusMock")
    bus.is_connected = False

    def _connect() -> None:
        bus.is_connected = True

    def _disconnect(*_args, **_kwargs) -> None:
        bus.is_connected = False

    def _bus_factory(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        bus.calibration_argument = kwargs["calibration"]
        return bus

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    with patch(f"{_MODULE}.DamiaoMotorsBus", side_effect=_bus_factory):
        leader = OpenArmLeader(OpenArmLeaderConfig(port="can0", id="leader", calibration_dir=tmp_path))
    return leader, bus


def test_connect_never_writes_motor_zero(tmp_path):
    leader, bus = _make_leader(tmp_path)

    leader.connect()

    assert leader.is_connected
    assert leader.is_calibrated
    assert bus.calibration_argument == {}
    bus.set_zero_position.assert_not_called()


def test_generic_lerobot_calibration_is_rejected_without_writing_zero(tmp_path):
    leader, bus = _make_leader(tmp_path)

    with pytest.raises(RuntimeError, match="openarm-can-zero-position-calibration"):
        leader.calibrate()

    bus.set_zero_position.assert_not_called()
