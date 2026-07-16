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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower import (
    OPENARM_V1_COORDINATE_FRAME,
    OPENARM_V1_PHYSICAL_JOINT_LIMITS,
    OPENARM_V1_SAFE_JOINT_LIMITS,
    OpenArmFollower,
    OpenArmFollowerConfig,
    OpenArmFollowerConfigBase,
)

_MODULE = "lerobot.robots.openarm_follower.openarm_follower"


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="DamiaoMotorsBusMock")
    bus.is_connected = False

    def _connect() -> None:
        bus.is_connected = True

    def _disconnect(_disable_torque: bool = True) -> None:
        bus.is_connected = False

    @contextmanager
    def _torque_disabled():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.torque_disabled.side_effect = _torque_disabled
    return bus


def _make_follower(config: OpenArmFollowerConfig) -> tuple[OpenArmFollower, MagicMock]:
    bus = _make_bus_mock()

    def _bus_factory(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        bus.calibration_argument = kwargs["calibration"]
        return bus

    with patch(f"{_MODULE}.DamiaoMotorsBus", side_effect=_bus_factory):
        return OpenArmFollower(config), bus


def test_side_selects_v1_safe_limits_without_sharing_mutable_state(tmp_path):
    robot, _ = _make_follower(OpenArmFollowerConfig(port="can0", side="left", calibration_dir=tmp_path))

    assert robot.config.coordinate_frame == OPENARM_V1_COORDINATE_FRAME
    assert robot.config.joint_limits == OPENARM_V1_SAFE_JOINT_LIMITS["left"]
    assert robot.config.joint_limits is not OPENARM_V1_SAFE_JOINT_LIMITS["left"]
    assert robot.config.joint_limits["gripper"] == (-60.0, 0.0)


def test_explicit_limits_are_preserved_when_side_is_set(tmp_path):
    custom_limits = dict(OPENARM_V1_SAFE_JOINT_LIMITS["left"])
    custom_limits["joint_1"] = (-70.0, 70.0)

    robot, _ = _make_follower(
        OpenArmFollowerConfig(port="can0", side="left", joint_limits=custom_limits, calibration_dir=tmp_path)
    )

    assert robot.config.joint_limits == custom_limits


def test_limits_outside_v1_physical_range_are_rejected(tmp_path):
    invalid_limits = dict(OPENARM_V1_SAFE_JOINT_LIMITS["right"])
    physical_minimum, physical_maximum = OPENARM_V1_PHYSICAL_JOINT_LIMITS["right"]["joint_6"]
    invalid_limits["joint_6"] = (physical_minimum - 1.0, physical_maximum)

    with pytest.raises(ValueError, match="exceeds the OpenArm v1 right physical limit"):
        _make_follower(
            OpenArmFollowerConfig(
                port="can0", side="right", joint_limits=invalid_limits, calibration_dir=tmp_path
            )
        )


def test_connect_never_writes_motor_zero(tmp_path):
    robot, bus = _make_follower(OpenArmFollowerConfig(port="can0", side="right", calibration_dir=tmp_path))

    robot.connect()

    assert robot.is_connected
    assert robot.is_calibrated
    assert bus.calibration_argument == {}
    bus.set_zero_position.assert_not_called()


def test_send_action_clips_in_v1_motor_zero_degrees(tmp_path):
    robot, bus = _make_follower(OpenArmFollowerConfig(port="can0", side="right", calibration_dir=tmp_path))
    robot.connect()

    sent = robot.send_action({"joint_6.pos": -46.0})

    assert sent == {"joint_6.pos": -40.0}
    assert bus._mit_control_batch.call_args.args[0]["joint_6"][2] == -40.0


def test_generic_lerobot_calibration_is_rejected_without_writing_zero(tmp_path):
    robot, bus = _make_follower(OpenArmFollowerConfig(port="can0", side="right", calibration_dir=tmp_path))

    with pytest.raises(RuntimeError, match="openarm-can-zero-position-calibration"):
        robot.calibrate()

    bus.set_zero_position.assert_not_called()


def test_unknown_coordinate_frame_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported OpenArm coordinate frame"):
        _make_follower(
            OpenArmFollowerConfig(
                port="can0", side="right", coordinate_frame="joint_offset", calibration_dir=tmp_path
            )
        )


def test_bimanual_children_use_bimanual_calibration_directory(tmp_path):
    buses = [_make_bus_mock(), _make_bus_mock()]

    def _bus_factory(*_args, **kwargs):
        bus = buses.pop(0)
        bus.motors = kwargs["motors"]
        return bus

    config = BiOpenArmFollowerConfig(
        id="test_bi_openarm",
        calibration_dir=tmp_path,
        left_arm_config=OpenArmFollowerConfigBase(port="can0", side="left"),
        right_arm_config=OpenArmFollowerConfigBase(port="can1", side="right"),
    )
    with patch(f"{_MODULE}.DamiaoMotorsBus", side_effect=_bus_factory):
        robot = BiOpenArmFollower(config)

    assert robot.left_arm.calibration_dir == tmp_path
    assert robot.right_arm.calibration_dir == tmp_path
    assert robot.left_arm.config.joint_limits == OPENARM_V1_SAFE_JOINT_LIMITS["left"]
    assert robot.right_arm.config.joint_limits == OPENARM_V1_SAFE_JOINT_LIMITS["right"]
