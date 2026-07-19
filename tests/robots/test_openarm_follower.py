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

import math
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
from lerobot.robots.openarm_follower.config_openarm_follower import (
    BILATERAL_FOLLOWER_FC,
    BILATERAL_FOLLOWER_FC_SCALE,
    BILATERAL_FOLLOWER_FO,
    BILATERAL_FOLLOWER_FO_SCALE,
    BILATERAL_FOLLOWER_FRICTION_K,
    BILATERAL_FOLLOWER_FV,
    BILATERAL_FOLLOWER_FV_SCALE,
    BILATERAL_FOLLOWER_GRAVITY_SCALE,
    BILATERAL_FOLLOWER_KD,
    BILATERAL_FOLLOWER_KP,
    BILATERAL_FRICTION_TANH_COEFFICIENT,
    BILATERAL_GRAVITY_M_S2,
)
from lerobot.robots.openarm_follower.openarm_dynamics import (
    OpenArmGravityModel,
    bilateral_friction_torque,
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


def _make_follower(
    config: OpenArmFollowerConfig, *, disable_compensation: bool = True
) -> tuple[OpenArmFollower, MagicMock]:
    if disable_compensation:
        config.gravity_compensation = False
        config.friction_compensation = False
    bus = _make_bus_mock()

    def _bus_factory(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        bus.calibration_argument = kwargs["calibration"]
        return bus

    with patch(f"{_MODULE}.DamiaoMotorsBus", side_effect=_bus_factory):
        return OpenArmFollower(config), bus


def _write_test_dynamics_urdf(path) -> None:
    links = [f'<link name="openarm_left_link{index}"/>' for index in range(1, 8)]
    joints = []
    parent = "openarm_body_link0"
    for index in range(1, 8):
        child = f"openarm_left_link{index}"
        joints.append(
            f'<joint name="openarm_left_joint{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            '<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>'
            "</joint>"
        )
        parent = child
    path.write_text(
        '<robot name="test">'
        '<link name="openarm_body_link0"/>'
        + "".join(links)
        + '<link name="openarm_left_hand"><inertial><origin xyz="1 0 0"/>'
        '<mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
        "</inertial></link>" + "".join(joints) + '<joint name="left_openarm_hand_joint" type="fixed">'
        '<parent link="openarm_left_link7"/><child link="openarm_left_hand"/>'
        '<origin xyz="0 0 0" rpy="0 0 0"/></joint>'
        "</robot>",
        encoding="utf-8",
    )


def test_lerobot_defaults_match_native_bilateral_follower():
    config = OpenArmFollowerConfig(port="can0", side="left")

    assert config.position_kp == list(BILATERAL_FOLLOWER_KP)
    assert config.position_kd == list(BILATERAL_FOLLOWER_KD)
    assert config.trajectory_position_kp == list(BILATERAL_FOLLOWER_KP)
    assert config.trajectory_position_kd == list(BILATERAL_FOLLOWER_KD)
    assert config.gravity_scale == list(BILATERAL_FOLLOWER_GRAVITY_SCALE)
    assert config.friction_fc == list(BILATERAL_FOLLOWER_FC)
    assert config.friction_k == list(BILATERAL_FOLLOWER_FRICTION_K)
    assert config.friction_fv == list(BILATERAL_FOLLOWER_FV)
    assert config.friction_fo == list(BILATERAL_FOLLOWER_FO)
    assert config.friction_fc_scale == list(BILATERAL_FOLLOWER_FC_SCALE)
    assert config.friction_fv_scale == list(BILATERAL_FOLLOWER_FV_SCALE)
    assert config.friction_fo_scale == list(BILATERAL_FOLLOWER_FO_SCALE)
    assert config.gravity_m_s2 == BILATERAL_GRAVITY_M_S2
    assert config.friction_tanh_coefficient == BILATERAL_FRICTION_TANH_COEFFICIENT
    assert config.gravity_compensation
    assert config.friction_compensation


def test_gravity_model_matches_static_single_mass_chain(tmp_path):
    urdf_path = tmp_path / "openarm_test.urdf"
    _write_test_dynamics_urdf(urdf_path)

    model = OpenArmGravityModel(urdf_path, "left")

    assert model.gravity_torques((0.0,) * 7) == pytest.approx((-9.81,) * 7)


def test_missing_gravity_urdf_fails_before_connecting_bus(tmp_path):
    config = OpenArmFollowerConfig(
        port="can0",
        side="left",
        calibration_dir=tmp_path,
        dynamics_urdf_path=str(tmp_path / "missing.urdf"),
    )
    robot, bus = _make_follower(config, disable_compensation=False)

    with pytest.raises(ValueError, match="Failed to load OpenArm dynamics URDF"):
        robot.connect()

    bus.connect.assert_not_called()


def test_send_action_applies_validated_bilateral_j7_friction(tmp_path):
    config = OpenArmFollowerConfig(
        port="can0",
        side="right",
        calibration_dir=tmp_path,
        gravity_compensation=False,
        friction_compensation=True,
    )
    robot, bus = _make_follower(config, disable_compensation=False)
    robot.connect()
    bus.sync_read_all_states.return_value = {
        motor: {
            "position": 0.0,
            "velocity": math.degrees(1.0),
            "torque": 0.0,
            "temp_mos": 0.0,
            "temp_rotor": 0.0,
        }
        for motor in bus.motors
    }

    robot.send_action({"joint_7.pos": 0.0})

    expected = bilateral_friction_torque(
        1.0,
        BILATERAL_FOLLOWER_FC[6] * BILATERAL_FOLLOWER_FC_SCALE[6],
        BILATERAL_FOLLOWER_FRICTION_K[6],
        BILATERAL_FOLLOWER_FV[6] * BILATERAL_FOLLOWER_FV_SCALE[6],
        BILATERAL_FOLLOWER_FO[6] * BILATERAL_FOLLOWER_FO_SCALE[6],
        BILATERAL_FRICTION_TANH_COEFFICIENT,
    )
    assert bus._mit_control_batch.call_args.args[0]["joint_7"][4] == pytest.approx(expected)


def test_send_action_applies_urdf_gravity_and_friction(tmp_path):
    urdf_path = tmp_path / "openarm_test.urdf"
    _write_test_dynamics_urdf(urdf_path)
    config = OpenArmFollowerConfig(
        port="can0",
        side="left",
        calibration_dir=tmp_path,
        dynamics_urdf_path=str(urdf_path),
        friction_fc=[0.0] * 8,
        friction_fv=[0.0] * 8,
        friction_fo=[0.0] * 8,
    )
    robot, bus = _make_follower(config, disable_compensation=False)
    robot.connect()
    bus.sync_read_all_states.return_value = {
        motor: {
            "position": 0.0,
            "velocity": 0.0,
            "torque": 0.0,
            "temp_mos": 0.0,
            "temp_rotor": 0.0,
        }
        for motor in bus.motors
    }

    robot.send_action({"joint_1.pos": 0.0})

    assert bus._mit_control_batch.call_args.args[0]["joint_1"][4] == pytest.approx(-9.81)


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


def test_trajectory_commands_use_gains_separate_from_policy_actions(tmp_path):
    robot, bus = _make_follower(
        OpenArmFollowerConfig(
            port="can0",
            side="right",
            calibration_dir=tmp_path,
            position_kp=[101.0] * 8,
            position_kd=[1.1] * 8,
            trajectory_position_kp=[51.0] * 8,
            trajectory_position_kd=[0.6] * 8,
        )
    )
    robot.connect()

    trajectory_kp, trajectory_kd = robot.trajectory_position_gains()
    robot.send_action({"joint_1.pos": 0.0}, trajectory_kp, trajectory_kd)
    trajectory_command = bus._mit_control_batch.call_args.args[0]["joint_1"]
    robot.send_action({"joint_1.pos": 0.0})
    policy_command = bus._mit_control_batch.call_args.args[0]["joint_1"]

    assert trajectory_command[:2] == (51.0, 0.6)
    assert policy_command[:2] == (101.0, 1.1)


def test_validated_trajectory_can_bypass_policy_relative_target_limit(tmp_path):
    robot, bus = _make_follower(
        OpenArmFollowerConfig(
            port="can0",
            side="right",
            calibration_dir=tmp_path,
            max_relative_target=2.0,
        )
    )
    robot.connect()
    bus.sync_read.return_value = {"joint_4": 16.0}

    limited = robot.send_action({"joint_4.pos": 18.315})

    assert limited == {"joint_4.pos": 18.0}
    assert bus._mit_control_batch.call_args.args[0]["joint_4"][2] == 18.0
    bus.sync_read.assert_called_once_with("Present_Position")
    bus.sync_read.reset_mock()

    sent = robot.send_action(
        {"joint_4.pos": 18.315},
        apply_max_relative_target=False,
    )

    assert sent == {"joint_4.pos": 18.315}
    assert bus._mit_control_batch.call_args.args[0]["joint_4"][2] == 18.315
    bus.sync_read.assert_not_called()


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


def test_bimanual_children_keep_trajectory_gains_separate_from_policy_gains(tmp_path):
    buses = [_make_bus_mock(), _make_bus_mock()]

    def _bus_factory(*_args, **kwargs):
        bus = buses.pop(0)
        bus.motors = kwargs["motors"]
        return bus

    policy_kp = [101.0] * 8
    trajectory_kp = [51.0] * 8
    config = BiOpenArmFollowerConfig(
        calibration_dir=tmp_path,
        left_arm_config=OpenArmFollowerConfigBase(
            port="can0", side="left", position_kp=policy_kp, trajectory_position_kp=trajectory_kp
        ),
        right_arm_config=OpenArmFollowerConfigBase(
            port="can1", side="right", position_kp=policy_kp, trajectory_position_kp=trajectory_kp
        ),
    )
    with patch(f"{_MODULE}.DamiaoMotorsBus", side_effect=_bus_factory):
        robot = BiOpenArmFollower(config)

    assert robot.left_arm.config.position_kp == policy_kp
    assert robot.left_arm.config.trajectory_position_kp == trajectory_kp
    assert robot.right_arm.config.position_kp == policy_kp
    assert robot.right_arm.config.trajectory_position_kp == trajectory_kp


def test_bimanual_deployment_targets_bypass_policy_relative_target_limit(tmp_path):
    buses = [_make_bus_mock(), _make_bus_mock()]

    def _bus_factory(*_args, **kwargs):
        bus = buses.pop(0)
        bus.motors = kwargs["motors"]
        return bus

    config = BiOpenArmFollowerConfig(
        calibration_dir=tmp_path,
        left_arm_config=OpenArmFollowerConfigBase(port="can0", side="left", max_relative_target=2.0),
        right_arm_config=OpenArmFollowerConfigBase(port="can1", side="right", max_relative_target=2.0),
    )
    with patch(f"{_MODULE}.DamiaoMotorsBus", side_effect=_bus_factory):
        robot = BiOpenArmFollower(config)

    targets = {"left": (0.0,) * 8, "right": (0.0,) * 8}
    robot._read_deployment_positions = MagicMock(return_value=targets)
    for arm in (robot.left_arm, robot.right_arm):
        arm.send_action = MagicMock(side_effect=lambda action, **_kwargs: action)

    robot._send_deployment_targets(targets, phase="test", reference_time_s=0.0)

    for arm in (robot.left_arm, robot.right_arm):
        assert arm.send_action.call_args.kwargs["apply_max_relative_target"] is False


def test_bimanual_action_error_disables_both_followers(tmp_path):
    left_bus = _make_bus_mock()
    right_bus = _make_bus_mock()
    buses = [left_bus, right_bus]

    def _bus_factory(*_args, **kwargs):
        bus = buses.pop(0)
        bus.motors = kwargs["motors"]
        return bus

    config = BiOpenArmFollowerConfig(
        calibration_dir=tmp_path,
        left_arm_config=OpenArmFollowerConfigBase(
            port="can0", side="left", gravity_compensation=False, friction_compensation=False
        ),
        right_arm_config=OpenArmFollowerConfigBase(
            port="can1", side="right", gravity_compensation=False, friction_compensation=False
        ),
    )
    with patch(f"{_MODULE}.DamiaoMotorsBus", side_effect=_bus_factory):
        robot = BiOpenArmFollower(config)
    left_bus.is_connected = True
    right_bus.is_connected = True
    robot.left_arm.send_action = MagicMock(side_effect=RuntimeError("compensation failed"))

    with pytest.raises(RuntimeError, match="compensation failed"):
        robot.send_action({"left_joint_1.pos": 0.0, "right_joint_1.pos": 0.0})

    left_bus.disable_torque.assert_called_once_with()
    right_bus.disable_torque.assert_called_once_with()
