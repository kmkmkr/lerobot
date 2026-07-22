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
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.openarm_follower.openarm_dynamics import bilateral_friction_torque
from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader, BiOpenArmLeaderConfig
from lerobot.teleoperators.openarm_leader import OpenArmLeader, OpenArmLeaderConfig
from lerobot.teleoperators.openarm_leader.config_openarm_leader import (
    BILATERAL_LEADER_FC,
    BILATERAL_LEADER_FC_SCALE,
    BILATERAL_LEADER_FO,
    BILATERAL_LEADER_FO_SCALE,
    BILATERAL_LEADER_FRICTION_K,
    BILATERAL_LEADER_FV,
    BILATERAL_LEADER_FV_SCALE,
    BILATERAL_LEADER_GRAVITY_SCALE,
    BILATERAL_LEADER_KD,
    BILATERAL_LEADER_KP,
    OpenArmLeaderConfigBase,
)

_MODULE = "lerobot.teleoperators.openarm_leader.openarm_leader"
_BI_MODULE = "lerobot.teleoperators.bi_openarm_leader.bi_openarm_leader"


def _config(tmp_path, **overrides) -> OpenArmLeaderConfig:
    values = {
        "port": "can0",
        "id": "leader",
        "calibration_dir": tmp_path,
        "side": "left",
        "gravity_compensation": False,
        "friction_compensation": False,
    }
    values.update(overrides)
    return OpenArmLeaderConfig(**values)


def _make_leader(tmp_path, config: OpenArmLeaderConfig | None = None) -> tuple[OpenArmLeader, MagicMock]:
    bus = MagicMock(name="DamiaoMotorsBusMock")
    bus.is_connected = False

    def _connect() -> None:
        bus.is_connected = True

    def _disconnect(*_args, **_kwargs) -> None:
        bus.is_connected = False

    def _bus_factory(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        bus.calibration_argument = kwargs["calibration"]
        bus.sync_read_all_states.return_value = {
            motor: {"position": 0.0, "velocity": 0.0, "torque": 0.0} for motor in bus.motors
        }
        return bus

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    with patch(f"{_MODULE}.DamiaoMotorsBus", side_effect=_bus_factory):
        leader = OpenArmLeader(config or _config(tmp_path))
    return leader, bus


def _feedback(**positions: float) -> dict[str, float]:
    return {
        f"{motor}.pos": positions.get(motor, 0.0)
        for motor in (
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "gripper",
        )
    }


def test_defaults_match_validated_native_bilateral_leader(tmp_path):
    config = OpenArmLeaderConfig(port="can0", id="leader", calibration_dir=tmp_path, side="left")

    assert not config.manual_control
    assert config.position_kp == list(BILATERAL_LEADER_KP)
    assert config.position_kd == list(BILATERAL_LEADER_KD)
    assert config.gravity_scale == list(BILATERAL_LEADER_GRAVITY_SCALE)
    assert config.friction_fc == list(BILATERAL_LEADER_FC)
    assert config.friction_k == list(BILATERAL_LEADER_FRICTION_K)
    assert config.friction_fv == list(BILATERAL_LEADER_FV)
    assert config.friction_fo == list(BILATERAL_LEADER_FO)
    assert config.friction_fc_scale == list(BILATERAL_LEADER_FC_SCALE)
    assert config.friction_fv_scale == list(BILATERAL_LEADER_FV_SCALE)
    assert config.friction_fo_scale == list(BILATERAL_LEADER_FO_SCALE)


def test_connect_never_writes_motor_zero(tmp_path):
    leader, bus = _make_leader(tmp_path)

    leader.connect()

    assert leader.is_connected
    assert leader.is_calibrated
    assert bus.calibration_argument == {}
    bus.set_zero_position.assert_not_called()
    bus.enable_torque.assert_called_once_with()
    assert bus._mit_control_batch.call_count == 1


def test_generic_lerobot_calibration_is_rejected_without_writing_zero(tmp_path):
    leader, bus = _make_leader(tmp_path)

    with pytest.raises(RuntimeError, match="openarm-can-zero-position-calibration"):
        leader.calibrate()

    bus.set_zero_position.assert_not_called()


def test_dynamics_failure_happens_before_bus_connect(tmp_path):
    config = _config(tmp_path, gravity_compensation=True, dynamics_urdf_path="missing.urdf")
    leader, bus = _make_leader(tmp_path, config)

    with (
        patch(f"{_MODULE}.OpenArmGravityModel", side_effect=RuntimeError("invalid dynamics")),
        pytest.raises(RuntimeError, match="invalid dynamics"),
    ):
        leader.connect()

    bus.connect.assert_not_called()
    bus.enable_torque.assert_not_called()


def test_send_feedback_uses_leader_pd_and_validated_j7_friction(tmp_path):
    config = _config(tmp_path, friction_compensation=True)
    leader, bus = _make_leader(tmp_path, config)
    leader.connect()
    bus._mit_control_batch.reset_mock()
    bus.sync_read_all_states.return_value["joint_7"]["velocity"] = 30.0
    leader.get_action()

    leader.send_feedback(_feedback(joint_7=12.0))

    commands = bus._mit_control_batch.call_args.args[0]
    expected_friction = bilateral_friction_torque(
        math.radians(30.0),
        BILATERAL_LEADER_FC[6] * BILATERAL_LEADER_FC_SCALE[6],
        BILATERAL_LEADER_FRICTION_K[6],
        BILATERAL_LEADER_FV[6] * BILATERAL_LEADER_FV_SCALE[6],
        BILATERAL_LEADER_FO[6] * BILATERAL_LEADER_FO_SCALE[6],
        config.friction_tanh_coefficient,
    )
    assert commands["joint_7"] == pytest.approx(
        (BILATERAL_LEADER_KP[6], BILATERAL_LEADER_KD[6], 12.0, 0.0, expected_friction)
    )


def test_send_feedback_applies_side_specific_urdf_gravity(tmp_path):
    config = _config(tmp_path, gravity_compensation=True)
    gravity_model = MagicMock()
    gravity_model.gravity_torques.return_value = (1.0,) * 7
    with patch(f"{_MODULE}.OpenArmGravityModel", return_value=gravity_model) as model_cls:
        leader, bus = _make_leader(tmp_path, config)
        leader.connect()

    model_cls.assert_called_once_with(config.dynamics_urdf_path, "left", config.gravity_m_s2)
    bus._mit_control_batch.reset_mock()
    leader.send_feedback(_feedback())

    commands = bus._mit_control_batch.call_args.args[0]
    assert commands["joint_1"][4] == pytest.approx(1.0)
    assert commands["joint_7"][4] == pytest.approx(BILATERAL_LEADER_GRAVITY_SCALE[6])
    assert commands["gripper"][4] == pytest.approx(0.0)


def test_invalid_feedback_disables_leader_before_raising(tmp_path):
    leader, bus = _make_leader(tmp_path)
    leader.connect()
    bus.disable_torque.reset_mock()

    with pytest.raises(ValueError, match="physical range"):
        leader.send_feedback(_feedback(joint_1=-201.0))

    bus.disable_torque.assert_called_once_with()
    bus._mit_control_batch.assert_called_once()


def test_feedback_velocity_outside_mit_range_disables_leader(tmp_path):
    leader, bus = _make_leader(tmp_path)
    leader.connect()
    bus.disable_torque.reset_mock()
    feedback = _feedback()
    feedback["joint_3.vel"] = math.degrees(8.0) + 1.0

    with pytest.raises(ValueError, match="feedback velocity.*MIT range"):
        leader.send_feedback(feedback)

    bus.disable_torque.assert_called_once_with()


def test_manual_control_keeps_legacy_free_motion_mode(tmp_path):
    leader, bus = _make_leader(tmp_path, _config(tmp_path, side=None, manual_control=True))
    leader.connect()

    assert leader.feedback_features == {}
    assert not leader.requires_continuous_feedback
    bus.disable_torque.assert_called_once_with()
    bus.enable_torque.assert_not_called()
    with pytest.raises(RuntimeError, match="manual_control"):
        leader.send_feedback(_feedback())


def test_bimanual_feedback_routes_sides_and_disables_both_on_error(tmp_path):
    left_arm = MagicMock(name="left_arm")
    right_arm = MagicMock(name="right_arm")
    left_arm.is_connected = True
    right_arm.is_connected = True
    left_arm.requires_continuous_feedback = True
    right_arm.requires_continuous_feedback = True
    left_arm.feedback_features = {"joint_1.pos": float}
    right_arm.feedback_features = {"joint_1.pos": float}
    right_arm.send_feedback.side_effect = RuntimeError("right feedback failed")
    config = BiOpenArmLeaderConfig(
        id="leader",
        calibration_dir=tmp_path,
        left_arm_config=OpenArmLeaderConfigBase(port="can1", gravity_compensation=False),
        right_arm_config=OpenArmLeaderConfigBase(port="can0", gravity_compensation=False),
    )
    with patch(f"{_BI_MODULE}.OpenArmLeader", side_effect=(left_arm, right_arm)) as factory:
        leader = BiOpenArmLeader(config)

    assert factory.call_args_list[0].args[0].side == "left"
    assert factory.call_args_list[1].args[0].side == "right"
    assert leader.requires_continuous_feedback
    with pytest.raises(RuntimeError, match="right feedback failed"):
        leader.send_feedback({"left_joint_1.pos": 1.0, "right_joint_1.pos": 2.0})

    left_arm.send_feedback.assert_called_once_with({"joint_1.pos": 1.0})
    right_arm.send_feedback.assert_called_once_with({"joint_1.pos": 2.0})
    left_arm.disable_torque.assert_called_once_with()
    right_arm.disable_torque.assert_called_once_with()


def test_bimanual_rejects_mixed_free_motion_and_bilateral_modes(tmp_path):
    config = BiOpenArmLeaderConfig(
        id="leader",
        calibration_dir=tmp_path,
        left_arm_config=OpenArmLeaderConfigBase(port="can1", manual_control=True, gravity_compensation=False),
        right_arm_config=OpenArmLeaderConfigBase(port="can0", gravity_compensation=False),
    )

    with pytest.raises(ValueError, match="same manual_control mode"):
        BiOpenArmLeader(config)


def test_bimanual_shutdown_error_hold_survives_disconnect_cleanup(tmp_path):
    left_arm = MagicMock(name="left_arm")
    right_arm = MagicMock(name="right_arm")
    for arm in (left_arm, right_arm):
        arm.bus.is_connected = True
        arm.config.disable_torque_on_disconnect = True
        arm.get_action.return_value = {
            "joint_1.pos": 1.0,
            "joint_1.vel": 2.0,
            "joint_1.torque": 3.0,
        }
    config = BiOpenArmLeaderConfig(
        id="leader",
        calibration_dir=tmp_path,
        left_arm_config=OpenArmLeaderConfigBase(port="can1", gravity_compensation=False),
        right_arm_config=OpenArmLeaderConfigBase(port="can0", gravity_compensation=False),
    )
    with patch(f"{_BI_MODULE}.OpenArmLeader", side_effect=(left_arm, right_arm)):
        leader = BiOpenArmLeader(config)

    leader.hold_position_after_shutdown_error()

    assert not config.left_arm_config.disable_torque_on_disconnect
    assert not config.right_arm_config.disable_torque_on_disconnect
    assert not left_arm.config.disable_torque_on_disconnect
    assert not right_arm.config.disable_torque_on_disconnect
    left_arm.send_feedback.assert_called_once_with({"joint_1.pos": 1.0})
    right_arm.send_feedback.assert_called_once_with({"joint_1.pos": 1.0})
