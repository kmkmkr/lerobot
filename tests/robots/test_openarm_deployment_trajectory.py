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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower
from lerobot.robots.bi_openarm_follower.deployment_trajectory import (
    MOTOR_NAMES,
    DeploymentTrajectorySample,
    build_return_to_zero_trajectory,
    interpolate_deployment_trajectory,
    load_deployment_trajectory,
    validate_deployment_trajectory,
)
from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader


def _write_profile_csv(path, *, side: str = "left", coordinate_frame: str = "openarm_v1_motor_zero"):
    path.write_text(
        "\n".join(
            (
                f"# side={side}",
                f"# coordinate_frame={coordinate_frame}",
                "# position_unit=radian",
                "time_s,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,joint_7,gripper",
                "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
                "0.01,0.1,-0.1,0.0,0.2,0.0,0.0,0.0,-0.01",
            )
        ),
        encoding="utf-8",
    )


def test_loads_motor_zero_radians_as_degrees(tmp_path):
    csv_path = tmp_path / "left_arm.csv"
    _write_profile_csv(csv_path)

    samples = load_deployment_trajectory(csv_path, "left")

    assert samples[1].positions_deg[0] == pytest.approx(math.degrees(0.1))
    assert samples[1].positions_deg[1] == pytest.approx(math.degrees(-0.1))
    assert len(samples[1].positions_deg) == len(MOTOR_NAMES)


def test_rejects_an_unknown_coordinate_frame(tmp_path):
    csv_path = tmp_path / "left_arm.csv"
    _write_profile_csv(csv_path, coordinate_frame="legacy_offset")

    with pytest.raises(ValueError, match="coordinate_frame must be openarm_v1_motor_zero"):
        load_deployment_trajectory(csv_path, "left")


def test_validation_checks_scaled_velocity_and_motor_zero_start():
    limits = dict.fromkeys(MOTOR_NAMES, (-180.0, 180.0))
    samples = [
        DeploymentTrajectorySample(0.0, (0.0,) * len(MOTOR_NAMES)),
        DeploymentTrajectorySample(0.01, (1.0,) + (0.0,) * (len(MOTOR_NAMES) - 1)),
    ]

    validate_deployment_trajectory(samples, limits, speed_scale=1.0)
    with pytest.raises(ValueError, match="velocity limit exceeded"):
        validate_deployment_trajectory(samples, limits, speed_scale=10.0)

    offset_samples = [
        DeploymentTrajectorySample(0.0, (20.0,) + (0.0,) * (len(MOTOR_NAMES) - 1)),
        DeploymentTrajectorySample(0.01, (20.0,) + (0.0,) * (len(MOTOR_NAMES) - 1)),
    ]
    with pytest.raises(ValueError, match="does not begin near motor zero"):
        validate_deployment_trajectory(offset_samples, limits, speed_scale=1.0)


def test_interpolation_and_reverse_append_exact_motor_zero():
    samples = [
        DeploymentTrajectorySample(0.0, (0.0,) * len(MOTOR_NAMES)),
        DeploymentTrajectorySample(1.0, (10.0,) * len(MOTOR_NAMES)),
    ]

    midpoint = interpolate_deployment_trajectory(samples, 0.5)
    returned = build_return_to_zero_trajectory(samples, zero_transition_s=1.0)

    assert midpoint == pytest.approx((5.0,) * len(MOTOR_NAMES))
    assert returned[0] == DeploymentTrajectorySample(0.0, (10.0,) * len(MOTOR_NAMES))
    assert returned[-1] == DeploymentTrajectorySample(2.0, (0.0,) * len(MOTOR_NAMES))


def test_shutdown_threshold_requires_confirmation_before_motion():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(shutdown_task_pose_warn_deg=5.0)
    final = DeploymentTrajectorySample(1.0, (0.0,) * len(MOTOR_NAMES))
    robot._deployment_trajectories = {"left": [final], "right": [final]}
    robot._read_deployment_positions = MagicMock(
        return_value={
            "left": (6.0,) + (0.0,) * (len(MOTOR_NAMES) - 1),
            "right": (0.0,) * len(MOTOR_NAMES),
        }
    )
    robot._confirm_shutdown_return = MagicMock(return_value=False)
    robot._blend_deployment_positions = MagicMock()

    assert robot.finish_policy_deployment() is True
    robot._confirm_shutdown_return.assert_called_once_with()
    robot._blend_deployment_positions.assert_not_called()


def test_small_measured_blend_start_overshoot_is_clamped(caplog):
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_start_limit_tolerance_deg=1.5)
    limits = dict.fromkeys(MOTOR_NAMES, (-45.0, 45.0))
    limits["gripper"] = (-60.0, 0.0)
    robot.left_arm = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    robot.right_arm = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    start = {
        "left": (0.0,) * len(MOTOR_NAMES),
        "right": (0.0,) * (len(MOTOR_NAMES) - 1) + (1.257,),
    }

    clamped = robot._clamp_measured_blend_start(start, "shutdown-task-pose")

    assert clamped["right"][-1] == 0.0
    assert "Clamping measured OpenArm blend start" in caplog.text


def test_large_measured_blend_start_overshoot_is_rejected():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_start_limit_tolerance_deg=1.5)
    limits = dict.fromkeys(MOTOR_NAMES, (-45.0, 45.0))
    robot.left_arm = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    robot.right_arm = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    start = {
        "left": (0.0,) * 5 + (-47.0,) + (0.0,) * 2,
        "right": (0.0,) * len(MOTOR_NAMES),
    }

    with pytest.raises(RuntimeError, match="blend start exceeds the joint limit"):
        robot._clamp_measured_blend_start(start, "shutdown-task-pose")


def test_shutdown_motion_error_holds_instead_of_disabling():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(
        shutdown_task_pose_warn_deg=180.0,
        shutdown_task_pose_blend_s=10.0,
        shutdown_zero_transition_s=1.0,
        shutdown_replay_speed=0.25,
        hold_position_on_shutdown_error=True,
    )
    final = DeploymentTrajectorySample(1.0, (0.0,) * len(MOTOR_NAMES))
    robot._deployment_trajectories = {"left": [final], "right": [final]}
    robot._read_deployment_positions = MagicMock(
        return_value={
            "left": (0.0,) * len(MOTOR_NAMES),
            "right": (0.0,) * len(MOTOR_NAMES),
        }
    )
    robot._blend_deployment_positions = MagicMock(side_effect=RuntimeError("tracking error"))
    robot._hold_both_arms_after_shutdown_error = MagicMock()
    robot._disable_both_arms = MagicMock()

    with pytest.raises(RuntimeError, match="tracking error"):
        robot.finish_policy_deployment()

    robot._hold_both_arms_after_shutdown_error.assert_called_once_with()
    robot._disable_both_arms.assert_not_called()


def test_shutdown_initial_read_error_also_preserves_torque():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(hold_position_on_shutdown_error=True)
    final = DeploymentTrajectorySample(1.0, (0.0,) * len(MOTOR_NAMES))
    robot._deployment_trajectories = {"left": [final], "right": [final]}
    robot._read_deployment_positions = MagicMock(side_effect=RuntimeError("CAN read failed"))
    robot._hold_both_arms_after_shutdown_error = MagicMock()
    robot._disable_both_arms = MagicMock()

    with pytest.raises(RuntimeError, match="CAN read failed"):
        robot.finish_policy_deployment()

    robot._hold_both_arms_after_shutdown_error.assert_called_once_with()
    robot._disable_both_arms.assert_not_called()


def test_shutdown_error_hold_survives_disconnect_cleanup():
    robot = object.__new__(BiOpenArmFollower)
    left_source_config = SimpleNamespace(disable_torque_on_disconnect=True)
    right_source_config = SimpleNamespace(disable_torque_on_disconnect=True)
    robot.config = SimpleNamespace(
        left_arm_config=left_source_config,
        right_arm_config=right_source_config,
        deployment_start_limit_tolerance_deg=1.5,
    )
    positions = dict.fromkeys(MOTOR_NAMES, 0.0)

    def make_arm():
        arm_config = SimpleNamespace(disable_torque_on_disconnect=True)
        return SimpleNamespace(
            config=arm_config,
            bus=SimpleNamespace(
                is_connected=True,
                enable_torque=MagicMock(),
                disable_torque=MagicMock(),
            ),
            get_motor_positions=MagicMock(return_value=positions),
            trajectory_position_gains=MagicMock(return_value=({"joint_1": 10.0}, {"joint_1": 1.0})),
            send_action=MagicMock(return_value={f"{name}.pos": 0.0 for name in MOTOR_NAMES}),
        )

    robot.left_arm = make_arm()
    robot.right_arm = make_arm()

    robot._hold_both_arms_after_shutdown_error()

    assert left_source_config.disable_torque_on_disconnect is False
    assert right_source_config.disable_torque_on_disconnect is False
    assert robot.left_arm.config.disable_torque_on_disconnect is False
    assert robot.right_arm.config.disable_torque_on_disconnect is False
    for arm in (robot.left_arm, robot.right_arm):
        arm.get_motor_positions.assert_called_once_with(require_response=True)
        arm.bus.enable_torque.assert_called_once_with(num_retry=2, require_response=True)
        assert arm.send_action.call_count == 2
        arm.send_action.assert_called_with(
            {f"{name}.pos": 0.0 for name in MOTOR_NAMES},
            custom_kp={"joint_1": 10.0},
            custom_kd={"joint_1": 1.0},
            apply_joint_limits=False,
            apply_max_relative_target=False,
            require_response=True,
        )


def _make_fault_hold_arm(positions: dict[str, float]):
    return SimpleNamespace(
        config=SimpleNamespace(disable_torque_on_disconnect=True),
        bus=SimpleNamespace(
            is_connected=True,
            enable_torque=MagicMock(),
            disable_torque=MagicMock(),
        ),
        get_motor_positions=MagicMock(return_value=positions),
        trajectory_position_gains=MagicMock(return_value=({"joint_1": 10.0}, {"joint_1": 1.0})),
        send_action=MagicMock(side_effect=lambda action, **_kwargs: dict(action)),
    )


def test_fault_hold_preserves_a_physically_valid_pose_outside_policy_limits():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_start_limit_tolerance_deg=1.5)
    source_config = SimpleNamespace(disable_torque_on_disconnect=True)
    positions = dict.fromkeys(MOTOR_NAMES, 0.0)
    positions["joint_1"] = -150.0
    arm = _make_fault_hold_arm(positions)

    assert robot._secure_follower_current_position_hold("left", source_config, arm)

    assert arm.send_action.call_count == 2
    for sent_call in arm.send_action.call_args_list:
        assert sent_call.args[0]["joint_1.pos"] == -150.0
        assert sent_call.kwargs["apply_joint_limits"] is False
        assert sent_call.kwargs["apply_max_relative_target"] is False
        assert sent_call.kwargs["require_response"] is True


def test_fault_hold_clamps_only_small_gripper_sensor_overshoot():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_start_limit_tolerance_deg=1.5)
    source_config = SimpleNamespace(disable_torque_on_disconnect=True)
    positions = dict.fromkeys(MOTOR_NAMES, 0.0)
    positions["gripper"] = 1.257
    arm = _make_fault_hold_arm(positions)

    assert robot._secure_follower_current_position_hold("right", source_config, arm)

    for sent_call in arm.send_action.call_args_list:
        assert sent_call.args[0]["gripper.pos"] == 0.0
        assert sent_call.kwargs["apply_joint_limits"] is False


def test_fault_hold_rejects_large_physical_excess_before_any_mit_hold_or_enable():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_start_limit_tolerance_deg=1.5)
    source_config = SimpleNamespace(disable_torque_on_disconnect=False)
    positions = dict.fromkeys(MOTOR_NAMES, 0.0)
    positions["gripper"] = 1.501
    arm = _make_fault_hold_arm(positions)

    assert not robot._secure_follower_current_position_hold("right", source_config, arm)

    arm.send_action.assert_not_called()
    arm.bus.enable_torque.assert_not_called()
    arm.bus.disable_torque.assert_called_once_with(num_retry=2, require_response=True)
    assert source_config.disable_torque_on_disconnect is True


def test_secure_intervention_after_fault_holds_all_four_without_return_motion():
    robot = object.__new__(BiOpenArmFollower)
    left_source_config = SimpleNamespace(disable_torque_on_disconnect=True)
    right_source_config = SimpleNamespace(disable_torque_on_disconnect=True)
    robot.config = SimpleNamespace(
        left_arm_config=left_source_config,
        right_arm_config=right_source_config,
        deployment_start_limit_tolerance_deg=1.5,
    )
    positions = dict.fromkeys(MOTOR_NAMES, 0.0)

    def make_arm():
        return SimpleNamespace(
            config=SimpleNamespace(disable_torque_on_disconnect=True),
            bus=SimpleNamespace(
                is_connected=True,
                enable_torque=MagicMock(),
                disable_torque=MagicMock(),
            ),
            get_motor_positions=MagicMock(return_value=positions),
            trajectory_position_gains=MagicMock(return_value=({"joint_1": 10.0}, {"joint_1": 1.0})),
            send_action=MagicMock(return_value={f"{name}.pos": 0.0 for name in MOTOR_NAMES}),
        )

    robot.left_arm = make_arm()
    robot.right_arm = make_arm()
    teleop = MagicMock()
    teleop.secure_current_position_hold.return_value = True

    statuses = robot.secure_intervention_after_fault(teleop)

    assert statuses == {
        "follower_left": True,
        "follower_right": True,
        "leader_left": True,
        "leader_right": True,
    }
    teleop.secure_current_position_hold.assert_called_once_with()
    for source_config, arm in (
        (left_source_config, robot.left_arm),
        (right_source_config, robot.right_arm),
    ):
        assert source_config.disable_torque_on_disconnect is False
        assert arm.config.disable_torque_on_disconnect is False
        arm.get_motor_positions.assert_called_once_with(require_response=True)
        arm.bus.enable_torque.assert_called_once_with(num_retry=2, require_response=True)
        arm.bus.disable_torque.assert_not_called()


def test_secure_intervention_after_fault_disables_a_follower_that_cannot_hold():
    robot = object.__new__(BiOpenArmFollower)
    left_source_config = SimpleNamespace(disable_torque_on_disconnect=True)
    right_source_config = SimpleNamespace(disable_torque_on_disconnect=False)
    robot.config = SimpleNamespace(
        left_arm_config=left_source_config,
        right_arm_config=right_source_config,
        deployment_start_limit_tolerance_deg=1.5,
    )
    positions = dict.fromkeys(MOTOR_NAMES, 0.0)

    def make_arm():
        return SimpleNamespace(
            config=SimpleNamespace(disable_torque_on_disconnect=False),
            bus=SimpleNamespace(
                is_connected=True,
                enable_torque=MagicMock(),
                disable_torque=MagicMock(),
            ),
            get_motor_positions=MagicMock(return_value=positions),
            trajectory_position_gains=MagicMock(return_value=({"joint_1": 10.0}, {"joint_1": 1.0})),
            send_action=MagicMock(return_value={f"{name}.pos": 0.0 for name in MOTOR_NAMES}),
        )

    robot.left_arm = make_arm()
    robot.right_arm = make_arm()
    robot.right_arm.send_action.side_effect = RuntimeError("hold command failed")
    teleop = MagicMock()
    teleop.secure_current_position_hold.return_value = True

    statuses = robot.secure_intervention_after_fault(teleop)

    assert statuses["follower_left"]
    assert not statuses["follower_right"]
    assert statuses["leader_left"] and statuses["leader_right"]
    assert right_source_config.disable_torque_on_disconnect is True
    assert robot.right_arm.config.disable_torque_on_disconnect is True
    robot.right_arm.bus.disable_torque.assert_called_once_with(num_retry=2, require_response=True)


def test_secure_intervention_after_fault_continues_after_follower_keyboard_interrupt():
    robot = object.__new__(BiOpenArmFollower)
    left_source_config = SimpleNamespace(disable_torque_on_disconnect=False)
    right_source_config = SimpleNamespace(disable_torque_on_disconnect=True)
    robot.config = SimpleNamespace(
        left_arm_config=left_source_config,
        right_arm_config=right_source_config,
        deployment_start_limit_tolerance_deg=1.5,
    )
    positions = dict.fromkeys(MOTOR_NAMES, 0.0)
    robot.left_arm = _make_fault_hold_arm(positions)
    robot.right_arm = _make_fault_hold_arm(positions)
    robot.left_arm.get_motor_positions.side_effect = KeyboardInterrupt
    teleop = MagicMock()
    teleop.secure_current_position_hold.return_value = True

    statuses = robot.secure_intervention_after_fault(teleop)

    assert statuses == {
        "follower_left": False,
        "follower_right": True,
        "leader_left": True,
        "leader_right": True,
    }
    robot.left_arm.bus.disable_torque.assert_called_once_with(num_retry=2, require_response=True)
    robot.right_arm.get_motor_positions.assert_called_once_with(require_response=True)
    teleop.secure_current_position_hold.assert_called_once_with()
    assert left_source_config.disable_torque_on_disconnect is True


def test_secure_intervention_after_fault_disables_leaders_when_hold_is_not_established():
    robot = object.__new__(BiOpenArmFollower)
    robot._secure_follower_current_position_hold = MagicMock(return_value=True)
    robot.config = SimpleNamespace(
        left_arm_config=SimpleNamespace(),
        right_arm_config=SimpleNamespace(),
    )
    robot.left_arm = MagicMock()
    robot.right_arm = MagicMock()
    teleop = MagicMock()
    teleop.secure_current_position_hold.return_value = False

    statuses = robot.secure_intervention_after_fault(teleop)

    assert not statuses["leader_left"]
    assert not statuses["leader_right"]
    teleop.disable_torque.assert_called_once_with(require_response=True)


def _side_positions(value: float = 0.0) -> dict[str, tuple[float, ...]]:
    return {"left": (value,) * len(MOTOR_NAMES), "right": (value,) * len(MOTOR_NAMES)}


def _intervention_teleop(measured: dict[str, tuple[float, ...]]) -> MagicMock:
    teleop = MagicMock()
    teleop.name = "bi_openarm_leader"
    teleop.requires_continuous_feedback = True
    teleop.feedback_features = {
        f"{side}_{motor_name}.pos": float for side in ("left", "right") for motor_name in MOTOR_NAMES
    }
    teleop.get_action.return_value = {
        f"{side}_{motor_name}.pos": position
        for side in ("left", "right")
        for motor_name, position in zip(MOTOR_NAMES, measured[side], strict=True)
    }
    return teleop


def test_coordinated_target_send_routes_role_targets_and_checks_leader_tracking():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_tracking_error_deg=5.0)
    robot._send_deployment_targets = MagicMock()
    leader_targets = _side_positions(0.0)
    follower_targets = _side_positions(1.0)
    teleop = _intervention_teleop(leader_targets)

    robot._send_coordinated_deployment_targets(
        teleop,
        follower_targets,
        leader_targets,
        phase="startup-zero",
        reference_time_s=0.5,
    )

    robot._send_deployment_targets.assert_called_once_with(
        follower_targets,
        phase="startup-zero",
        reference_time_s=0.5,
    )
    teleop.send_feedback.assert_called_once_with(robot._as_bimanual_action(leader_targets))


def test_coordinated_target_send_rejects_leader_tracking_error():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_tracking_error_deg=5.0)
    robot._send_deployment_targets = MagicMock()
    teleop = _intervention_teleop(_side_positions(0.0))

    with pytest.raises(RuntimeError, match="tracking error: role=leader"):
        robot._send_coordinated_deployment_targets(
            teleop,
            _side_positions(10.0),
            _side_positions(10.0),
            phase="startup-replay",
            reference_time_s=1.0,
        )


def test_intervention_blend_uses_independent_starts_and_shared_final_target():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(
        deployment_control_frequency_hz=2.0,
        deployment_start_limit_tolerance_deg=1.0,
    )
    limits = dict.fromkeys(MOTOR_NAMES, (-180.0, 180.0))
    robot.left_arm = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    robot.right_arm = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    robot._send_coordinated_deployment_targets = MagicMock()
    teleop = MagicMock()

    with patch("lerobot.robots.bi_openarm_follower.bi_openarm_follower.precise_sleep"):
        robot._blend_intervention_positions(
            teleop,
            _side_positions(0.0),
            _side_positions(20.0),
            _side_positions(10.0),
            duration_s=1.0,
            phase="startup-zero",
        )

    first_call = robot._send_coordinated_deployment_targets.call_args_list[0]
    final_call = robot._send_coordinated_deployment_targets.call_args_list[-1]
    assert first_call.args[1]["left"] == pytest.approx((5.0,) * len(MOTOR_NAMES))
    assert first_call.args[2]["left"] == pytest.approx((15.0,) * len(MOTOR_NAMES))
    assert final_call.args[1] == _side_positions(10.0)
    assert final_call.args[2] == _side_positions(10.0)


def test_intervention_replay_sends_one_shared_target_to_both_roles():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_control_frequency_hz=100.0)
    robot._send_coordinated_deployment_targets = MagicMock()
    samples = [
        DeploymentTrajectorySample(0.0, (0.0,) * len(MOTOR_NAMES)),
        DeploymentTrajectorySample(0.01, (1.0,) * len(MOTOR_NAMES)),
    ]

    with patch("lerobot.robots.bi_openarm_follower.bi_openarm_follower.precise_sleep"):
        robot._replay_intervention_trajectories(
            MagicMock(),
            {"left": samples, "right": samples},
            speed_scale=1.0,
            phase="startup-replay",
        )

    assert robot._send_coordinated_deployment_targets.call_count == 2
    for call in robot._send_coordinated_deployment_targets.call_args_list:
        assert call.args[1] == call.args[2]


def test_intervention_disable_requires_leader_acknowledgements():
    robot = object.__new__(BiOpenArmFollower)
    robot._disable_both_arms = MagicMock()
    teleop = MagicMock()

    robot._disable_intervention_hardware(teleop)

    robot._disable_both_arms.assert_called_once_with()
    teleop.disable_torque.assert_called_once_with(require_response=True)


@pytest.mark.parametrize("interrupt_role", ["follower", "leader"])
def test_intervention_disable_attempts_all_four_arms_before_reraising_interrupt(interrupt_role):
    robot = object.__new__(BiOpenArmFollower)
    follower_arms = (MagicMock(name="left_follower"), MagicMock(name="right_follower"))
    robot.left_arm, robot.right_arm = follower_arms
    for arm in follower_arms:
        arm.bus.is_connected = True

    teleop = object.__new__(BiOpenArmLeader)
    leader_arms = (MagicMock(name="left_leader"), MagicMock(name="right_leader"))
    teleop.left_arm, teleop.right_arm = leader_arms
    for arm in leader_arms:
        arm.bus.is_connected = True

    interrupt = KeyboardInterrupt(f"{interrupt_role} disable interrupted")
    if interrupt_role == "follower":
        follower_arms[0].bus.disable_torque.side_effect = interrupt
    else:
        leader_arms[0].disable_torque.side_effect = interrupt

    with pytest.raises(KeyboardInterrupt, match=f"{interrupt_role} disable interrupted") as exc_info:
        robot._disable_intervention_hardware(teleop)

    assert exc_info.value is interrupt
    for arm in follower_arms:
        arm.bus.disable_torque.assert_called_once_with(num_retry=2, require_response=True)
    for arm in leader_arms:
        arm.disable_torque.assert_called_once_with(require_response=True)


@pytest.mark.parametrize("startup_error", [RuntimeError, KeyboardInterrupt])
def test_intervention_startup_requires_profile_and_disables_all_roles_on_error(startup_error):
    robot = object.__new__(BiOpenArmFollower)
    robot._deployment_trajectories = {}
    with pytest.raises(ValueError, match="deployment_trajectory_profile"):
        robot.prepare_for_intervention_deployment(MagicMock())

    final = DeploymentTrajectorySample(1.0, (0.0,) * len(MOTOR_NAMES))
    robot._deployment_trajectories = {"left": [final], "right": [final]}
    robot.config = SimpleNamespace(startup_zero_pose_duration_s=2.2)
    robot._validate_intervention_teleop = MagicMock()
    robot._read_deployment_positions = MagicMock(return_value=_side_positions())
    robot._read_intervention_positions = MagicMock(return_value=_side_positions())
    robot._blend_intervention_positions = MagicMock(side_effect=startup_error("tracking error"))
    robot._disable_intervention_hardware = MagicMock()
    teleop = MagicMock()

    with pytest.raises(startup_error, match="tracking error"):
        robot.prepare_for_intervention_deployment(teleop)

    robot._disable_intervention_hardware.assert_called_once_with(teleop)


def test_intervention_startup_replays_all_roles_and_hands_leader_to_follower_feedback():
    robot = object.__new__(BiOpenArmFollower)
    first = DeploymentTrajectorySample(0.0, (0.0,) * len(MOTOR_NAMES))
    final = DeploymentTrajectorySample(1.0, (2.0,) * len(MOTOR_NAMES))
    trajectories = {"left": [first, final], "right": [first, final]}
    robot._deployment_trajectories = trajectories
    robot.config = SimpleNamespace(
        startup_zero_pose_duration_s=2.2,
        startup_trajectory_blend_s=1.0,
        startup_trajectory_speed=0.5,
    )
    measured = _side_positions(0.5)
    robot._validate_intervention_teleop = MagicMock()
    robot._read_deployment_positions = MagicMock(return_value=measured)
    robot._read_intervention_positions = MagicMock(return_value=measured)
    robot._blend_intervention_positions = MagicMock()
    robot._replay_intervention_trajectories = MagicMock()
    robot.send_action = MagicMock()
    robot._check_deployment_tracking = MagicMock()
    robot._disable_intervention_hardware = MagicMock()
    teleop = MagicMock()

    assert robot.prepare_for_intervention_deployment(teleop) is True

    zero = _side_positions(0.0)
    first_targets = {"left": first.positions_deg, "right": first.positions_deg}
    assert robot._blend_intervention_positions.call_args_list[0].args[4:] == (
        2.2,
        "startup-zero",
    )
    assert robot._blend_intervention_positions.call_args_list[0].args[3] == zero
    assert robot._blend_intervention_positions.call_args_list[1].args[3] == first_targets
    robot._replay_intervention_trajectories.assert_called_once_with(
        teleop,
        trajectories,
        0.5,
        "startup-replay",
    )
    expected_final_action = {
        f"{side}_{motor_name}.pos": 2.0 for side in ("left", "right") for motor_name in MOTOR_NAMES
    }
    robot.send_action.assert_called_once_with(expected_final_action)
    teleop.send_feedback.assert_called_once_with(robot._as_bimanual_action(measured))
    robot._check_deployment_tracking.assert_called_once_with(
        measured,
        measured,
        role="leader",
        phase="startup-handover",
        reference_time_s=0.0,
    )
    robot._disable_intervention_hardware.assert_not_called()


@pytest.mark.parametrize("shutdown_error", [RuntimeError, KeyboardInterrupt])
def test_intervention_shutdown_error_holds_all_four_devices(shutdown_error):
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(
        shutdown_task_pose_warn_deg=180.0,
        shutdown_task_pose_blend_s=10.0,
        hold_position_on_shutdown_error=True,
    )
    final = DeploymentTrajectorySample(1.0, (0.0,) * len(MOTOR_NAMES))
    robot._deployment_trajectories = {"left": [final], "right": [final]}
    robot._validate_intervention_teleop = MagicMock()
    robot._read_deployment_positions = MagicMock(return_value=_side_positions())
    robot._read_intervention_positions = MagicMock(return_value=_side_positions())
    robot._blend_intervention_positions = MagicMock(side_effect=shutdown_error("tracking error"))
    robot._hold_intervention_hardware_after_shutdown_error = MagicMock()
    robot._disable_intervention_hardware = MagicMock()
    teleop = MagicMock()

    with pytest.raises(shutdown_error, match="tracking error"):
        robot.finish_intervention_deployment(teleop)

    robot._hold_intervention_hardware_after_shutdown_error.assert_called_once_with(teleop)
    robot._disable_intervention_hardware.assert_not_called()


def test_intervention_shutdown_replays_reversed_csv_for_all_roles():
    robot = object.__new__(BiOpenArmFollower)
    first = DeploymentTrajectorySample(0.0, (0.0,) * len(MOTOR_NAMES))
    final = DeploymentTrajectorySample(1.0, (2.0,) * len(MOTOR_NAMES))
    robot._deployment_trajectories = {"left": [first, final], "right": [first, final]}
    robot.config = SimpleNamespace(
        shutdown_task_pose_warn_deg=180.0,
        shutdown_task_pose_blend_s=10.0,
        shutdown_zero_transition_s=1.0,
        shutdown_replay_speed=0.25,
        hold_position_on_shutdown_error=True,
    )
    follower_measured = _side_positions(0.0)
    leader_measured = _side_positions(1.0)
    robot._validate_intervention_teleop = MagicMock()
    robot._read_deployment_positions = MagicMock(return_value=follower_measured)
    robot._read_intervention_positions = MagicMock(return_value=leader_measured)
    robot._blend_intervention_positions = MagicMock()
    robot._replay_intervention_trajectories = MagicMock()
    robot._hold_intervention_hardware_after_shutdown_error = MagicMock()
    robot._disable_intervention_hardware = MagicMock()
    teleop = MagicMock()

    assert robot.finish_intervention_deployment(teleop) is True

    task_targets = _side_positions(2.0)
    robot._blend_intervention_positions.assert_called_once_with(
        teleop,
        follower_measured,
        leader_measured,
        task_targets,
        10.0,
        "shutdown-task-pose",
    )
    replay_call = robot._replay_intervention_trajectories.call_args
    assert replay_call.args[0] is teleop
    assert replay_call.args[2:] == (0.25, "shutdown-replay")
    for samples in replay_call.args[1].values():
        assert samples[0].positions_deg == final.positions_deg
        assert samples[-1].positions_deg == (0.0,) * len(MOTOR_NAMES)
    robot._hold_intervention_hardware_after_shutdown_error.assert_not_called()
    robot._disable_intervention_hardware.assert_not_called()
