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
    robot.config = SimpleNamespace(deployment_start_limit_tolerance_deg=1.0)
    limits = dict.fromkeys(MOTOR_NAMES, (-45.0, 45.0))
    robot.left_arm = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    robot.right_arm = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    start = {
        "left": (0.0,) * 5 + (-45.25,) + (0.0,) * 2,
        "right": (0.0,) * len(MOTOR_NAMES),
    }

    clamped = robot._clamp_measured_blend_start(start, "shutdown-task-pose")

    assert clamped["left"][5] == -45.0
    assert "Clamping measured OpenArm blend start" in caplog.text


def test_large_measured_blend_start_overshoot_is_rejected():
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_start_limit_tolerance_deg=1.0)
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
    )
    positions = dict.fromkeys(MOTOR_NAMES, 0.0)

    def make_arm():
        arm_config = SimpleNamespace(disable_torque_on_disconnect=True)
        return SimpleNamespace(
            config=arm_config,
            bus=SimpleNamespace(is_connected=True),
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
        arm.send_action.assert_called_once_with(
            {f"{name}.pos": 0.0 for name in MOTOR_NAMES},
            custom_kp={"joint_1": 10.0},
            custom_kd={"joint_1": 1.0},
            apply_max_relative_target=False,
        )


def _side_positions(value: float = 0.0) -> dict[str, tuple[float, ...]]:
    return {"left": (value,) * len(MOTOR_NAMES), "right": (value,) * len(MOTOR_NAMES)}


def _intervention_teleop(measured: dict[str, tuple[float, ...]]) -> MagicMock:
    teleop = MagicMock()
    teleop.name = "bi_openarm_leader"
    teleop.requires_continuous_feedback = True
    teleop.feedback_features = {
        f"{side}_{motor_name}.pos": float
        for side in ("left", "right")
        for motor_name in MOTOR_NAMES
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

    with patch(
        "lerobot.robots.bi_openarm_follower.bi_openarm_follower.precise_sleep"
    ):
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

    with patch(
        "lerobot.robots.bi_openarm_follower.bi_openarm_follower.precise_sleep"
    ):
        robot._replay_intervention_trajectories(
            MagicMock(),
            {"left": samples, "right": samples},
            speed_scale=1.0,
            phase="startup-replay",
        )

    assert robot._send_coordinated_deployment_targets.call_count == 2
    for call in robot._send_coordinated_deployment_targets.call_args_list:
        assert call.args[1] == call.args[2]


def test_intervention_startup_requires_profile_and_disables_all_roles_on_error():
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
    robot._blend_intervention_positions = MagicMock(side_effect=RuntimeError("tracking error"))
    robot._disable_intervention_hardware = MagicMock()
    teleop = MagicMock()

    with pytest.raises(RuntimeError, match="tracking error"):
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
        f"{side}_{motor_name}.pos": 2.0
        for side in ("left", "right")
        for motor_name in MOTOR_NAMES
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


def test_intervention_shutdown_error_holds_all_four_devices():
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
    robot._blend_intervention_positions = MagicMock(side_effect=RuntimeError("tracking error"))
    robot._hold_intervention_hardware_after_shutdown_error = MagicMock()
    robot._disable_intervention_hardware = MagicMock()
    teleop = MagicMock()

    with pytest.raises(RuntimeError, match="tracking error"):
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
