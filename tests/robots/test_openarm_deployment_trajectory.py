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
from unittest.mock import MagicMock

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
