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

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from lerobot.robots.bi_openarm_follower.deployment_trajectory import MOTOR_NAMES
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.scripts import lerobot_openarm_home_motion_test as home_motion_script
from lerobot.scripts.lerobot_openarm_home_motion_test import OpenArmHomeMotionTestConfig


def _make_config(*, confirm_before_motion: bool = False, home_hold_s: float = 1.5):
    return OpenArmHomeMotionTestConfig(
        robot=BiOpenArmFollowerConfig(
            left_arm_config=OpenArmFollowerConfigBase(port="can3", side="left"),
            right_arm_config=OpenArmFollowerConfigBase(port="can2", side="right"),
            deployment_trajectory_profile="/profile",
        ),
        confirm_before_motion=confirm_before_motion,
        home_hold_s=home_hold_s,
    )


def _make_robot(*, final_position_deg: float = 0.0):
    robot = object.__new__(BiOpenArmFollower)
    robot.config = SimpleNamespace(deployment_tracking_error_deg=20.0)
    robot._deployment_trajectories = {"left": [], "right": []}
    robot.connect = MagicMock()
    robot.prepare_for_policy_deployment = MagicMock(return_value=True)
    robot.finish_policy_deployment = MagicMock(return_value=True)

    positions = dict.fromkeys(MOTOR_NAMES, final_position_deg)
    robot.left_arm = MagicMock(is_connected=True)
    robot.right_arm = MagicMock(is_connected=True)
    robot.left_arm.get_motor_positions.return_value = positions
    robot.right_arm.get_motor_positions.return_value = positions
    return robot


def test_runs_only_startup_hold_shutdown_and_disconnects(monkeypatch):
    cfg = _make_config()
    robot = _make_robot()
    sleep = MagicMock()
    monkeypatch.setattr(home_motion_script, "make_robot_from_config", MagicMock(return_value=robot))
    monkeypatch.setattr(home_motion_script, "precise_sleep", sleep)

    assert home_motion_script.run_openarm_home_motion_test(cfg) is True

    robot.connect.assert_called_once_with()
    robot.prepare_for_policy_deployment.assert_called_once_with()
    sleep.assert_called_once_with(1.5)
    robot.finish_policy_deployment.assert_called_once_with()
    robot.left_arm.disconnect.assert_called_once_with()
    robot.right_arm.disconnect.assert_called_once_with()


def test_cancellation_happens_before_connect(monkeypatch):
    cfg = _make_config(confirm_before_motion=True)
    robot = _make_robot()
    monkeypatch.setattr(home_motion_script, "make_robot_from_config", MagicMock(return_value=robot))
    monkeypatch.setattr(home_motion_script, "_confirm_motion", MagicMock(return_value=False))

    assert home_motion_script.run_openarm_home_motion_test(cfg) is False
    robot.connect.assert_not_called()
    robot.prepare_for_policy_deployment.assert_not_called()


def test_motion_error_disconnects_both_arms(monkeypatch):
    cfg = _make_config()
    robot = _make_robot()
    robot.prepare_for_policy_deployment.side_effect = RuntimeError("tracking error")
    monkeypatch.setattr(home_motion_script, "make_robot_from_config", MagicMock(return_value=robot))

    with pytest.raises(RuntimeError, match="tracking error"):
        home_motion_script.run_openarm_home_motion_test(cfg)

    robot.finish_policy_deployment.assert_not_called()
    robot.left_arm.disconnect.assert_called_once_with()
    robot.right_arm.disconnect.assert_called_once_with()


def test_fails_if_shutdown_did_not_reach_motor_zero(monkeypatch):
    cfg = _make_config(home_hold_s=0.0)
    robot = _make_robot(final_position_deg=21.0)
    monkeypatch.setattr(home_motion_script, "make_robot_from_config", MagicMock(return_value=robot))

    with pytest.raises(RuntimeError, match="did not finish near motor zero"):
        home_motion_script.run_openarm_home_motion_test(cfg)

    robot.left_arm.disconnect.assert_called_once_with()
    robot.right_arm.disconnect.assert_called_once_with()


@pytest.mark.parametrize("home_hold_s", [-0.1, math.inf])
def test_rejects_invalid_home_hold_duration(home_hold_s):
    with pytest.raises(ValueError, match="home_hold_s must be between"):
        _make_config(home_hold_s=home_hold_s)
