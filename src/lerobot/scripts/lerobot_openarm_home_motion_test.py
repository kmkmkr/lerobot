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

"""Exercise only the OpenArm rollout startup and shutdown trajectory motions."""

import logging
import math
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.robots import RobotConfig, bi_openarm_follower, make_robot_from_config  # noqa: F401
from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from lerobot.robots.bi_openarm_follower.confirmation import confirm_openarm_motion
from lerobot.robots.bi_openarm_follower.deployment_trajectory import MOTOR_NAMES
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class OpenArmHomeMotionTestConfig:
    """Configuration for one zero-to-home-to-zero hardware test cycle."""

    robot: RobotConfig
    home_hold_s: float = 3.0
    confirm_before_motion: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.robot, BiOpenArmFollowerConfig):
            raise ValueError("robot.type must be bi_openarm_follower")
        if self.robot.deployment_trajectory_profile is None:
            raise ValueError("robot.deployment_trajectory_profile is required")
        if not math.isfinite(self.home_hold_s) or not 0.0 <= self.home_hold_s <= 300.0:
            raise ValueError(f"home_hold_s must be between 0 and 300 seconds, got {self.home_hold_s}")

        configured_cameras = {
            **self.robot.cameras,
            **self.robot.left_arm_config.cameras,
            **self.robot.right_arm_config.cameras,
        }
        if configured_cameras:
            raise ValueError(
                "This motion-only test does not initialize cameras; remove all robot camera settings"
            )
        if not (
            self.robot.left_arm_config.disable_torque_on_disconnect
            and self.robot.right_arm_config.disable_torque_on_disconnect
        ):
            raise ValueError("This hardware test requires disable_torque_on_disconnect=true for both arms")


def _confirm_motion() -> bool:
    return confirm_openarm_motion(
        "Run the OpenArm motor-zero -> task-ready -> motor-zero motion test? [yes/no]: ",
        "Motion test was not confirmed; cancelling before connecting hardware.",
    )


def _disconnect_connected_arms(robot: BiOpenArmFollower) -> None:
    """Disconnect each connected arm independently, including after a partial connect."""
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        if not arm.is_connected:
            continue
        try:
            arm.disconnect()
        except Exception:
            logger.exception("Failed to disconnect the %s OpenArm follower", side)


def _verify_motor_zero(robot: BiOpenArmFollower) -> None:
    tolerance_deg = robot.config.deployment_tracking_error_deg
    violations: list[str] = []
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        positions = arm.get_motor_positions()
        for motor_name in MOTOR_NAMES:
            position = float(positions[motor_name])
            if not math.isfinite(position) or abs(position) > tolerance_deg:
                violations.append(f"side={side} motor={motor_name} position={position:.3f} deg")

    if violations:
        details = "; ".join(violations)
        raise RuntimeError(
            "OpenArm home-motion test did not finish near motor zero "
            f"(tolerance={tolerance_deg:.3f} deg): {details}"
        )


@parser.wrap()
def run_openarm_home_motion_test(cfg: OpenArmHomeMotionTestConfig) -> bool:
    """Connect, run one rollout home-motion round trip, and disconnect."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    if not isinstance(robot, BiOpenArmFollower):
        raise TypeError(f"Expected BiOpenArmFollower, got {type(robot).__name__}")
    if not robot.has_policy_deployment_trajectory:
        raise RuntimeError("The OpenArm deployment trajectory profile did not load")

    if cfg.confirm_before_motion and not _confirm_motion():
        logger.info("OpenArm home-motion test cancelled before connecting hardware.")
        return False

    try:
        logger.info("Connecting both OpenArm followers without cameras or policy inference...")
        robot.connect()

        logger.info("Phase 1/3: moving through motor zero and replaying the CSV to task-ready pose.")
        if not robot.prepare_for_policy_deployment():
            raise RuntimeError("OpenArm startup trajectory was not executed")

        logger.info("Phase 2/3: holding the task-ready pose for %.1f seconds.", cfg.home_hold_s)
        if cfg.home_hold_s > 0.0:
            precise_sleep(cfg.home_hold_s)

        logger.info("Phase 3/3: replaying the CSV in reverse and returning to motor zero.")
        if not robot.finish_policy_deployment():
            raise RuntimeError("OpenArm shutdown trajectory was not executed")
        _verify_motor_zero(robot)
    finally:
        _disconnect_connected_arms(robot)

    logger.info("OpenArm motor-zero -> task-ready -> motor-zero motion test completed successfully.")
    return True


def main() -> None:
    register_third_party_plugins()
    run_openarm_home_motion_test()


if __name__ == "__main__":
    main()
