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
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..openarm_follower import OpenArmFollowerConfigBase


@RobotConfig.register_subclass("bi_openarm_follower")
@dataclass(kw_only=True)
class BiOpenArmFollowerConfig(RobotConfig):
    """Configuration class for Bi OpenArm Follower robots."""

    id: str | None = "bi_openarm_follower"

    left_arm_config: OpenArmFollowerConfigBase
    right_arm_config: OpenArmFollowerConfigBase

    # Top-level cameras not attached to a specific side. Keys are kept as-is in
    # observations (no `left_`/`right_` prefix). Per-arm cameras (declared on
    # `{left,right}_arm_config.cameras`) are prefixed.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Optional task-ready CSV profile used only by lerobot-rollout's policy
    # deployment lifecycle. Base rollout moves both followers; OpenArm DAgger
    # replays the same targets on both leaders and followers. The profile
    # directory must contain left_arm.csv and right_arm.csv in
    # openarm_v1_motor_zero radians.
    deployment_trajectory_profile: str | None = None
    deployment_control_frequency_hz: float = 50.0
    startup_zero_pose_duration_s: float = 2.2
    startup_trajectory_speed: float = 1.0
    startup_trajectory_blend_s: float = 1.0
    shutdown_task_pose_blend_s: float = 10.0
    shutdown_replay_speed: float = 0.25
    # This duration is part of the recorded-time return trajectory, so its
    # wall-clock duration is divided by shutdown_replay_speed.
    shutdown_zero_transition_s: float = 1.0
    shutdown_task_pose_warn_deg: float = math.degrees(0.5)
    deployment_tracking_error_deg: float = math.degrees(0.35)
    # Covers the observed OpenArm gripper sensor overshoot (+1.257 deg) while
    # still treating larger deviations from a physical limit as a fault.
    deployment_start_limit_tolerance_deg: float = 1.5
    hold_position_on_shutdown_error: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.deployment_trajectory_profile is None:
            return

        bounded_values = {
            "deployment_control_frequency_hz": (self.deployment_control_frequency_hz, 10.0, 200.0),
            "startup_zero_pose_duration_s": (self.startup_zero_pose_duration_s, 0.1, 60.0),
            "startup_trajectory_speed": (self.startup_trajectory_speed, 0.1, 2.0),
            "startup_trajectory_blend_s": (self.startup_trajectory_blend_s, 0.0, 10.0),
            "shutdown_task_pose_blend_s": (self.shutdown_task_pose_blend_s, 1.0, 60.0),
            "shutdown_replay_speed": (self.shutdown_replay_speed, 0.1, 1.0),
            "shutdown_zero_transition_s": (self.shutdown_zero_transition_s, 0.1, 10.0),
            "shutdown_task_pose_warn_deg": (self.shutdown_task_pose_warn_deg, math.degrees(0.1), 180.0),
            "deployment_tracking_error_deg": (self.deployment_tracking_error_deg, 1.0, 90.0),
            "deployment_start_limit_tolerance_deg": (
                self.deployment_start_limit_tolerance_deg,
                0.0,
                10.0,
            ),
        }
        for name, (value, minimum, maximum) in bounded_values.items():
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
