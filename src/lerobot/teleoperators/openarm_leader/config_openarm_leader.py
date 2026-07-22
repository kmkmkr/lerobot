#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import os
from dataclasses import dataclass, field

from lerobot.robots.openarm_follower.config_openarm_follower import (
    BILATERAL_FRICTION_TANH_COEFFICIENT,
    BILATERAL_GRAVITY_M_S2,
    OPENARM_V1_COORDINATE_FRAME,
)

from ..config import TeleoperatorConfig

# Canonical defaults mirrored from openarm_teleop/config/leader.yaml and the
# dora-openarm-data-collection launcher's default J7_TUNING_PROFILE=validated.
# Keep these values synchronized with the native bilateral leader.
BILATERAL_LEADER_KP = (192.0, 192.0, 192.0, 96.0, 19.2, 24.8, 20.0, 16.0)
BILATERAL_LEADER_KD = (3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2)
BILATERAL_LEADER_FC = (0.306, 0.306, 0.40, 0.166, 0.050, 0.083, 0.172, 0.0512)
BILATERAL_LEADER_FRICTION_K = (28.417, 28.417, 29.065, 130.038, 151.771, 242.287, 7.888, 4.0)
BILATERAL_LEADER_FV = (0.063, 0.063, 0.604, 0.813, 0.029, 0.072, 0.084, 0.084)
BILATERAL_LEADER_FO = (0.088, 0.088, 0.008, -0.058, 0.005, 0.009, -0.059, -0.050)
BILATERAL_LEADER_GRAVITY_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95)
BILATERAL_LEADER_FC_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 1.0)
BILATERAL_LEADER_FV_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 1.0)
BILATERAL_LEADER_FO_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 1.0)


def _compensation_enabled_by_default() -> bool:
    return os.environ.get("LEROBOT_OPENARM_ENABLE_COMPENSATION", "1") != "0"


@dataclass
class OpenArmLeaderConfigBase:
    """Base configuration for the OpenArms leader/teleoperator with Damiao motors."""

    # CAN interfaces - one per arm
    # Arm CAN interface (e.g., "can3")
    # Linux: "can0", "can1", etc.
    port: str

    # Required for the side-specific v10 dynamics chain when bilateral gravity
    # compensation is active. BiOpenArmLeader assigns this automatically.
    side: str | None = None

    # OpenArm v1 positions are raw motor angles relative to the official motor zero.
    coordinate_frame: str = OPENARM_V1_COORDINATE_FRAME

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # CAN FD settings (OpenArms uses CAN FD by default)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

    # Motor configuration for OpenArms (7 DOF per arm)
    # Maps motor names to (send_can_id, recv_can_id, motor_type)
    # Based on: https://docs.openarm.dev/software/setup/configure-test
    # OpenArms uses 4 types of motors:
    # - DM8009 (DM-J8009P-2EC) for shoulders (high torque)
    # - DM4340P and DM4340 for shoulder rotation and elbow
    # - DM4310 (DM-J4310-2EC V1.1) for wrist and gripper
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm8009"),  # J1 - Shoulder pan (DM8009)
            "joint_2": (0x02, 0x12, "dm8009"),  # J2 - Shoulder lift (DM8009)
            "joint_3": (0x03, 0x13, "dm4340"),  # J3 - Shoulder rotation (DM4340)
            "joint_4": (0x04, 0x14, "dm4340"),  # J4 - Elbow flex (DM4340)
            "joint_5": (0x05, 0x15, "dm4310"),  # J5 - Wrist roll (DM4310)
            "joint_6": (0x06, 0x16, "dm4310"),  # J6 - Wrist pitch (DM4310)
            "joint_7": (0x07, 0x17, "dm4310"),  # J7 - Wrist rotation (DM4310)
            "gripper": (0x08, 0x18, "dm4310"),  # J8 - Gripper (DM4310)
        }
    )

    # The default is the native-style bilateral mode: follower observations are
    # sent back as leader MIT position references. Set this True only for the
    # legacy free-motion diagnostic mode, which disables leader torque and
    # feed-forward compensation.
    manual_control: bool = False

    # Whether disconnect should disable leader torque. Keep enabled for both
    # bilateral and free-motion operation.
    disable_torque_on_disconnect: bool = True

    # When True, expose `.vel` and `.torque` per motor in action features.
    # Default False for compatibility with the position-only openarm_mini teleoperator.
    use_velocity_and_torque: bool = False

    # MIT control parameters used for follower-to-leader bilateral feedback.
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_KP))
    position_kd: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_KD))

    # Native bilateral leader feed-forward compensation. Gravity is computed
    # from the same v10 bimanual URDF; friction uses motor-zero radians/second.
    # J7 scales match Dora's validated profile.
    gravity_compensation: bool = field(default_factory=_compensation_enabled_by_default)
    friction_compensation: bool = field(default_factory=_compensation_enabled_by_default)
    dynamics_urdf_path: str = field(
        default_factory=lambda: os.environ.get(
            "LEROBOT_OPENARM_DYNAMICS_URDF", "/workspace/openarm_v10_bimanual.urdf"
        )
    )
    compensation_state_max_age_s: float = 0.1
    gravity_m_s2: float = BILATERAL_GRAVITY_M_S2
    gravity_scale: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_GRAVITY_SCALE))
    friction_tanh_coefficient: float = BILATERAL_FRICTION_TANH_COEFFICIENT
    friction_fc: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_FC))
    friction_k: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_FRICTION_K))
    friction_fv: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_FV))
    friction_fo: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_FO))
    friction_fc_scale: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_FC_SCALE))
    friction_fv_scale: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_FV_SCALE))
    friction_fo_scale: list[float] = field(default_factory=lambda: list(BILATERAL_LEADER_FO_SCALE))


@TeleoperatorConfig.register_subclass("openarm_leader")
@dataclass
class OpenArmLeaderConfig(TeleoperatorConfig, OpenArmLeaderConfigBase):
    pass
