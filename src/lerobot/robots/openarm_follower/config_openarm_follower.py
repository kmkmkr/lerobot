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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

OPENARM_V1_HARDWARE_VERSION = "1.1"
OPENARM_V1_DESCRIPTION_PROFILE = "v10"
OPENARM_V1_DESCRIPTION_REF = "1.0.4"
OPENARM_V1_COORDINATE_FRAME = "openarm_v1_motor_zero"
OPENARM_V1_CALIBRATION_METHOD = "openarm_can_zero_position"

# Canonical defaults mirrored from openarm_teleop/config/follower.yaml and the
# dora-openarm-data-collection launcher's default J7_TUNING_PROFILE=validated.
# Keep these values synchronized with the native bilateral follower.
BILATERAL_FOLLOWER_KP = (240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0)
BILATERAL_FOLLOWER_KD = (3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2)
BILATERAL_FOLLOWER_FC = (0.306, 0.306, 0.40, 0.166, 0.050, 0.093, 0.172, 0.0512)
BILATERAL_FOLLOWER_FRICTION_K = (28.417, 28.417, 29.065, 130.038, 151.771, 242.287, 7.888, 4.0)
BILATERAL_FOLLOWER_FV = (0.063, 0.063, 0.604, 0.813, 0.029, 0.072, 0.084, 0.084)
BILATERAL_FOLLOWER_FO = (0.088, 0.088, 0.008, -0.058, 0.005, 0.009, -0.059, -0.050)
BILATERAL_FOLLOWER_GRAVITY_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95)
BILATERAL_FOLLOWER_FC_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 1.0)
BILATERAL_FOLLOWER_FV_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 1.0)
BILATERAL_FOLLOWER_FO_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 1.0)
BILATERAL_GRAVITY_M_S2 = 9.81
BILATERAL_FRICTION_TANH_COEFFICIENT = 0.1


def _compensation_enabled_by_default() -> bool:
    return os.environ.get("LEROBOT_OPENARM_ENABLE_COMPENSATION", "1") != "0"


# Limits in the OpenArm v1 v10 description, expressed in motor-zero degrees.
# Source: openarm_description 1.0.4, config/arm/v10/joint_limits.yaml.
OPENARM_V1_PHYSICAL_JOINT_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "left": {
        "joint_1": (-200.0, 80.0),
        "joint_2": (-190.0, 10.0),
        "joint_3": (-90.0, 90.0),
        "joint_4": (0.0, 140.0),
        "joint_5": (-90.0, 90.0),
        "joint_6": (-45.0, 45.0),
        "joint_7": (-90.0, 90.0),
        "gripper": (-60.0, 0.0),
    },
    "right": {
        "joint_1": (-80.0, 200.0),
        "joint_2": (-10.0, 190.0),
        "joint_3": (-90.0, 90.0),
        "joint_4": (0.0, 140.0),
        "joint_5": (-90.0, 90.0),
        "joint_6": (-45.0, 45.0),
        "joint_7": (-90.0, 90.0),
        "gripper": (-60.0, 0.0),
    },
}

# Deployment limits. J1 and the outward J2 direction keep task-space
# restrictions, while J3-J7 use the OpenArm v1 physical range. They are also
# consumed by the OpenArmDataset converter so collection and deployment are
# audited against the same coordinate contract.
LEFT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-90.0, 10.0),
    "joint_3": (-90.0, 90.0),
    "joint_4": (0.0, 140.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-45.0, 45.0),
    "joint_7": (-90.0, 90.0),
    "gripper": (-60.0, 0.0),
}

RIGHT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-10.0, 90.0),
    "joint_3": (-90.0, 90.0),
    "joint_4": (0.0, 140.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-45.0, 45.0),
    "joint_7": (-90.0, 90.0),
    "gripper": (-60.0, 0.0),
}

OPENARM_V1_SAFE_JOINT_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "left": LEFT_DEFAULT_JOINTS_LIMITS,
    "right": RIGHT_DEFAULT_JOINTS_LIMITS,
}

OPENARM_FALLBACK_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-5.0, 5.0),
    "joint_2": (-5.0, 5.0),
    "joint_3": (-5.0, 5.0),
    "joint_4": (0.0, 5.0),
    "joint_5": (-5.0, 5.0),
    "joint_6": (-5.0, 5.0),
    "joint_7": (-5.0, 5.0),
    "gripper": (-5.0, 0.0),
}


@dataclass
class OpenArmFollowerConfigBase:
    """Base configuration for the OpenArms follower robot with Damiao motors."""

    # CAN interfaces - one per arm
    # arm CAN interface (e.g., "can1")
    # Linux: "can0", "can1", etc.
    port: str

    # side of the arm: "left" or "right". If "None" default values will be used
    side: str | None = None

    # OpenArm v1 positions are raw motor angles relative to the zero written by
    # the official openarm-can zero-position calibration tool.
    coordinate_frame: str = OPENARM_V1_COORDINATE_FRAME

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # CAN FD settings (OpenArms uses CAN FD by default)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

    # Whether to disable torque when disconnecting
    disable_torque_on_disconnect: bool = True

    # When True, expose `.vel` and `.torque` per motor in observation features.
    # Default False for compatibility with the position-only openarm_mini teleoperator.
    use_velocity_and_torque: bool = False

    # Safety limit for relative target positions
    # Set to a positive scalar for all motors, or a dict mapping motor names to limits
    max_relative_target: float | dict[str, float] | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

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

    # MIT control parameters for position control (used in send_action)
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_KP))
    position_kd: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_KD))

    # CSV home-motion gains remain independently configurable, but both sets of
    # defaults match the native bilateral follower. Compensation is shared too.
    trajectory_position_kp: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_KP))
    trajectory_position_kd: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_KD))

    # Native bilateral follower feed-forward compensation. Gravity is computed
    # from the same v10 bimanual URDF, while friction uses the current motor
    # velocity in radians per second. J7 scales match Dora's validated profile.
    gravity_compensation: bool = field(default_factory=_compensation_enabled_by_default)
    friction_compensation: bool = field(default_factory=_compensation_enabled_by_default)
    dynamics_urdf_path: str = field(
        default_factory=lambda: os.environ.get(
            "LEROBOT_OPENARM_DYNAMICS_URDF", "/workspace/openarm_v10_bimanual.urdf"
        )
    )
    compensation_state_max_age_s: float = 0.1
    gravity_m_s2: float = BILATERAL_GRAVITY_M_S2
    gravity_scale: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_GRAVITY_SCALE))
    friction_tanh_coefficient: float = BILATERAL_FRICTION_TANH_COEFFICIENT
    friction_fc: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_FC))
    friction_k: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_FRICTION_K))
    friction_fv: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_FV))
    friction_fo: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_FO))
    friction_fc_scale: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_FC_SCALE))
    friction_fv_scale: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_FV_SCALE))
    friction_fo_scale: list[float] = field(default_factory=lambda: list(BILATERAL_FOLLOWER_FO_SCALE))

    # Values for joint limits. When omitted, side-specific conservative v1
    # limits are selected. Explicit values are preserved and validated instead
    # of being silently overwritten when ``side`` is set.
    joint_limits: dict[str, tuple[float, float]] | None = None


@RobotConfig.register_subclass("openarm_follower")
@dataclass
class OpenArmFollowerConfig(RobotConfig, OpenArmFollowerConfigBase):
    pass
