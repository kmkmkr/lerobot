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

import logging
import math
import time
from typing import Any

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MOTOR_LIMIT_PARAMS, MotorType
from lerobot.robots.openarm_follower import (
    OPENARM_V1_COORDINATE_FRAME,
    OPENARM_V1_PHYSICAL_JOINT_LIMITS,
)
from lerobot.robots.openarm_follower.openarm_dynamics import (
    OpenArmGravityModel,
    bilateral_friction_torque,
)
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_leader import OpenArmLeaderConfig

logger = logging.getLogger(__name__)

OPENARM_MOTOR_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
)


def _validate_position_gains(name: str, values: list[float], motor_count: int, maximum: float) -> None:
    if len(values) != motor_count:
        raise ValueError(f"{name} must contain {motor_count} values, got {len(values)}")
    for index, value in enumerate(values):
        if not math.isfinite(value) or not 0.0 <= value <= maximum:
            raise ValueError(f"{name}[{index}] must be between 0 and {maximum}, got {value}")


def _validate_finite_values(name: str, values: list[float], expected_count: int) -> None:
    if len(values) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} values, got {len(values)}")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{name}[{index}] must be finite, got {value}")


class OpenArmLeader(Teleoperator):
    """
    OpenArm Leader/Teleoperator Arm with Damiao motors.

    By default this teleoperator applies native-style bilateral feedback: the
    follower observation is used as the leader position reference, while the
    leader's own gravity and friction feed-forward terms preserve manual feel.
    """

    config_class = OpenArmLeaderConfig
    name = "openarm_leader"

    def __init__(self, config: OpenArmLeaderConfig):
        super().__init__(config)
        self.config = config

        if config.coordinate_frame != OPENARM_V1_COORDINATE_FRAME:
            raise ValueError(
                f"Unsupported OpenArm coordinate frame {config.coordinate_frame!r}; "
                f"expected {OPENARM_V1_COORDINATE_FRAME!r}"
            )
        if config.side not in (None, "left", "right"):
            raise ValueError("config.side must be either 'left', 'right', or None")
        if not config.manual_control and config.side is None:
            raise ValueError("bilateral OpenArm leader control requires config.side to be 'left' or 'right'")
        if tuple(config.motor_config) != OPENARM_MOTOR_NAMES:
            raise ValueError(
                "OpenArm leader motor_config must use the canonical ordered motors "
                f"{OPENARM_MOTOR_NAMES}, got {tuple(config.motor_config)}"
            )

        # Arm motors
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Always use degrees for Damiao motors
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        if self.calibration:
            logger.warning(
                "Ignoring legacy LeRobot calibration file %s. OpenArm v1 uses the motor zero written "
                "by the official openarm-can calibration tool.",
                self.calibration_fpath,
            )

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration={},
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        motor_count = len(config.motor_config)
        _validate_position_gains("position_kp", config.position_kp, motor_count, 500.0)
        _validate_position_gains("position_kd", config.position_kd, motor_count, 5.0)
        _validate_finite_values("gravity_scale", config.gravity_scale, 7)
        for name in (
            "friction_fc",
            "friction_k",
            "friction_fv",
            "friction_fo",
            "friction_fc_scale",
            "friction_fv_scale",
            "friction_fo_scale",
        ):
            _validate_finite_values(name, getattr(config, name), motor_count)
        if (
            not math.isfinite(config.compensation_state_max_age_s)
            or config.compensation_state_max_age_s < 0.0
        ):
            raise ValueError("compensation_state_max_age_s must be finite and non-negative")
        if not math.isfinite(config.gravity_m_s2) or config.gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be positive and finite")
        if not math.isfinite(config.friction_tanh_coefficient) or config.friction_tanh_coefficient <= 0.0:
            raise ValueError("friction_tanh_coefficient must be positive and finite")

        self._gravity_model: OpenArmGravityModel | None = None
        self._latest_motor_states: dict[str, dict[str, float]] | None = None
        self._latest_motor_states_time: float | None = None

    @property
    def action_features(self) -> dict[str, type]:
        """Features produced by this teleoperator."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Follower positions accepted as native-style bilateral references."""
        if self.config.manual_control:
            return {}
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def requires_continuous_feedback(self) -> bool:
        """Tell LeRobot control loops to return each follower observation."""
        return not self.config.manual_control

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    def _prepare_gravity_model(self) -> None:
        """Validate the side-specific URDF before any leader motor is enabled."""
        if self.config.manual_control or not self.config.gravity_compensation:
            return
        assert self.config.side is not None
        if self._gravity_model is None:
            self._gravity_model = OpenArmGravityModel(
                self.config.dynamics_urdf_path,
                self.config.side,
                self.config.gravity_m_s2,
            )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the teleoperator.

        Bilateral mode first validates the dynamics model and current state,
        then enables torque and holds the measured pose until follower feedback
        arrives. Free-motion diagnostic mode leaves torque disabled.
        """

        self._prepare_gravity_model()

        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        if calibrate:
            logger.info(
                "Using the existing OpenArm motor zero. LeRobot will not write or replace zero positions."
            )

        try:
            self.configure()
            if not self.config.manual_control:
                states = self._read_motor_states()
                initial_feedback = {f"{motor}.pos": states[motor]["position"] for motor in self.bus.motors}
                self.bus.enable_torque()
                self._send_bilateral_feedback(initial_feedback)
        except Exception as error:
            try:
                self.bus.disable_torque()
            except Exception as disable_error:
                logger.error("Failed to disable OpenArm leader after connect error: %s", disable_error)
            try:
                self.bus.disconnect(disable_torque=False)
            except Exception as disconnect_error:
                logger.error("Failed to disconnect OpenArm leader after connect error: %s", disconnect_error)
            logger.error("OpenArm leader connect failed: %s", error)
            raise

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """OpenArm motor-zero calibration is managed externally, not by LeRobot."""
        return True

    def calibrate(self) -> None:
        """Reject generic calibration because it would replace the OpenArm motor zero."""
        raise RuntimeError(
            "OpenArm v1 cannot be calibrated with lerobot-calibrate. Stop all controllers and run "
            "the official openarm-can-zero-position-calibration procedure for each arm instead. "
            f"LeRobot expects positions in {OPENARM_V1_COORDINATE_FRAME} degrees."
        )

    def configure(self) -> None:
        """
        Configure the selected leader operating mode.

        Bilateral mode is enabled only after its initial state has been read in
        ``connect``. Free-motion mode explicitly disables torque here.
        """
        if self.config.manual_control:
            self.bus.disable_torque()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get current action from the leader arm.

        This is the main method for teleoperators - it reads the current state
        of the leader arm and returns it as an action that can be sent to a follower.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle.
        """
        start = time.perf_counter()

        action_dict: dict[str, Any] = {}

        # Use sync_read_all_states to get pos/vel/torque in one go
        states = self._read_motor_states()
        for motor in self.bus.motors:
            state = states.get(motor, {})
            action_dict[f"{motor}.pos"] = state.get("position")
            if self.config.use_velocity_and_torque:
                action_dict[f"{motor}.vel"] = state.get("velocity")
                action_dict[f"{motor}.torque"] = state.get("torque")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        return action_dict

    def _read_motor_states(self) -> dict[str, dict[str, float]]:
        states = self.bus.sync_read_all_states()
        self._latest_motor_states = {
            motor: {key: float(value) for key, value in state.items()} for motor, state in states.items()
        }
        self._latest_motor_states_time = time.monotonic()
        return self._latest_motor_states

    def _compensation_motor_states(self) -> dict[str, dict[str, float]]:
        if self._latest_motor_states is not None and self._latest_motor_states_time is not None:
            age_s = time.monotonic() - self._latest_motor_states_time
            if age_s <= self.config.compensation_state_max_age_s:
                return self._latest_motor_states
        return self._read_motor_states()

    def _compensation_torques(self) -> dict[str, float]:
        motor_names = list(self.bus.motors)
        if not self.config.gravity_compensation and not self.config.friction_compensation:
            return dict.fromkeys(motor_names, 0.0)

        states = self._compensation_motor_states()
        gravity = (0.0,) * 7
        if self.config.gravity_compensation:
            if self._gravity_model is None:
                raise RuntimeError("OpenArm leader gravity model is unavailable before connect()")
            gravity = self._gravity_model.gravity_torques(
                tuple(math.radians(states[motor]["position"]) for motor in motor_names[:7])
            )

        torques: dict[str, float] = {}
        for index, motor_name in enumerate(motor_names):
            torque = gravity[index] * self.config.gravity_scale[index] if index < 7 else 0.0
            if self.config.friction_compensation:
                velocity_rad_s = math.radians(states[motor_name]["velocity"])
                torque += bilateral_friction_torque(
                    velocity_rad_s,
                    self.config.friction_fc[index] * self.config.friction_fc_scale[index],
                    self.config.friction_k[index],
                    self.config.friction_fv[index] * self.config.friction_fv_scale[index],
                    self.config.friction_fo[index] * self.config.friction_fo_scale[index],
                    self.config.friction_tanh_coefficient,
                )

            motor_type_name = self.config.motor_config[motor_name][2].upper().replace("-", "_")
            motor_type = getattr(MotorType, motor_type_name)
            torque_limit = MOTOR_LIMIT_PARAMS[motor_type][2]
            if not math.isfinite(torque) or not -torque_limit <= torque <= torque_limit:
                raise RuntimeError(
                    f"OpenArm leader compensation torque is outside the {motor_name} MIT range: "
                    f"value={torque} allowed=[{-torque_limit}, {torque_limit}] Nm"
                )
            torques[motor_name] = torque
        return torques

    def _send_bilateral_feedback(self, feedback: dict[str, float]) -> None:
        motor_names = list(self.bus.motors)
        missing = [f"{motor}.pos" for motor in motor_names if f"{motor}.pos" not in feedback]
        if missing:
            raise ValueError(f"OpenArm leader feedback is missing position keys: {missing}")

        assert self.config.side is not None
        physical_limits = OPENARM_V1_PHYSICAL_JOINT_LIMITS[self.config.side]
        compensation_torques = self._compensation_torques()
        commands: dict[str, tuple[float, float, float, float, float]] = {}
        for index, motor_name in enumerate(motor_names):
            position = float(feedback[f"{motor_name}.pos"])
            velocity = float(feedback.get(f"{motor_name}.vel", 0.0))
            if not math.isfinite(position) or not math.isfinite(velocity):
                raise ValueError(
                    f"OpenArm leader feedback for {motor_name} must contain finite position/velocity"
                )
            minimum, maximum = physical_limits[motor_name]
            if not minimum <= position <= maximum:
                raise ValueError(
                    f"OpenArm leader feedback target is outside the {self.config.side} {motor_name} "
                    f"physical range: value={position} allowed=[{minimum}, {maximum}] deg"
                )
            motor_type_name = self.config.motor_config[motor_name][2].upper().replace("-", "_")
            motor_type = getattr(MotorType, motor_type_name)
            velocity_limit_deg_s = math.degrees(MOTOR_LIMIT_PARAMS[motor_type][1])
            if not -velocity_limit_deg_s <= velocity <= velocity_limit_deg_s:
                raise ValueError(
                    f"OpenArm leader feedback velocity is outside the {motor_name} MIT range: "
                    f"value={velocity} allowed=[{-velocity_limit_deg_s}, {velocity_limit_deg_s}] deg/s"
                )
            commands[motor_name] = (
                self.config.position_kp[index],
                self.config.position_kd[index],
                position,
                velocity,
                compensation_torques[motor_name],
            )

        self.bus._mit_control_batch(commands)

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Apply follower position feedback with native leader gains and compensation."""
        if self.config.manual_control:
            raise RuntimeError("send_feedback is unavailable in manual_control free-motion mode")
        try:
            self._send_bilateral_feedback(feedback)
        except Exception as error:
            try:
                self.bus.disable_torque()
            except Exception as disable_error:
                logger.error("Failed to disable OpenArm leader after feedback error: %s", disable_error)
            logger.error("OpenArm leader feedback failed: %s", error)
            raise

    @check_if_not_connected
    def disable_torque(self) -> None:
        self.bus.disable_torque()

    @check_if_not_connected
    def enable_torque(self) -> None:
        self.bus.enable_torque()

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from teleoperator."""

        self.bus.disconnect(disable_torque=self.config.disable_torque_on_disconnect)
        logger.info(f"{self} disconnected.")
