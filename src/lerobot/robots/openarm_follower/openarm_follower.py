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
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MOTOR_LIMIT_PARAMS, MotorType
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    OPENARM_FALLBACK_JOINT_LIMITS,
    OPENARM_V1_COORDINATE_FRAME,
    OPENARM_V1_PHYSICAL_JOINT_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArmFollowerConfig,
)
from .openarm_dynamics import OpenArmGravityModel, bilateral_friction_torque

logger = logging.getLogger(__name__)


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


def _validate_joint_limits(
    joint_limits: dict[str, tuple[float, float]],
    motor_names: set[str],
    side: str | None,
) -> None:
    missing = motor_names - set(joint_limits)
    unknown = set(joint_limits) - motor_names
    if missing or unknown:
        raise ValueError(
            "joint_limits must contain exactly the configured motors "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})"
        )

    physical_limits = OPENARM_V1_PHYSICAL_JOINT_LIMITS.get(side) if side is not None else None
    for motor_name, limits in joint_limits.items():
        if len(limits) != 2:
            raise ValueError(f"joint_limits[{motor_name!r}] must contain (minimum, maximum)")
        minimum, maximum = limits
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            raise ValueError(f"joint_limits[{motor_name!r}] must be finite and increasing, got {limits}")
        if physical_limits is None:
            continue
        physical_minimum, physical_maximum = physical_limits[motor_name]
        if minimum < physical_minimum or maximum > physical_maximum:
            raise ValueError(
                f"joint_limits[{motor_name!r}]={limits} exceeds the OpenArm v1 {side} physical "
                f"limit {(physical_minimum, physical_maximum)} in {OPENARM_V1_COORDINATE_FRAME} degrees"
            )


class OpenArmFollower(Robot):
    """
    OpenArms Follower Robot which uses CAN bus communication to control 7 DOF arm with a gripper.
    The arm uses Damiao motors in MIT control mode.
    """

    config_class = OpenArmFollowerConfig
    name = "openarm_follower"

    def __init__(self, config: OpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        if config.coordinate_frame != OPENARM_V1_COORDINATE_FRAME:
            raise ValueError(
                f"Unsupported OpenArm coordinate frame {config.coordinate_frame!r}; "
                f"expected {OPENARM_V1_COORDINATE_FRAME!r}"
            )
        if config.side not in (None, "left", "right"):
            raise ValueError("config.side must be either 'left', 'right', or None")

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

        if config.joint_limits is None:
            if config.side == "left":
                config.joint_limits = dict(LEFT_DEFAULT_JOINTS_LIMITS)
            elif config.side == "right":
                config.joint_limits = dict(RIGHT_DEFAULT_JOINTS_LIMITS)
            else:
                config.joint_limits = dict(OPENARM_FALLBACK_JOINT_LIMITS)
                logger.warning(
                    "config.side is not set; using narrow fallback joint limits. Set side to 'left' or "
                    "'right' for the OpenArm v1 deployment limits."
                )
        else:
            config.joint_limits = dict(config.joint_limits)

        assert config.joint_limits is not None
        _validate_joint_limits(config.joint_limits, set(config.motor_config), config.side)
        if config.side is None:
            logger.warning("Physical joint-limit validation is unavailable until config.side is set.")
        logger.info(f"Values used for joint limits: {config.joint_limits}.")

        motor_count = len(config.motor_config)
        _validate_position_gains("position_kp", config.position_kp, motor_count, 500.0)
        _validate_position_gains("position_kd", config.position_kd, motor_count, 5.0)
        _validate_position_gains("trajectory_position_kp", config.trajectory_position_kp, motor_count, 500.0)
        _validate_position_gains("trajectory_position_kd", config.trajectory_position_kd, motor_count, 5.0)
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

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect without modifying the motor-zero coordinate frame.

        ``calibrate`` is accepted for the common Robot API, but OpenArm zero
        calibration is an external hardware setup step and is never run here.
        """

        if self.config.gravity_compensation:
            if self.config.side is None:
                raise ValueError("gravity_compensation requires config.side to be 'left' or 'right'")
            self._gravity_model = OpenArmGravityModel(
                self.config.dynamics_urdf_path,
                self.config.side,
                self.config.gravity_m_s2,
            )

        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        try:
            if calibrate:
                logger.info(
                    "Using the existing OpenArm motor zero. LeRobot will not write or replace zero positions."
                )

            for cam in self.cameras.values():
                cam.connect()

            self.configure()

            self.bus.enable_torque()

            logger.info(f"{self} connected.")
        except BaseException as error:
            try:
                self._disconnect_components(disable_torque=True)
            except BaseException as cleanup_error:
                error.add_note(f"Additional OpenArm connection cleanup error: {cleanup_error!r}")
            raise

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
        """Configure motors with appropriate settings."""
        # TODO(Steven, Pepijn): Slightly different from what it is happening in the leader
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_motor_positions(self, *, require_response: bool = False) -> dict[str, float]:
        """Read motor-zero positions without touching any configured cameras."""
        states = self._read_motor_states(require_response=require_response)
        return {motor: float(states[motor]["position"]) for motor in self.bus.motors}

    def trajectory_position_gains(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return the dedicated CSV-motion PD gains keyed by motor name."""
        motor_names = list(self.bus.motors)
        _validate_position_gains(
            "trajectory_position_kp", self.config.trajectory_position_kp, len(motor_names), 500.0
        )
        _validate_position_gains(
            "trajectory_position_kd", self.config.trajectory_position_kd, len(motor_names), 5.0
        )
        return (
            dict(zip(motor_names, self.config.trajectory_position_kp, strict=True)),
            dict(zip(motor_names, self.config.trajectory_position_kd, strict=True)),
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Get current observation from robot including position, velocity, and torque.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle
        instead of 3 separate reads.
        """
        start = time.perf_counter()

        obs_dict: dict[str, Any] = {}

        states = self._read_motor_states()

        for motor in self.bus.motors:
            state = states.get(motor, {})
            obs_dict[f"{motor}.pos"] = state.get("position", 0.0)
            if self.config.use_velocity_and_torque:
                obs_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
                obs_dict[f"{motor}.torque"] = state.get("torque", 0.0)

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    def _read_motor_states(self, *, require_response: bool = False) -> dict[str, dict[str, float]]:
        if require_response:
            states = self.bus.sync_read_all_states(num_retry=2, require_response=True)
        else:
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
                raise RuntimeError("OpenArm gravity model is unavailable before connect()")
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
                    f"OpenArm compensation torque is outside the {motor_name} MIT range: "
                    f"value={torque} allowed=[{-torque_limit}, {torque_limit}] Nm"
                )
            torques[motor_name] = torque
        return torques

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
        *,
        apply_joint_limits: bool = True,
        apply_max_relative_target: bool = True,
        require_response: bool = False,
    ) -> RobotAction:
        """
        Send action command to robot.

        The action magnitude may be clipped based on safety limits.

        Args:
            action: Dictionary with motor positions (e.g., "joint_1.pos", "joint_2.pos")
            custom_kp: Optional custom kp gains per motor (e.g., {"joint_1": 120.0, "joint_2": 150.0})
            custom_kd: Optional custom kd gains per motor (e.g., {"joint_1": 1.5, "joint_2": 2.0})
            apply_joint_limits: Clip targets to the configured policy/deployment
                limits. Fault handling may disable this only after independently
                validating a fresh measured pose against physical limits.
            apply_max_relative_target: Apply the policy-action relative target
                limiter. Validated deployment trajectories disable this because
                they have their own joint, velocity, clipping, and tracking
                checks.

        Returns:
            The action actually sent (potentially clipped)
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        goal_vel = {key.removesuffix(".vel"): val for key, val in action.items() if key.endswith(".vel")}

        if apply_joint_limits:
            joint_limits = self.config.joint_limits
            assert joint_limits is not None
            for motor_name, position in goal_pos.items():
                if motor_name in joint_limits:
                    min_limit, max_limit = joint_limits[motor_name]
                    clipped_position = max(min_limit, min(max_limit, position))
                    if clipped_position != position:
                        logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
                    goal_pos[motor_name] = clipped_position

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if apply_max_relative_target and self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        compensation_torques = self._compensation_torques()

        # TODO(Steven, Pepijn): Refactor writing
        # Motor name to index mapping for gains
        motor_index = {
            "joint_1": 0,
            "joint_2": 1,
            "joint_3": 2,
            "joint_4": 3,
            "joint_5": 4,
            "joint_6": 5,
            "joint_7": 6,
            "gripper": 7,
        }

        # Use batch MIT control for arm (sends all commands, then collects responses)
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            idx = motor_index.get(motor_name, 0)
            # Use custom gains if provided, otherwise use config defaults
            if custom_kp is not None and motor_name in custom_kp:
                kp = custom_kp[motor_name]
            else:
                kp = (
                    self.config.position_kp[idx]
                    if isinstance(self.config.position_kp, list)
                    else self.config.position_kp
                )
            if custom_kd is not None and motor_name in custom_kd:
                kd = custom_kd[motor_name]
            else:
                kd = (
                    self.config.position_kd[idx]
                    if isinstance(self.config.position_kd, list)
                    else self.config.position_kd
                )
            target_velocity = float(goal_vel.get(motor_name, 0.0))
            motor_type_name = self.config.motor_config[motor_name][2].upper().replace("-", "_")
            motor_type = getattr(MotorType, motor_type_name)
            velocity_limit_deg_s = math.degrees(MOTOR_LIMIT_PARAMS[motor_type][1])
            if not math.isfinite(target_velocity) or not (
                -velocity_limit_deg_s <= target_velocity <= velocity_limit_deg_s
            ):
                raise ValueError(
                    f"OpenArm target velocity is outside the {motor_name} MIT range: "
                    f"value={target_velocity} "
                    f"allowed=[{-velocity_limit_deg_s}, {velocity_limit_deg_s}] deg/s"
                )
            commands[motor_name] = (
                kp,
                kd,
                position_degrees,
                target_velocity,
                compensation_torques[motor_name],
            )

        if require_response:
            self.bus._mit_control_batch(commands, require_response=True)
        else:
            self.bus._mit_control_batch(commands)

        sent_action = {f"{motor}.pos": val for motor, val in goal_pos.items()}
        sent_action.update(
            {f"{motor}.vel": float(goal_vel[motor]) for motor in goal_pos if motor in goal_vel}
        )
        return sent_action

    def _disconnect_components(self, *, disable_torque: bool) -> None:
        """Disconnect live components, optionally requiring verified torque disable."""
        errors: list[BaseException] = []
        if self.bus.is_connected:
            if disable_torque:
                try:
                    self.bus.disable_torque(num_retry=2, require_response=True)
                except BaseException as error:
                    error.add_note("Failed to verify OpenArm torque disable before CAN close")
                    errors.append(error)
            try:
                # Torque handling above is strict. Always close the socket even when a motor ACK is missing.
                self.bus.disconnect(False)
            except BaseException as error:
                error.add_note("Failed while closing the OpenArm CAN bus")
                errors.append(error)

        for name, camera in self.cameras.items():
            if not camera.is_connected:
                continue
            try:
                camera.disconnect()
            except BaseException as error:
                error.add_note(f"Failed while disconnecting OpenArm camera {name!r}")
                errors.append(error)

        logger.info(f"{self} disconnected.")
        if errors:
            first_error = errors[0]
            for additional_error in errors[1:]:
                first_error.add_note(f"Additional OpenArm disconnect error: {additional_error!r}")
            raise first_error

    def disconnect(self) -> None:
        """Disconnect every live component, including a CAN-only partial connection."""
        self._disconnect_components(disable_torque=self.config.disable_torque_on_disconnect)
