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

import logging
import math
import time
from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.robot_utils import precise_sleep

from ..openarm_follower import OPENARM_V1_PHYSICAL_JOINT_LIMITS, OpenArmFollower, OpenArmFollowerConfig
from ..robot import Robot
from .config_bi_openarm_follower import BiOpenArmFollowerConfig
from .confirmation import confirm_openarm_motion
from .deployment_trajectory import (
    MOTOR_NAMES,
    DeploymentTrajectorySample,
    build_return_to_zero_trajectory,
    interpolate_deployment_trajectory,
    load_deployment_trajectory,
    validate_deployment_trajectory,
)

logger = logging.getLogger(__name__)


class BiOpenArmFollower(BimanualMixin, Robot):
    """
    Bimanual OpenArm Follower Arms
    """

    config_class = BiOpenArmFollowerConfig
    name = "bi_openarm_follower"

    def __init__(self, config: BiOpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        # Top-level cameras are opened by `left_arm` for convenience, but their
        # keys stay unprefixed in observations (tracked via `_top_level_cam_keys`).
        self._top_level_cam_keys = set(config.cameras)
        _collisions = self._top_level_cam_keys & set(
            config.left_arm_config.cameras
        ) | self._top_level_cam_keys & set(config.right_arm_config.cameras)
        if _collisions:
            raise ValueError(
                f"Top-level camera names collide with per-arm camera names: {sorted(_collisions)}"
            )
        left_arm_cameras = {**config.left_arm_config.cameras, **config.cameras}

        left_arm_config = OpenArmFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=self.calibration_dir,
            port=config.left_arm_config.port,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            use_velocity_and_torque=config.left_arm_config.use_velocity_and_torque,
            max_relative_target=config.left_arm_config.max_relative_target,
            cameras=left_arm_cameras,
            side=config.left_arm_config.side,
            coordinate_frame=config.left_arm_config.coordinate_frame,
            can_interface=config.left_arm_config.can_interface,
            use_can_fd=config.left_arm_config.use_can_fd,
            can_bitrate=config.left_arm_config.can_bitrate,
            can_data_bitrate=config.left_arm_config.can_data_bitrate,
            motor_config=config.left_arm_config.motor_config,
            position_kd=config.left_arm_config.position_kd,
            position_kp=config.left_arm_config.position_kp,
            trajectory_position_kd=config.left_arm_config.trajectory_position_kd,
            trajectory_position_kp=config.left_arm_config.trajectory_position_kp,
            gravity_compensation=config.left_arm_config.gravity_compensation,
            friction_compensation=config.left_arm_config.friction_compensation,
            dynamics_urdf_path=config.left_arm_config.dynamics_urdf_path,
            compensation_state_max_age_s=config.left_arm_config.compensation_state_max_age_s,
            gravity_m_s2=config.left_arm_config.gravity_m_s2,
            gravity_scale=config.left_arm_config.gravity_scale,
            friction_tanh_coefficient=config.left_arm_config.friction_tanh_coefficient,
            friction_fc=config.left_arm_config.friction_fc,
            friction_k=config.left_arm_config.friction_k,
            friction_fv=config.left_arm_config.friction_fv,
            friction_fo=config.left_arm_config.friction_fo,
            friction_fc_scale=config.left_arm_config.friction_fc_scale,
            friction_fv_scale=config.left_arm_config.friction_fv_scale,
            friction_fo_scale=config.left_arm_config.friction_fo_scale,
            joint_limits=config.left_arm_config.joint_limits,
        )

        right_arm_config = OpenArmFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=self.calibration_dir,
            port=config.right_arm_config.port,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            use_velocity_and_torque=config.right_arm_config.use_velocity_and_torque,
            max_relative_target=config.right_arm_config.max_relative_target,
            cameras=config.right_arm_config.cameras,
            side=config.right_arm_config.side,
            coordinate_frame=config.right_arm_config.coordinate_frame,
            can_interface=config.right_arm_config.can_interface,
            use_can_fd=config.right_arm_config.use_can_fd,
            can_bitrate=config.right_arm_config.can_bitrate,
            can_data_bitrate=config.right_arm_config.can_data_bitrate,
            motor_config=config.right_arm_config.motor_config,
            position_kd=config.right_arm_config.position_kd,
            position_kp=config.right_arm_config.position_kp,
            trajectory_position_kd=config.right_arm_config.trajectory_position_kd,
            trajectory_position_kp=config.right_arm_config.trajectory_position_kp,
            gravity_compensation=config.right_arm_config.gravity_compensation,
            friction_compensation=config.right_arm_config.friction_compensation,
            dynamics_urdf_path=config.right_arm_config.dynamics_urdf_path,
            compensation_state_max_age_s=config.right_arm_config.compensation_state_max_age_s,
            gravity_m_s2=config.right_arm_config.gravity_m_s2,
            gravity_scale=config.right_arm_config.gravity_scale,
            friction_tanh_coefficient=config.right_arm_config.friction_tanh_coefficient,
            friction_fc=config.right_arm_config.friction_fc,
            friction_k=config.right_arm_config.friction_k,
            friction_fv=config.right_arm_config.friction_fv,
            friction_fo=config.right_arm_config.friction_fo,
            friction_fc_scale=config.right_arm_config.friction_fc_scale,
            friction_fv_scale=config.right_arm_config.friction_fv_scale,
            friction_fo_scale=config.right_arm_config.friction_fo_scale,
            joint_limits=config.right_arm_config.joint_limits,
        )

        self.left_arm = OpenArmFollower(left_arm_config)
        self.right_arm = OpenArmFollower(right_arm_config)

        self._deployment_trajectories: dict[str, list[DeploymentTrajectorySample]] = {}
        if config.deployment_trajectory_profile is not None:
            if self.left_arm.config.side != "left" or self.right_arm.config.side != "right":
                raise ValueError(
                    "deployment trajectories require left_arm_config.side=left and "
                    "right_arm_config.side=right"
                )
            profile_dir = Path(config.deployment_trajectory_profile).expanduser()
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                # Validate dedicated motion gains before touching hardware.
                arm.trajectory_position_gains()
                trajectory = load_deployment_trajectory(profile_dir / f"{side}_arm.csv", side)
                assert arm.config.joint_limits is not None
                validate_deployment_trajectory(
                    trajectory, arm.config.joint_limits, config.startup_trajectory_speed
                )
                validate_deployment_trajectory(
                    trajectory, arm.config.joint_limits, config.shutdown_replay_speed
                )
                self._deployment_trajectories[side] = trajectory
            logger.info("Loaded OpenArm deployment trajectory profile: %s", profile_dir)

        # Only for compatibility with other parts of the codebase that expect a `robot.cameras` attribute
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def has_policy_deployment_trajectory(self) -> bool:
        return bool(self._deployment_trajectories)


    @staticmethod
    def _raise_after_all_safety_attempts(
        failures: list[tuple[str, BaseException]], operation: str
    ) -> None:
        if not failures:
            return
        interruptions = [item for item in failures if not isinstance(item[1], Exception)]
        primary_name, primary_error = (interruptions or failures)[0]
        for name, error in failures:
            if error is not primary_error:
                primary_error.add_note(f"Additional {name} {operation} error: {error!r}")
        primary_error.add_note(
            f"{operation} failed or was interrupted at {primary_name}; all targets were still attempted"
        )
        raise primary_error
    def _disable_both_arms(self) -> None:
        failures: list[tuple[str, BaseException]] = []
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            if not arm.bus.is_connected:
                continue
            try:
                arm.bus.disable_torque(num_retry=2, require_response=True)
            except BaseException as error:
                error.add_note(f"Failed to disable {side} OpenArm follower after deployment motion error")
                failures.append((side, error))
                logger.exception("Failed to disable %s OpenArm after deployment motion error", side)

        self._raise_after_all_safety_attempts(failures, "follower torque disable")

    @staticmethod
    def _set_arm_disable_torque_on_disconnect(arm_config: Any, arm: Any, disable_torque: bool) -> None:
        arm_config.disable_torque_on_disconnect = disable_torque
        arm.config.disable_torque_on_disconnect = disable_torque

    def _validated_fault_hold_positions(
        self,
        side: str,
        current: dict[str, float],
    ) -> dict[str, float]:
        """Validate a fresh measured hold against physical, not policy, limits."""
        physical_limits = OPENARM_V1_PHYSICAL_JOINT_LIMITS[side]
        tolerance = self.config.deployment_start_limit_tolerance_deg
        validated: dict[str, float] = {}
        for motor_name in MOTOR_NAMES:
            position = float(current[motor_name])
            if not math.isfinite(position):
                raise RuntimeError(
                    f"OpenArm fault hold position is not finite: side={side} "
                    f"motor={motor_name} actual={position}"
                )
            minimum, maximum = physical_limits[motor_name]
            bounded = min(max(position, minimum), maximum)
            excess = abs(position - bounded)
            if excess > tolerance:
                raise RuntimeError(
                    f"OpenArm fault hold exceeds the physical limit: side={side} "
                    f"motor={motor_name} actual={position:.3f} "
                    f"limit=[{minimum:.3f}, {maximum:.3f}] excess={excess:.3f} "
                    f"tolerance={tolerance:.3f} deg"
                )
            if excess > 0.0:
                logger.warning(
                    "Clamping measured OpenArm fault hold to its physical boundary: side=%s "
                    "motor=%s actual=%.3f clamped=%.3f excess=%.3f deg",
                    side,
                    motor_name,
                    position,
                    bounded,
                    excess,
                )
            validated[motor_name] = bounded
        return validated

    def _secure_follower_current_position_hold(self, side: str, arm_config: Any, arm: Any) -> bool:
        """Enable one follower and issue a fresh measured-position hold command."""
        self._set_arm_disable_torque_on_disconnect(arm_config, arm, True)
        if not arm.bus.is_connected:
            logger.critical(
                "%s OpenArm follower is not held after fault: CAN is disconnected and torque state "
                "is unknown.",
                side,
            )
            return False

        try:
            current = arm.get_motor_positions(require_response=True)
            hold_positions = self._validated_fault_hold_positions(side, current)
            trajectory_kp, trajectory_kd = arm.trajectory_position_gains()
            hold_action = {f"{motor_name}.pos": hold_positions[motor_name] for motor_name in MOTOR_NAMES}
            # Preload the measured reference before enabling a device that may
            # still contain a stale command, then confirm it again afterwards.
            arm.send_action(
                hold_action,
                custom_kp=trajectory_kp,
                custom_kd=trajectory_kd,
                apply_joint_limits=False,
                apply_max_relative_target=False,
                require_response=True,
            )
            arm.bus.enable_torque(num_retry=2, require_response=True)
            sent = arm.send_action(
                hold_action,
                custom_kp=trajectory_kp,
                custom_kd=trajectory_kd,
                apply_joint_limits=False,
                apply_max_relative_target=False,
                require_response=True,
            )
            for motor_name in MOTOR_NAMES:
                requested = hold_positions[motor_name]
                actual = float(sent[f"{motor_name}.pos"])
                if not math.isclose(actual, requested, abs_tol=1e-6):
                    raise RuntimeError(
                        f"OpenArm fault hold target changed after physical validation: side={side} "
                        f"motor={motor_name} requested={requested:.3f} sent={actual:.3f} deg"
                    )
        except BaseException as error:
            try:
                arm.bus.disable_torque(num_retry=2, require_response=True)
            except BaseException as disable_error:
                disable_status = f"torque-disable failed: {disable_error}"
                logger.exception("Failed to disable %s OpenArm follower after hold error", side)
            else:
                disable_status = "torque-disable response verified"
            logger.critical(
                "%s OpenArm follower is not held after fault: %s. Fail-safe status: %s.",
                side,
                error,
                disable_status,
            )
            return False

        self._set_arm_disable_torque_on_disconnect(arm_config, arm, False)
        logger.critical(
            "%s OpenArm follower is held by a current-position command issued after torque enable; "
            "torque will be preserved across CAN disconnect.",
            side,
        )
        return True

    def _hold_both_arms_after_shutdown_error(self) -> bool:
        """Secure both followers and report only holds that were established."""
        statuses = {
            side: self._secure_follower_current_position_hold(side, arm_config, arm)
            for side, arm_config, arm in (
                ("left", self.config.left_arm_config, self.left_arm),
                ("right", self.config.right_arm_config, self.right_arm),
            )
        }
        logger.critical(
            "OpenArm follower shutdown hold status: left=%s; right=%s. Support the arms before "
            "removing power.",
            "held" if statuses["left"] else "not held",
            "held" if statuses["right"] else "not held",
        )
        return all(statuses.values())

    def secure_intervention_after_fault(self, teleop: Any) -> dict[str, bool]:
        """Secure all four OpenArm devices in place without a return trajectory.

        Each follower and both leaders receive fresh measured-position holds.
        A device is preserved across disconnect only after its torque-enable and
        post-enable hold operations complete; failed devices are disabled.
        """
        statuses = {
            f"follower_{side}": self._secure_follower_current_position_hold(side, arm_config, arm)
            for side, arm_config, arm in (
                ("left", self.config.left_arm_config, self.left_arm),
                ("right", self.config.right_arm_config, self.right_arm),
            )
        }

        secure_leaders = getattr(teleop, "secure_current_position_hold", None)
        try:
            if not callable(secure_leaders):
                raise RuntimeError("teleoperator does not provide secure_current_position_hold()")
            leaders_held = bool(secure_leaders())
            if not leaders_held:
                teleop.disable_torque(require_response=True)
        except BaseException as error:
            leaders_held = False
            try:
                teleop.disable_torque(require_response=True)
            except BaseException:
                logger.exception("Failed to disable OpenArm leaders after fault hold error")
            logger.critical("OpenArm leaders are not held after intervention fault: %s", error)

        statuses["leader_left"] = leaders_held
        statuses["leader_right"] = leaders_held
        logger.critical(
            "OpenArm intervention fault handling completed without return motion: %s",
            ", ".join(f"{device}={'held' if held else 'not held'}" for device, held in statuses.items()),
        )
        return statuses

    def _read_deployment_positions(self) -> dict[str, tuple[float, ...]]:
        left_positions = self.left_arm.get_motor_positions()
        right_positions = self.right_arm.get_motor_positions()
        return {
            "left": tuple(left_positions[motor] for motor in MOTOR_NAMES),
            "right": tuple(right_positions[motor] for motor in MOTOR_NAMES),
        }

    def _validate_intervention_teleop(self, teleop: Any) -> None:
        """Validate the bimanual leader contract before coordinated motion."""
        if getattr(teleop, "name", None) != "bi_openarm_leader":
            raise ValueError("OpenArm DAgger deployment trajectories require --teleop.type=bi_openarm_leader")
        if not bool(getattr(teleop, "requires_continuous_feedback", False)):
            raise ValueError(
                "OpenArm DAgger deployment trajectories require bilateral leader control; "
                "manual_control must be false on both leaders"
            )
        required_methods = (
            "get_action",
            "send_feedback",
            "secure_current_position_hold",
            "disable_torque",
            "hold_position_after_shutdown_error",
        )
        missing_methods = [name for name in required_methods if not callable(getattr(teleop, name, None))]
        if missing_methods:
            raise ValueError(
                f"OpenArm DAgger teleoperator is missing coordinated deployment methods: {missing_methods}"
            )
        expected_features = {
            f"{side}_{motor_name}.pos" for side in ("left", "right") for motor_name in MOTOR_NAMES
        }
        missing_features = sorted(expected_features - set(teleop.feedback_features))
        if missing_features:
            raise ValueError(
                f"OpenArm DAgger teleoperator is missing deployment feedback features: {missing_features}"
            )

    def _read_intervention_positions(self, teleop: Any) -> dict[str, tuple[float, ...]]:
        action = teleop.get_action()
        measured: dict[str, tuple[float, ...]] = {}
        for side in ("left", "right"):
            side_positions: list[float] = []
            physical_limits = OPENARM_V1_PHYSICAL_JOINT_LIMITS[side]
            for motor_name in MOTOR_NAMES:
                key = f"{side}_{motor_name}.pos"
                if key not in action:
                    raise RuntimeError(f"OpenArm leader deployment state is missing {key}")
                position = float(action[key])
                if not math.isfinite(position):
                    raise RuntimeError(
                        f"OpenArm leader deployment state is not finite: side={side} "
                        f"motor={motor_name} actual={position}"
                    )
                minimum, maximum = physical_limits[motor_name]
                if not minimum <= position <= maximum:
                    raise RuntimeError(
                        f"OpenArm leader deployment state exceeds the physical limit: side={side} "
                        f"motor={motor_name} actual={position:.3f} "
                        f"limit=[{minimum:.3f}, {maximum:.3f}] deg"
                    )
                side_positions.append(position)
            measured[side] = tuple(side_positions)
        return measured

    @staticmethod
    def _as_action(positions: tuple[float, ...]) -> RobotAction:
        return {
            f"{motor_name}.pos": position for motor_name, position in zip(MOTOR_NAMES, positions, strict=True)
        }

    @classmethod
    def _as_bimanual_action(cls, targets: dict[str, tuple[float, ...]]) -> RobotAction:
        return {
            f"{side}_{key}": value
            for side in ("left", "right")
            for key, value in cls._as_action(targets[side]).items()
        }

    def _check_deployment_tracking(
        self,
        measured: dict[str, tuple[float, ...]],
        targets: dict[str, tuple[float, ...]],
        *,
        role: str,
        phase: str,
        reference_time_s: float,
    ) -> None:
        error_limit = self.config.deployment_tracking_error_deg
        for side in ("left", "right"):
            for motor_name, target, actual in zip(MOTOR_NAMES, targets[side], measured[side], strict=True):
                tracking_error = abs(actual - target)
                if tracking_error > error_limit:
                    raise RuntimeError(
                        f"OpenArm deployment trajectory tracking error: role={role} side={side} "
                        f"phase={phase} time={reference_time_s:.3f}s motor={motor_name} "
                        f"target={target:.3f} actual={actual:.3f} error={tracking_error:.3f} "
                        f"limit={error_limit:.3f} deg"
                    )

    def _send_deployment_targets(
        self,
        targets: dict[str, tuple[float, ...]],
        *,
        phase: str,
        reference_time_s: float,
    ) -> None:
        sent_targets: dict[str, RobotAction] = {}
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            trajectory_kp, trajectory_kd = arm.trajectory_position_gains()
            sent_targets[side] = arm.send_action(
                self._as_action(targets[side]),
                custom_kp=trajectory_kp,
                custom_kd=trajectory_kd,
                apply_max_relative_target=False,
            )

        for side in ("left", "right"):
            for motor_name, requested in zip(MOTOR_NAMES, targets[side], strict=True):
                sent = float(sent_targets[side][f"{motor_name}.pos"])
                if not math.isclose(sent, requested, abs_tol=1e-6):
                    raise RuntimeError(
                        f"OpenArm deployment trajectory target was clipped: side={side} "
                        f"motor={motor_name} requested={requested:.3f} sent={sent:.3f} deg"
                    )

        self._check_deployment_tracking(
            self._read_deployment_positions(),
            targets,
            role="follower",
            phase=phase,
            reference_time_s=reference_time_s,
        )

    def _send_coordinated_deployment_targets(
        self,
        teleop: Any,
        follower_targets: dict[str, tuple[float, ...]],
        leader_targets: dict[str, tuple[float, ...]],
        *,
        phase: str,
        reference_time_s: float,
    ) -> None:
        self._send_deployment_targets(
            follower_targets,
            phase=phase,
            reference_time_s=reference_time_s,
        )
        teleop.send_feedback(self._as_bimanual_action(leader_targets))
        self._check_deployment_tracking(
            self._read_intervention_positions(teleop),
            leader_targets,
            role="leader",
            phase=phase,
            reference_time_s=reference_time_s,
        )

    def _clamp_measured_blend_start(
        self,
        start: dict[str, tuple[float, ...]],
        phase: str,
    ) -> dict[str, tuple[float, ...]]:
        """Clamp only small encoder overshoots before interpolating a measured pose."""
        clamped: dict[str, tuple[float, ...]] = {}
        tolerance = self.config.deployment_start_limit_tolerance_deg
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            joint_limits = arm.config.joint_limits
            assert joint_limits is not None
            side_positions: list[float] = []
            for motor_name, position in zip(MOTOR_NAMES, start[side], strict=True):
                if not math.isfinite(position):
                    raise RuntimeError(
                        f"OpenArm deployment blend start is not finite: side={side} phase={phase} "
                        f"motor={motor_name} actual={position}"
                    )
                minimum, maximum = joint_limits[motor_name]
                bounded = min(max(position, minimum), maximum)
                excess = abs(position - bounded)
                if excess > tolerance:
                    raise RuntimeError(
                        f"OpenArm deployment blend start exceeds the joint limit: side={side} "
                        f"phase={phase} motor={motor_name} actual={position:.3f} "
                        f"limit=[{minimum:.3f}, {maximum:.3f}] excess={excess:.3f} "
                        f"tolerance={tolerance:.3f} deg"
                    )
                if excess > 0.0:
                    logger.warning(
                        "Clamping measured OpenArm blend start to its joint limit: side=%s phase=%s "
                        "motor=%s actual=%.3f clamped=%.3f excess=%.3f deg",
                        side,
                        phase,
                        motor_name,
                        position,
                        bounded,
                        excess,
                    )
                side_positions.append(bounded)
            clamped[side] = tuple(side_positions)
        return clamped

    def _blend_deployment_positions(
        self,
        start: dict[str, tuple[float, ...]],
        target: dict[str, tuple[float, ...]],
        duration_s: float,
        phase: str,
    ) -> None:
        if duration_s <= 0.0:
            self._send_deployment_targets(target, phase=phase, reference_time_s=0.0)
            return
        start = self._clamp_measured_blend_start(start, phase)
        steps = max(math.ceil(duration_s * self.config.deployment_control_frequency_hz), 1)
        start_time = time.perf_counter()
        for step in range(1, steps + 1):
            elapsed_s = duration_s * step / steps
            sleep_s = start_time + elapsed_s - time.perf_counter()
            if sleep_s > 0.0:
                precise_sleep(sleep_s)
            alpha = step / steps
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            positions = {
                side: tuple(
                    before + (after - before) * smooth_alpha
                    for before, after in zip(start[side], target[side], strict=True)
                )
                for side in ("left", "right")
            }
            self._send_deployment_targets(positions, phase=phase, reference_time_s=elapsed_s)

    def _blend_intervention_positions(
        self,
        teleop: Any,
        follower_start: dict[str, tuple[float, ...]],
        leader_start: dict[str, tuple[float, ...]],
        target: dict[str, tuple[float, ...]],
        duration_s: float,
        phase: str,
    ) -> None:
        """Blend both roles from their own measured pose to one shared target."""
        if duration_s <= 0.0:
            self._send_coordinated_deployment_targets(
                teleop,
                target,
                target,
                phase=phase,
                reference_time_s=0.0,
            )
            return
        follower_start = self._clamp_measured_blend_start(follower_start, phase)
        steps = max(math.ceil(duration_s * self.config.deployment_control_frequency_hz), 1)
        start_time = time.perf_counter()
        for step in range(1, steps + 1):
            elapsed_s = duration_s * step / steps
            sleep_s = start_time + elapsed_s - time.perf_counter()
            if sleep_s > 0.0:
                precise_sleep(sleep_s)
            alpha = step / steps
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            follower_targets = {
                side: tuple(
                    before + (after - before) * smooth_alpha
                    for before, after in zip(follower_start[side], target[side], strict=True)
                )
                for side in ("left", "right")
            }
            leader_targets = {
                side: tuple(
                    before + (after - before) * smooth_alpha
                    for before, after in zip(leader_start[side], target[side], strict=True)
                )
                for side in ("left", "right")
            }

            self._send_coordinated_deployment_targets(
                teleop,
                follower_targets,
                leader_targets,
                phase=phase,
                reference_time_s=elapsed_s,
            )

    def _replay_deployment_trajectories(
        self,
        trajectories: dict[str, list[DeploymentTrajectorySample]],
        speed_scale: float,
        phase: str,
    ) -> None:
        recorded_duration_s = max(samples[-1].time_s for samples in trajectories.values())
        playback_duration_s = recorded_duration_s / speed_scale
        steps = max(math.ceil(playback_duration_s * self.config.deployment_control_frequency_hz), 1)
        start_time = time.perf_counter()
        for step in range(steps + 1):
            elapsed_s = playback_duration_s * step / steps
            if step > 0:
                sleep_s = start_time + elapsed_s - time.perf_counter()
                if sleep_s > 0.0:
                    precise_sleep(sleep_s)
            recorded_time_s = elapsed_s * speed_scale
            targets = {
                side: interpolate_deployment_trajectory(samples, recorded_time_s)
                for side, samples in trajectories.items()
            }
            self._send_deployment_targets(targets, phase=phase, reference_time_s=elapsed_s)

    def _replay_intervention_trajectories(
        self,
        teleop: Any,
        trajectories: dict[str, list[DeploymentTrajectorySample]],
        speed_scale: float,
        phase: str,
    ) -> None:
        """Replay one time-indexed target stream on both leaders and followers."""
        recorded_duration_s = max(samples[-1].time_s for samples in trajectories.values())
        playback_duration_s = recorded_duration_s / speed_scale
        steps = max(math.ceil(playback_duration_s * self.config.deployment_control_frequency_hz), 1)
        start_time = time.perf_counter()
        for step in range(steps + 1):
            elapsed_s = playback_duration_s * step / steps
            if step > 0:
                sleep_s = start_time + elapsed_s - time.perf_counter()
                if sleep_s > 0.0:
                    precise_sleep(sleep_s)
            recorded_time_s = elapsed_s * speed_scale
            targets = {
                side: interpolate_deployment_trajectory(samples, recorded_time_s)
                for side, samples in trajectories.items()
            }
            self._send_coordinated_deployment_targets(
                teleop,
                targets,
                targets,
                phase=phase,
                reference_time_s=elapsed_s,
            )

    def _disable_intervention_hardware(self, teleop: Any) -> None:
        failures: list[tuple[str, BaseException]] = []
        try:
            self._disable_both_arms()
        except BaseException as error:
            failures.append(("followers", error))
            logger.exception("Failed to disable both OpenArm followers after deployment motion error")
        try:
            teleop.disable_torque(require_response=True)
        except BaseException as error:
            failures.append(("leaders", error))
            logger.exception("Failed to disable both OpenArm leaders after deployment motion error")

        self._raise_after_all_safety_attempts(failures, "intervention torque disable")

    def _hold_intervention_hardware_after_shutdown_error(self, teleop: Any) -> None:
        failures: list[tuple[str, BaseException]] = []
        try:
            self._hold_both_arms_after_shutdown_error()
        except BaseException as error:
            failures.append(("followers", error))
            logger.exception("Failed to preserve both OpenArm followers after shutdown return error")
        try:
            teleop.hold_position_after_shutdown_error()
        except BaseException as error:
            failures.append(("leaders", error))
            logger.exception("Failed to preserve both OpenArm leaders after shutdown return error")

        self._raise_after_all_safety_attempts(failures, "intervention position hold")

    def prepare_for_intervention_deployment(self, teleop: Any) -> bool:
        """Move all four OpenArm devices through zero and the task-ready CSV."""
        if not self.has_policy_deployment_trajectory:
            raise ValueError("OpenArm DAgger intervention requires --robot.deployment_trajectory_profile")
        self._validate_intervention_teleop(teleop)

        logger.info("Moving both OpenArm leaders and followers to exact motor zero before CSV replay...")
        zero_targets = {"left": (0.0,) * len(MOTOR_NAMES), "right": (0.0,) * len(MOTOR_NAMES)}
        try:
            self._blend_intervention_positions(
                teleop,
                self._read_deployment_positions(),
                self._read_intervention_positions(teleop),
                zero_targets,
                self.config.startup_zero_pose_duration_s,
                "startup-zero",
            )

            first_targets = {
                side: samples[0].positions_deg for side, samples in self._deployment_trajectories.items()
            }
            if self.config.startup_trajectory_blend_s > 0.0:
                self._blend_intervention_positions(
                    teleop,
                    self._read_deployment_positions(),
                    self._read_intervention_positions(teleop),
                    first_targets,
                    self.config.startup_trajectory_blend_s,
                    "startup-blend",
                )

            logger.info(
                "Replaying synchronized leader/follower task-ready CSV trajectory at %.2fx...",
                self.config.startup_trajectory_speed,
            )
            self._replay_intervention_trajectories(
                teleop,
                self._deployment_trajectories,
                self.config.startup_trajectory_speed,
                "startup-replay",
            )

            # Switch the followers from trajectory gains to policy gains at the
            # same pose, then switch the leaders from CSV references to measured
            # follower feedback without a target discontinuity.
            final_action = {
                f"{side}_{motor_name}.pos": position
                for side, samples in self._deployment_trajectories.items()
                for motor_name, position in zip(MOTOR_NAMES, samples[-1].positions_deg, strict=True)
            }
            self.send_action(final_action)
            follower_measured = self._read_deployment_positions()
            teleop.send_feedback(self._as_bimanual_action(follower_measured))
            self._check_deployment_tracking(
                self._read_intervention_positions(teleop),
                follower_measured,
                role="leader",
                phase="startup-handover",
                reference_time_s=0.0,
            )
        except BaseException as error:
            try:
                self._disable_intervention_hardware(teleop)
            except BaseException as cleanup_error:
                error.add_note(f"Additional intervention startup cleanup error: {cleanup_error!r}")
                logger.exception("Failed to disable all OpenArm intervention devices after startup error")
            raise

        logger.info(
            "OpenArm leaders and followers are task-ready; policy inference and bilateral feedback "
            "gains are active."
        )
        return True

    def prepare_for_policy_deployment(self) -> bool:
        """Move both followers through motor zero and the task-ready CSV before inference."""
        if not self.has_policy_deployment_trajectory:
            return False

        logger.info("Moving both OpenArm followers to exact motor zero before CSV replay...")
        zero_targets = {"left": (0.0,) * len(MOTOR_NAMES), "right": (0.0,) * len(MOTOR_NAMES)}
        try:
            current = self._read_deployment_positions()
            self._blend_deployment_positions(
                current,
                zero_targets,
                self.config.startup_zero_pose_duration_s,
                "startup-zero",
            )

            first_targets = {
                side: samples[0].positions_deg for side, samples in self._deployment_trajectories.items()
            }
            if self.config.startup_trajectory_blend_s > 0.0:
                current = self._read_deployment_positions()
                self._blend_deployment_positions(
                    current,
                    first_targets,
                    self.config.startup_trajectory_blend_s,
                    "startup-blend",
                )

            logger.info(
                "Replaying task-ready OpenArm CSV trajectory at %.2fx...",
                self.config.startup_trajectory_speed,
            )
            self._replay_deployment_trajectories(
                self._deployment_trajectories,
                self.config.startup_trajectory_speed,
                "startup-replay",
            )

            # MIT gains are carried in each command. Re-send the same final pose
            # once with policy gains so the gain transition cannot change pose.
            final_action = {
                f"{side}_{motor_name}.pos": position
                for side, samples in self._deployment_trajectories.items()
                for motor_name, position in zip(MOTOR_NAMES, samples[-1].positions_deg, strict=True)
            }
            self.send_action(final_action)
        except BaseException as error:
            try:
                self._disable_both_arms()
            except BaseException as cleanup_error:
                error.add_note(f"Additional follower startup cleanup error: {cleanup_error!r}")
                logger.exception("Failed to disable both OpenArm followers after startup error")
            raise

        logger.info("OpenArm followers are task-ready; policy inference gains are active.")
        return True

    @staticmethod
    def _confirm_shutdown_return() -> bool:
        return confirm_openarm_motion(
            "Return both OpenArm followers to motor zero? [yes/no]: ",
            "Shutdown return was not confirmed; choosing no.",
        )

    def finish_policy_deployment(self) -> bool:
        """Return task-ready followers along the reversed CSV path to exact motor zero."""
        if not self.has_policy_deployment_trajectory:
            return False

        try:
            current = self._read_deployment_positions()
            task_targets = {
                side: samples[-1].positions_deg for side, samples in self._deployment_trajectories.items()
            }
            warnings: list[str] = []
            for side in ("left", "right"):
                for motor_name, actual, target in zip(
                    MOTOR_NAMES, current[side], task_targets[side], strict=True
                ):
                    difference = abs(actual - target)
                    if difference > self.config.shutdown_task_pose_warn_deg:
                        warnings.append(
                            f"side={side} motor={motor_name} actual={actual:.3f} "
                            f"task_ready={target:.3f} difference={difference:.3f} deg"
                        )

            if warnings:
                logger.warning(
                    "One or more OpenArm joints are farther than %.3f deg from the task-ready pose:",
                    self.config.shutdown_task_pose_warn_deg,
                )
                for warning in warnings:
                    logger.warning("  %s", warning)
                if not self._confirm_shutdown_return():
                    logger.info("Shutdown return declined; disabling motors in place.")
                    return True

            logger.info(
                "Blending both OpenArm followers to the task-ready pose over %.1fs...",
                self.config.shutdown_task_pose_blend_s,
            )
            self._blend_deployment_positions(
                current,
                task_targets,
                self.config.shutdown_task_pose_blend_s,
                "shutdown-task-pose",
            )
            return_trajectories = {
                side: build_return_to_zero_trajectory(samples, self.config.shutdown_zero_transition_s)
                for side, samples in self._deployment_trajectories.items()
            }
            logger.info(
                "Replaying reversed OpenArm CSV trajectory to motor zero at %.2fx...",
                self.config.shutdown_replay_speed,
            )
            self._replay_deployment_trajectories(
                return_trajectories,
                self.config.shutdown_replay_speed,
                "shutdown-replay",
            )
        except BaseException as error:
            try:
                if self.config.hold_position_on_shutdown_error:
                    self._hold_both_arms_after_shutdown_error()
                else:
                    self._disable_both_arms()
            except BaseException as cleanup_error:
                error.add_note(f"Additional follower shutdown cleanup error: {cleanup_error!r}")
                logger.exception("Failed to secure both OpenArm followers after shutdown error")
            raise

        logger.info("Both OpenArm followers reached exact motor zero.")
        return True

    def finish_intervention_deployment(self, teleop: Any) -> bool:
        """Return all four OpenArm devices along the reversed CSV path to zero."""
        if not self.has_policy_deployment_trajectory:
            raise ValueError("OpenArm DAgger intervention requires --robot.deployment_trajectory_profile")
        self._validate_intervention_teleop(teleop)

        try:
            follower_current = self._read_deployment_positions()
            leader_current = self._read_intervention_positions(teleop)
            task_targets = {
                side: samples[-1].positions_deg for side, samples in self._deployment_trajectories.items()
            }
            warnings: list[str] = []
            for role, measured in (("leader", leader_current), ("follower", follower_current)):
                for side in ("left", "right"):
                    for motor_name, actual, target in zip(
                        MOTOR_NAMES, measured[side], task_targets[side], strict=True
                    ):
                        difference = abs(actual - target)
                        if difference > self.config.shutdown_task_pose_warn_deg:
                            warnings.append(
                                f"role={role} side={side} motor={motor_name} actual={actual:.3f} "
                                f"task_ready={target:.3f} difference={difference:.3f} deg"
                            )

            if warnings:
                logger.warning(
                    "One or more OpenArm leader/follower joints are farther than %.3f deg "
                    "from the task-ready pose:",
                    self.config.shutdown_task_pose_warn_deg,
                )
                for warning in warnings:
                    logger.warning("  %s", warning)
                if not self._confirm_shutdown_return():
                    logger.info("Coordinated shutdown return declined; disabling all devices in place.")
                    return True

            logger.info(
                "Blending both OpenArm leaders and followers to the task-ready pose over %.1fs...",
                self.config.shutdown_task_pose_blend_s,
            )
            self._blend_intervention_positions(
                teleop,
                follower_current,
                leader_current,
                task_targets,
                self.config.shutdown_task_pose_blend_s,
                "shutdown-task-pose",
            )
            return_trajectories = {
                side: build_return_to_zero_trajectory(samples, self.config.shutdown_zero_transition_s)
                for side, samples in self._deployment_trajectories.items()
            }
            logger.info(
                "Replaying synchronized reversed leader/follower CSV trajectory to motor zero at %.2fx...",
                self.config.shutdown_replay_speed,
            )
            self._replay_intervention_trajectories(
                teleop,
                return_trajectories,
                self.config.shutdown_replay_speed,
                "shutdown-replay",
            )
        except BaseException as error:
            try:
                if self.config.hold_position_on_shutdown_error:
                    self._hold_intervention_hardware_after_shutdown_error(teleop)
                else:
                    self._disable_intervention_hardware(teleop)
            except BaseException as cleanup_error:
                error.add_note(f"Additional intervention shutdown cleanup error: {cleanup_error!r}")
                logger.exception("Failed to secure all OpenArm devices after shutdown error")
            raise

        logger.info("Both OpenArm leaders and followers reached exact motor zero.")
        return True

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._motors_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for k, v in self.left_arm._cameras_ft.items():
            out[k if k in self._top_level_cam_keys else f"left_{k}"] = v
        for k, v in self.right_arm._cameras_ft.items():
            out[f"right_{k}"] = v
        return out

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}

        # Add "left_" prefix to per-arm keys; keep top-level camera keys unprefixed.
        for key, value in self.left_arm.get_observation().items():
            obs_dict[key if key in self._top_level_cam_keys else f"left_{key}"] = value

        # Add "right_" prefix
        for key, value in self.right_arm.get_observation().items():
            obs_dict[f"right_{key}"] = value

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        try:
            sent_action_left = self.left_arm.send_action(left_action, custom_kp, custom_kd)
            sent_action_right = self.right_arm.send_action(right_action, custom_kp, custom_kd)
        except BaseException as error:
            try:
                self._disable_both_arms()
            except BaseException as cleanup_error:
                error.add_note(f"Additional bimanual follower action cleanup error: {cleanup_error!r}")
                logger.exception("Failed to disable both OpenArm followers after action error")
            raise

        # Add prefixes back
        prefixed_sent_action_left = {f"left_{key}": value for key, value in sent_action_left.items()}
        prefixed_sent_action_right = {f"right_{key}": value for key, value in sent_action_right.items()}

        return {**prefixed_sent_action_left, **prefixed_sent_action_right}
