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
from functools import cached_property

from lerobot.types import RobotAction
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..openarm_leader import OpenArmLeader, OpenArmLeaderConfig
from ..teleoperator import Teleoperator
from .config_bi_openarm_leader import BiOpenArmLeaderConfig

logger = logging.getLogger(__name__)


class BiOpenArmLeader(BimanualMixin, Teleoperator):
    """
    Bimanual OpenArm Leader Arms
    """

    config_class = BiOpenArmLeaderConfig
    name = "bi_openarm_leader"

    def __init__(self, config: BiOpenArmLeaderConfig):
        super().__init__(config)
        self.config = config

        if config.left_arm_config.side not in (None, "left"):
            raise ValueError("left_arm_config.side must be 'left' when provided")
        if config.right_arm_config.side not in (None, "right"):
            raise ValueError("right_arm_config.side must be 'right' when provided")
        if config.left_arm_config.manual_control != config.right_arm_config.manual_control:
            raise ValueError("left and right OpenArm leaders must use the same manual_control mode")

        left_arm_config = OpenArmLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            side="left",
            coordinate_frame=config.left_arm_config.coordinate_frame,
            can_interface=config.left_arm_config.can_interface,
            use_can_fd=config.left_arm_config.use_can_fd,
            can_bitrate=config.left_arm_config.can_bitrate,
            can_data_bitrate=config.left_arm_config.can_data_bitrate,
            motor_config=config.left_arm_config.motor_config,
            manual_control=config.left_arm_config.manual_control,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            use_velocity_and_torque=config.left_arm_config.use_velocity_and_torque,
            position_kd=config.left_arm_config.position_kd,
            position_kp=config.left_arm_config.position_kp,
            feedback_position_limit_tolerance_deg=(
                config.left_arm_config.feedback_position_limit_tolerance_deg
            ),
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
        )

        right_arm_config = OpenArmLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            side="right",
            coordinate_frame=config.right_arm_config.coordinate_frame,
            can_interface=config.right_arm_config.can_interface,
            use_can_fd=config.right_arm_config.use_can_fd,
            can_bitrate=config.right_arm_config.can_bitrate,
            can_data_bitrate=config.right_arm_config.can_data_bitrate,
            motor_config=config.right_arm_config.motor_config,
            manual_control=config.right_arm_config.manual_control,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            use_velocity_and_torque=config.right_arm_config.use_velocity_and_torque,
            position_kd=config.right_arm_config.position_kd,
            position_kp=config.right_arm_config.position_kp,
            feedback_position_limit_tolerance_deg=(
                config.right_arm_config.feedback_position_limit_tolerance_deg
            ),
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
        )

        self.left_arm = OpenArmLeader(left_arm_config)
        self.right_arm = OpenArmLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.feedback_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.feedback_features.items()},
        }

    @property
    def requires_continuous_feedback(self) -> bool:
        return self.left_arm.requires_continuous_feedback and self.right_arm.requires_continuous_feedback

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Preflight both dynamics chains before enabling either leader."""
        self.left_arm._prepare_gravity_model()
        self.right_arm._prepare_gravity_model()
        try:
            self.left_arm.connect(calibrate)
            self.right_arm.connect(calibrate)
        except BaseException as error:
            cleanup_errors: list[BaseException] = []
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                if not self._arm_has_live_connection(arm):
                    continue
                try:
                    arm.disconnect()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                    logger.exception("Failed to disconnect %s OpenArm leader after connect error", side)
            for cleanup_error in cleanup_errors:
                error.add_note(f"Additional bimanual leader connection cleanup error: {cleanup_error!r}")
            raise

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        if not self.requires_continuous_feedback:
            raise RuntimeError("send_feedback is unavailable in manual_control free-motion mode")
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }
        try:
            # Prepare both sides before either CAN bus receives a new target.
            # This prevents a right-side validation fault from leaving only the
            # left leader operating on the next feedback sample.
            left_commands = self.left_arm._prepare_bilateral_feedback(left_feedback)
            right_commands = self.right_arm._prepare_bilateral_feedback(right_feedback)
            self.left_arm._send_prepared_bilateral_feedback(left_commands)
            self.right_arm._send_prepared_bilateral_feedback(right_commands)
        except BaseException as error:
            try:
                self.disable_torque(require_response=True)
            except BaseException as disable_error:
                error.add_note(f"Failed to verify bilateral leader torque disable: {disable_error!r}")
                logger.exception("Failed to verify both OpenArm leader torque disables")
            raise

    def _set_disable_torque_on_disconnect(self, disable_torque: bool) -> None:
        for source_config, arm in (
            (self.config.left_arm_config, self.left_arm),
            (self.config.right_arm_config, self.right_arm),
        ):
            source_config.disable_torque_on_disconnect = disable_torque
            arm.config.disable_torque_on_disconnect = disable_torque

    def _disable_connected_arms(
        self, *, require_response: bool = False
    ) -> tuple[dict[str, str], list[tuple[str, BaseException]]]:
        statuses: dict[str, str] = {}
        failures: list[tuple[str, BaseException]] = []
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            if not arm.bus.is_connected:
                statuses[side] = "CAN disconnected; torque state unknown"
                continue
            try:
                if require_response:
                    arm.disable_torque(require_response=True)
                else:
                    arm.disable_torque()
            except BaseException as error:
                statuses[side] = f"torque-disable failed: {error}"
                failures.append((side, error))
                logger.exception("Failed to disable %s OpenArm leader torque", side)
            else:
                statuses[side] = (
                    "torque-disable response verified"
                    if require_response
                    else "torque-disable command completed"
                )
        return statuses, failures

    def secure_current_position_hold(self) -> bool:
        """Establish and verify a measured-position hold on both leaders.

        Safe references are prepared for both sides and preloaded before torque
        is enabled. Only after both enable calls and both post-enable hold
        commands complete do we preserve torque across CAN disconnect.
        """
        arms = (
            ("left", self.config.left_arm_config, self.left_arm),
            ("right", self.config.right_arm_config, self.right_arm),
        )
        self._set_disable_torque_on_disconnect(True)
        prepared: dict[str, dict[str, tuple[float, float, float, float, float]]] = {}
        try:
            for side, _source_config, arm in arms:
                if not arm.bus.is_connected:
                    raise RuntimeError(f"{side} leader CAN bus is closed")
                action = arm.get_action(require_response=True)
                feedback = {key: float(value) for key, value in action.items() if key.endswith(".pos")}
                prepared[side] = arm._prepare_bilateral_feedback(feedback)

            # Preload safe measured-position references before re-enabling a
            # leader that a feedback fault may already have disabled.
            for side, _source_config, arm in arms:
                arm._send_prepared_bilateral_feedback(prepared[side], require_response=True)
            for _side, _source_config, arm in arms:
                arm.enable_torque(require_response=True)
            for side, _source_config, arm in arms:
                arm._send_prepared_bilateral_feedback(prepared[side], require_response=True)
        except BaseException as error:
            statuses, disable_failures = self._disable_connected_arms(require_response=True)
            self._set_disable_torque_on_disconnect(True)
            logger.critical(
                "OpenArm leader current-position hold was not established after fault: %s. "
                "Fail-safe status: left=%s; right=%s. Support all arms before removing power.",
                error,
                statuses["left"],
                statuses["right"],
            )
            for side, disable_error in disable_failures:
                error.add_note(f"Additional {side} leader disable error: {disable_error!r}")
            if not isinstance(error, Exception):
                raise
            for side, disable_error in disable_failures:
                if not isinstance(disable_error, Exception):
                    disable_error.add_note(
                        f"Leader hold failed before the {side} disable interruption: {error!r}"
                    )
                    raise disable_error from error
            return False

        self._set_disable_torque_on_disconnect(False)
        logger.critical(
            "OpenArm leaders are held at their measured positions: both current-position commands "
            "and both torque-enable operations completed. Torque will be preserved across CAN "
            "disconnect; support all arms before removing power or manually disabling torque."
        )
        return True

    def hold_position_after_shutdown_error(self) -> bool:
        """Compatibility entry point for shutdown and fault teardown."""
        return self.secure_current_position_hold()

    def disable_torque(self, *, require_response: bool = False) -> None:
        statuses, disable_failures = self._disable_connected_arms(require_response=require_response)
        signal_failures = [
            (side, error) for side, error in disable_failures if not isinstance(error, Exception)
        ]
        if signal_failures:
            first_side, first_error = signal_failures[0]
            for side, error in disable_failures:
                if error is not first_error:
                    first_error.add_note(f"Additional {side} leader disable error: {error!r}")
            first_error.add_note(f"First leader disable interruption occurred on {first_side}")
            raise first_error
        if not require_response:
            return

        failures = {
            side: status for side, status in statuses.items() if status != "torque-disable response verified"
        }
        if failures:
            details = ", ".join(f"{side}={status}" for side, status in failures.items())
            error = RuntimeError(f"Failed to verify bilateral OpenArm leader torque disable: {details}")
            for side, status in failures.items():
                error.add_note(f"{side}: {status}")
            raise error

    @check_if_not_connected
    def enable_torque(self, *, require_response: bool = False) -> None:
        try:
            if require_response:
                self.left_arm.enable_torque(require_response=True)
                self.right_arm.enable_torque(require_response=True)
            else:
                self.left_arm.enable_torque()
                self.right_arm.enable_torque()
        except BaseException as error:
            try:
                self.disable_torque(require_response=require_response)
            except BaseException as disable_error:
                error.add_note(f"Failed to disable both leaders after enable error: {disable_error!r}")
            raise
