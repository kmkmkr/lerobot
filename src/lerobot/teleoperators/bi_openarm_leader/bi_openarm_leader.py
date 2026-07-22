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
        except Exception:
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                if not arm.bus.is_connected:
                    continue
                try:
                    arm.disconnect()
                except Exception as error:
                    logger.error(
                        "Failed to disconnect %s OpenArm leader after connect error: %s", side, error
                    )
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
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }
        try:
            self.left_arm.send_feedback(left_feedback)
            self.right_arm.send_feedback(right_feedback)
        except Exception:
            self.disable_torque()
            raise

    def hold_position_after_shutdown_error(self) -> None:
        """Keep both leaders energized at their measured pose after a failed return.

        The coordinated OpenArm shutdown path uses the same fail-safe as the
        followers: refresh a current-position command, then close CAN without
        disabling torque so none of the four arms free-falls unexpectedly.
        """
        for side, source_config, arm in (
            ("left", self.config.left_arm_config, self.left_arm),
            ("right", self.config.right_arm_config, self.right_arm),
        ):
            source_config.disable_torque_on_disconnect = False
            arm.config.disable_torque_on_disconnect = False
            if not arm.bus.is_connected:
                logger.error("Cannot refresh the %s OpenArm leader hold because its CAN bus is closed", side)
                continue
            try:
                action = arm.get_action()
                feedback = {key: float(value) for key, value in action.items() if key.endswith(".pos")}
                arm.send_feedback(feedback)
            except Exception:
                logger.exception("Failed to refresh the %s OpenArm leader hold command", side)

        logger.critical(
            "OpenArm coordinated shutdown return failed. Both leaders will remain torque-enabled "
            "at their measured pose when CAN is disconnected. Support all arms before removing "
            "power or manually disabling torque."
        )

    @check_if_not_connected
    def disable_torque(self) -> None:
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            try:
                arm.disable_torque()
            except Exception as error:
                logger.error("Failed to disable %s OpenArm leader torque: %s", side, error)

    @check_if_not_connected
    def enable_torque(self) -> None:
        try:
            self.left_arm.enable_torque()
            self.right_arm.enable_torque()
        except Exception:
            self.disable_torque()
            raise
