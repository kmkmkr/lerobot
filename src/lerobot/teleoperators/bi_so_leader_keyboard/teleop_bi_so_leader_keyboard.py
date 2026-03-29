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

from functools import cached_property
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..bi_so_leader import BiSOLeader, BiSOLeaderConfig
from ..keyboard import KeyboardTeleop, KeyboardTeleopConfig
from ..so_leader import SOLeaderTeleopConfig
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_bi_so_leader_keyboard import BiSOLeaderKeyboardConfig


class BiSOLeaderKeyboardTeleop(Teleoperator):
    """Expose bimanual leader joint targets with keyboard-based episode events."""

    config_class = BiSOLeaderKeyboardConfig
    name = "bi_so_leader_keyboard"

    def __init__(self, config: BiSOLeaderKeyboardConfig):
        super().__init__(config)
        self.config = config
        self._intervention_enabled = False
        self._pressed_keys: set[str] = set()

        leader_cfg = BiSOLeaderConfig(
            id=config.leader.id or (f"{config.id}_leader" if config.id is not None else None),
            calibration_dir=config.leader.calibration_dir or config.calibration_dir,
            left_arm_config=SOLeaderTeleopConfig(
                port=config.leader.left_arm_config.port,
                use_degrees=config.leader.left_arm_config.use_degrees,
            ),
            right_arm_config=SOLeaderTeleopConfig(
                port=config.leader.right_arm_config.port,
                use_degrees=config.leader.right_arm_config.use_degrees,
            ),
        )
        keyboard_cfg = KeyboardTeleopConfig(
            id=config.keyboard.id or (f"{config.id}_keyboard" if config.id is not None else None),
            calibration_dir=config.keyboard.calibration_dir or config.calibration_dir,
        )

        self.leader = BiSOLeader(leader_cfg)
        self.keyboard = KeyboardTeleop(keyboard_cfg)

    @cached_property
    def action_names(self) -> list[str]:
        return list(self.leader.action_features.keys())

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (len(self.action_names),),
            "names": self.action_names,
        }

    @cached_property
    def feedback_features(self) -> dict[str, Any]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.leader.is_connected and self.keyboard.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.leader.connect(calibrate=calibrate)
        self.keyboard.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.leader.is_calibrated

    def calibrate(self) -> None:
        self.leader.calibrate()
        self.keyboard.calibrate()

    def configure(self) -> None:
        self.leader.configure()
        self.keyboard.configure()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return self.leader.get_action()

    @check_if_not_connected
    def get_teleop_events(self) -> dict[str, Any]:
        pressed_keys = set(self.keyboard.get_action().keys())
        new_presses = pressed_keys - self._pressed_keys
        self._pressed_keys = pressed_keys

        if self.config.intervention_key in new_presses:
            self._intervention_enabled = not self._intervention_enabled

        success = self.config.success_key in new_presses
        rerecord_episode = self.config.rerecord_key in new_presses
        failure = self.config.failure_key in new_presses

        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_enabled,
            TeleopEvents.TERMINATE_EPISODE: rerecord_episode or failure,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self.leader.send_feedback(feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.keyboard.disconnect()
        self.leader.disconnect()
