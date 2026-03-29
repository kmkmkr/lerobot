#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.teleoperators.so_leader import SOLeaderConfig

from ..config import TeleoperatorConfig


@dataclass
class BiSOLeaderKeyboardLeaderConfig:
    """Plain nested config to keep draccus parsing shallow."""

    id: str | None = None
    calibration_dir: Path | None = None
    left_arm_config: SOLeaderConfig | None = None
    right_arm_config: SOLeaderConfig | None = None


@dataclass
class BiSOLeaderKeyboardKeyboardConfig:
    """Keyboard event config for the composite teleoperator."""

    id: str | None = None
    calibration_dir: Path | None = None


@TeleoperatorConfig.register_subclass("bi_so_leader_keyboard")
@dataclass
class BiSOLeaderKeyboardConfig(TeleoperatorConfig):
    """Composite teleoperator for bimanual SO leaders plus keyboard events."""

    leader: BiSOLeaderKeyboardLeaderConfig = field(default_factory=BiSOLeaderKeyboardLeaderConfig)
    keyboard: BiSOLeaderKeyboardKeyboardConfig = field(default_factory=BiSOLeaderKeyboardKeyboardConfig)

    intervention_key: str = " "
    success_key: str = "s"
    failure_key: str = "q"
    rerecord_key: str = "r"
