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

import torch
from pathlib import Path

from lerobot.processor import create_transition
from lerobot.processor.hil_processor import (
    AddTeleopActionAsComplimentaryDataStep,
    InterventionActionProcessorStep,
    TELEOP_ACTION_KEY,
    TELEOP_ACTION_NAMES_KEY,
)
from lerobot.teleoperators.bi_so_leader_keyboard import (
    BiSOLeaderKeyboardConfig,
    BiSOLeaderKeyboardKeyboardConfig,
    BiSOLeaderKeyboardLeaderConfig,
)
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import TransitionKey
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


def test_add_teleop_action_step_stores_action_names_for_structured_actions(tmp_path: Path):
    teleop = MockTeleop(
        MockTeleopConfig(
            n_motors=2,
            random_values=False,
            static_values=[1.0, 2.0],
            calibration_dir=tmp_path,
        )
    )
    teleop.connect()
    step = AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop)

    transition = create_transition(complementary_data={})
    updated = step(transition)

    assert updated[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY] == {
        "motor_1.pos": 1.0,
        "motor_2.pos": 2.0,
    }
    assert updated[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_NAMES_KEY] == [
        "motor_1.pos",
        "motor_2.pos",
    ]


def test_intervention_action_processor_supports_named_joint_actions():
    step = InterventionActionProcessorStep()

    transition = create_transition(
        action=torch.tensor([0.0, 0.0]),
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={
            TELEOP_ACTION_KEY: {
                "left_joint.pos": 10.0,
                "right_joint.pos": 20.0,
            },
            TELEOP_ACTION_NAMES_KEY: ["left_joint.pos", "right_joint.pos"],
        },
    )

    updated = step(transition)

    torch.testing.assert_close(updated[TransitionKey.ACTION], torch.tensor([10.0, 20.0]))


def test_bi_so_leader_keyboard_config_registers_with_parser():
    cfg = BiSOLeaderKeyboardConfig(
        id="teleop",
        leader=BiSOLeaderKeyboardLeaderConfig(
            left_arm_config=SOLeaderConfig(port="/dev/null"),
            right_arm_config=SOLeaderConfig(port="/dev/null"),
        ),
        keyboard=BiSOLeaderKeyboardKeyboardConfig(),
    )

    assert cfg.type == "bi_so_leader_keyboard"
