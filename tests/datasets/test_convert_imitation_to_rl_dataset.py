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

from pathlib import Path

import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.convert_imitation_to_rl_dataset import (
    align_visual_feature_shapes_to_runtime_sample,
    convert_imitation_dataset_to_terminal_reward_dataset,
    validate_output_location,
)
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD


def _make_source_dataset(root: Path) -> LeRobotDataset:
    features = {
        ACTION: {"dtype": "float32", "shape": (2,), "names": ["joint_1", "joint_2"]},
        OBS_STATE: {"dtype": "float32", "shape": (2,), "names": ["joint_1", "joint_2"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="dummy/imitation_source",
        fps=10,
        root=root,
        features=features,
        use_videos=False,
    )

    for episode_idx, episode_length in enumerate([5, 4]):
        for frame_idx in range(episode_length):
            dataset.add_frame(
                {
                    "task": f"task_{episode_idx}",
                    ACTION: torch.tensor([frame_idx, frame_idx + 1], dtype=torch.float32),
                    OBS_STATE: torch.tensor([frame_idx + 10, frame_idx + 11], dtype=torch.float32),
                }
            )
        dataset.save_episode()

    dataset.finalize()
    return LeRobotDataset(repo_id="dummy/imitation_source", root=root)


def test_validate_output_location_rejects_same_repo_and_root(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    existing_output_root = tmp_path / "existing_output"
    existing_output_root.mkdir()

    with pytest.raises(ValueError, match="output_repo_id must be different"):
        validate_output_location("dummy/source", "dummy/source", input_root, tmp_path / "output")

    with pytest.raises(ValueError, match="output_root must be different"):
        validate_output_location("dummy/source", "dummy/output", input_root, input_root)

    with pytest.raises(ValueError, match="already exists"):
        validate_output_location("dummy/source", "dummy/output", input_root, existing_output_root)


def test_convert_imitation_dataset_creates_trimmed_terminal_reward_dataset(tmp_path):
    source_dataset = _make_source_dataset(tmp_path / "source")

    converted_root = tmp_path / "converted"
    _, stats = convert_imitation_dataset_to_terminal_reward_dataset(
        source_dataset=source_dataset,
        output_repo_id="dummy/imitation_source_terminal_rl",
        output_root=converted_root,
        trim_tail_frames=2,
        min_remaining_frames=2,
        positive_reward_frames=1,
        image_writer_threads=0,
        image_writer_processes=0,
        encoder_threads=1,
        parallel_video_encoding=False,
    )

    assert stats.source_episodes == 2
    assert stats.kept_episodes == 2
    assert stats.dropped_episodes == 0
    assert stats.source_frames == 9
    assert stats.kept_frames == 5

    converted_dataset = LeRobotDataset(
        repo_id="dummy/imitation_source_terminal_rl",
        root=converted_root,
    )
    assert len(source_dataset) == 9
    assert len(converted_dataset) == 5

    rewards = [converted_dataset[i][REWARD].item() for i in range(len(converted_dataset))]
    dones = [bool(converted_dataset[i][DONE].item()) for i in range(len(converted_dataset))]
    tasks = [converted_dataset[i]["task"] for i in range(len(converted_dataset))]

    assert rewards == [0.0, 0.0, 1.0, 0.0, 1.0]
    assert dones == [False, False, True, False, True]
    assert tasks == ["task_0", "task_0", "task_0", "task_1", "task_1"]


def test_align_visual_feature_shapes_to_runtime_sample_updates_hwc_to_chw():
    class DummyDataset:
        def __getitem__(self, idx):
            return {"observation.images.front": torch.zeros(3, 480, 640)}

    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    }

    aligned = align_visual_feature_shapes_to_runtime_sample(DummyDataset(), features)

    assert aligned["observation.images.front"]["shape"] == (3, 480, 640)
    assert aligned["observation.images.front"]["names"] == ["channels", "height", "width"]
