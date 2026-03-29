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

import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import DONE, REWARD


@dataclass
class ConversionStats:
    source_episodes: int = 0
    kept_episodes: int = 0
    dropped_episodes: int = 0
    source_frames: int = 0
    kept_frames: int = 0
    trimmed_frames: int = 0


def build_terminal_reward_features(source_features: dict) -> dict:
    features = deepcopy(source_features)
    features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}
    features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}
    return features


def align_visual_feature_shapes_to_runtime_sample(source_dataset: LeRobotDataset, features: dict) -> dict:
    """Match visual feature metadata to the actual tensor layout returned by __getitem__.

    Some datasets store image/video shapes in metadata as HWC while runtime samples are
    returned as CHW tensors. When we recreate a dataset, validation happens against the
    runtime sample tensors, so we need the feature shapes to match those tensors.
    """
    aligned_features = deepcopy(features)
    sample = source_dataset[0]

    for key, feature in aligned_features.items():
        if feature["dtype"] not in {"image", "video"} or key not in sample:
            continue

        value = sample[key]
        if isinstance(value, torch.Tensor):
            feature["shape"] = tuple(value.shape)
            feature["names"] = ["channels", "height", "width"]

    return aligned_features


def validate_output_location(
    input_repo_id: str,
    output_repo_id: str,
    input_root: str | Path | None,
    output_root: str | Path,
) -> Path:
    if input_repo_id == output_repo_id:
        raise ValueError("output_repo_id must be different from input_repo_id to avoid overwriting the source repo.")

    output_root_path = Path(output_root).expanduser()
    if input_root is not None:
        input_root_path = Path(input_root).expanduser().resolve()
        if output_root_path.resolve() == input_root_path:
            raise ValueError("output_root must be different from input_root to avoid overwriting the source data.")

    if output_root_path.exists():
        raise ValueError(
            f"output_root '{output_root_path}' already exists. Please provide a new destination path."
        )

    output_root_path.parent.mkdir(parents=True, exist_ok=True)
    return output_root_path


def build_output_frame(source_sample: dict, source_feature_keys: set[str], reward: float, done: bool) -> dict:
    frame = {
        "task": source_sample["task"],
        REWARD: torch.tensor([reward], dtype=torch.float32),
        DONE: torch.tensor([done], dtype=torch.bool),
    }

    for key in source_feature_keys:
        if key in DEFAULT_FEATURES or key in {REWARD, DONE}:
            continue

        value = source_sample[key]
        if isinstance(value, torch.Tensor):
            frame[key] = value.cpu()
        else:
            frame[key] = value

    return frame


def convert_episode(
    source_episode: list[dict],
    target_dataset: LeRobotDataset,
    source_feature_keys: set[str],
    trim_tail_frames: int,
    min_remaining_frames: int,
    positive_reward_frames: int,
    parallel_video_encoding: bool,
) -> tuple[bool, int, int]:
    source_length = len(source_episode)
    trimmed_length = max(source_length - trim_tail_frames, 0)

    if trimmed_length < min_remaining_frames:
        return False, source_length, 0

    first_positive_idx = max(trimmed_length - positive_reward_frames, 0)
    for idx, sample in enumerate(source_episode[:trimmed_length]):
        reward = 1.0 if idx >= first_positive_idx else 0.0
        done = idx == trimmed_length - 1
        frame = build_output_frame(
            source_sample=sample,
            source_feature_keys=source_feature_keys,
            reward=reward,
            done=done,
        )
        target_dataset.add_frame(frame)

    target_dataset.save_episode(parallel_encoding=parallel_video_encoding)
    return True, source_length, trimmed_length


def convert_imitation_dataset_to_terminal_reward_dataset(
    source_dataset: LeRobotDataset,
    output_repo_id: str,
    output_root: str | Path,
    trim_tail_frames: int,
    min_remaining_frames: int = 1,
    positive_reward_frames: int = 1,
    image_writer_threads: int = 1,
    image_writer_processes: int = 0,
    vcodec: str = "h264",
    encoder_threads: int = 1,
    batch_encoding_size: int = 1,
    streaming_encoding: bool = False,
    parallel_video_encoding: bool = False,
) -> tuple[LeRobotDataset, ConversionStats]:
    if trim_tail_frames < 0:
        raise ValueError("trim_tail_frames must be >= 0.")
    if min_remaining_frames <= 0:
        raise ValueError("min_remaining_frames must be > 0.")
    if positive_reward_frames <= 0:
        raise ValueError("positive_reward_frames must be > 0.")

    source_features = build_terminal_reward_features(source_dataset.meta.info["features"])
    source_features = align_visual_feature_shapes_to_runtime_sample(source_dataset, source_features)
    target_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=source_dataset.fps,
        root=output_root,
        robot_type=source_dataset.meta.robot_type,
        features=source_features,
        use_videos=len(source_dataset.meta.video_keys) > 0,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        vcodec=vcodec,
        encoder_threads=encoder_threads,
        batch_encoding_size=batch_encoding_size,
        streaming_encoding=streaming_encoding,
    )

    stats = ConversionStats()
    source_feature_keys = set(source_dataset.features)

    current_episode_idx = None
    current_episode_samples: list[dict] = []

    try:
        for frame_idx in tqdm(range(len(source_dataset)), desc="Converting frames"):
            sample = source_dataset[frame_idx]
            episode_idx = int(sample["episode_index"].item())

            if current_episode_idx is None:
                current_episode_idx = episode_idx

            if episode_idx != current_episode_idx:
                kept, source_length, kept_length = convert_episode(
                    source_episode=current_episode_samples,
                    target_dataset=target_dataset,
                    source_feature_keys=source_feature_keys,
                    trim_tail_frames=trim_tail_frames,
                    min_remaining_frames=min_remaining_frames,
                    positive_reward_frames=positive_reward_frames,
                    parallel_video_encoding=parallel_video_encoding,
                )
                stats.source_episodes += 1
                stats.source_frames += source_length
                stats.trimmed_frames += min(trim_tail_frames, source_length)
                if kept:
                    stats.kept_episodes += 1
                    stats.kept_frames += kept_length
                else:
                    stats.dropped_episodes += 1

                current_episode_samples = []
                current_episode_idx = episode_idx

            current_episode_samples.append(sample)

        if current_episode_samples:
            kept, source_length, kept_length = convert_episode(
                source_episode=current_episode_samples,
                target_dataset=target_dataset,
                source_feature_keys=source_feature_keys,
                trim_tail_frames=trim_tail_frames,
                min_remaining_frames=min_remaining_frames,
                positive_reward_frames=positive_reward_frames,
                parallel_video_encoding=parallel_video_encoding,
            )
            stats.source_episodes += 1
            stats.source_frames += source_length
            stats.trimmed_frames += min(trim_tail_frames, source_length)
            if kept:
                stats.kept_episodes += 1
                stats.kept_frames += kept_length
            else:
                stats.dropped_episodes += 1

        target_dataset.finalize()
    except Exception:
        target_dataset.finalize()
        raise

    return target_dataset, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a new RL-style LeRobot dataset from an imitation dataset with terminal rewards."
    )
    parser.add_argument("--input-repo-id", type=str, required=True, help="Source dataset repo id.")
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Optional local root for the source dataset. Use this for already-downloaded snapshots.",
    )
    parser.add_argument("--output-repo-id", type=str, required=True, help="Destination dataset repo id.")
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Destination root for the converted dataset. Must be a new empty directory.",
    )
    parser.add_argument(
        "--trim-tail-frames",
        type=int,
        required=True,
        help="Number of frames to drop from the end of each episode before labeling terminal reward.",
    )
    parser.add_argument(
        "--min-remaining-frames",
        type=int,
        default=1,
        help="Drop episodes that have fewer than this many frames left after trimming.",
    )
    parser.add_argument(
        "--positive-reward-frames",
        type=int,
        default=1,
        help="Number of frames at the end of the trimmed episode to label with reward=1.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=1,
        help="Number of async image writer threads. Lower values reduce CPU usage.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Number of async image writer processes. Keep this at 0 to avoid extra CPU contention.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="Video codec for the output dataset. h264 is usually lighter on CPU than libsvtav1.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=1,
        help="Threads per video encoder instance. Lower values reduce CPU usage.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=1,
        help="Number of episodes to accumulate before batch video encoding.",
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Encode videos during frame writing instead of at episode end.",
    )
    parser.add_argument(
        "--parallel-video-encoding",
        action="store_true",
        help="Encode different camera videos in parallel. Disabled by default to keep CPU usage lower.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted dataset to the output repo id after local conversion completes.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the output dataset repo as private when pushing to the Hub.",
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use upload_large_folder when pushing to the Hub.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    output_root = validate_output_location(
        input_repo_id=args.input_repo_id,
        output_repo_id=args.output_repo_id,
        input_root=args.input_root,
        output_root=args.output_root,
    )

    source_dataset = LeRobotDataset(
        repo_id=args.input_repo_id,
        root=args.input_root,
    )
    converted_dataset, stats = convert_imitation_dataset_to_terminal_reward_dataset(
        source_dataset=source_dataset,
        output_repo_id=args.output_repo_id,
        output_root=output_root,
        trim_tail_frames=args.trim_tail_frames,
        min_remaining_frames=args.min_remaining_frames,
        positive_reward_frames=args.positive_reward_frames,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        batch_encoding_size=args.batch_encoding_size,
        streaming_encoding=args.streaming_encoding,
        parallel_video_encoding=args.parallel_video_encoding,
    )

    logging.info("Source episodes: %s", stats.source_episodes)
    logging.info("Kept episodes: %s", stats.kept_episodes)
    logging.info("Dropped episodes: %s", stats.dropped_episodes)
    logging.info("Source frames: %s", stats.source_frames)
    logging.info("Kept frames: %s", stats.kept_frames)
    logging.info("Trimmed frames: %s", stats.trimmed_frames)
    logging.info("Converted dataset saved to %s", converted_dataset.root)

    if args.push_to_hub:
        logging.info("Pushing converted dataset to Hub repo %s", args.output_repo_id)
        converted_dataset.push_to_hub(private=args.private, upload_large_folder=args.upload_large_folder)


if __name__ == "__main__":
    main()
