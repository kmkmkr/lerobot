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

from __future__ import annotations

import threading

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout.async_dataset import AsyncEpisodeSaveError, AsyncEpisodeSaver


class _BlockingDataset:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.save_started = threading.Event()
        self.release_save = threading.Event()
        self.save_finished = threading.Event()
        self.save_thread_names: list[str] = []
        self.parallel_encoding_values: list[bool] = []
        self.save_error: Exception | None = None

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def has_pending_frames(self) -> bool:
        return bool(self.frames)

    def save_episode(self, episode_data=None, parallel_encoding: bool = True) -> None:
        self.save_thread_names.append(threading.current_thread().name)
        self.parallel_encoding_values.append(parallel_encoding)
        self.save_started.set()
        if not self.release_save.wait(timeout=5):
            raise TimeoutError("test did not release the save worker")
        if self.save_error is not None:
            raise self.save_error
        self.frames.clear()
        self.save_finished.set()


def test_save_runs_on_background_worker_and_defaults_to_sequential_encoding() -> None:
    dataset = _BlockingDataset()
    saver = AsyncEpisodeSaver(dataset, thread_name_prefix="test-episode-save")
    saver.add_frame({"frame": 0})

    future = saver.submit_save_episode()

    assert dataset.save_started.wait(timeout=1)
    assert not future.done()
    assert dataset.save_thread_names == ["test-episode-save_0"]
    assert dataset.parallel_encoding_values == [False]

    dataset.release_save.set()
    assert saver.wait_for_pending_save()
    assert dataset.save_finished.is_set()
    saver.shutdown()


def test_frames_are_rejected_until_finished_save_is_explicitly_collected() -> None:
    dataset = _BlockingDataset()
    saver = AsyncEpisodeSaver(dataset)
    saver.add_frame({"frame": 0})
    future = saver.submit_save_episode()
    dataset.release_save.set()
    future.result(timeout=1)

    assert saver.save_pending
    assert not saver.save_in_progress
    with pytest.raises(RuntimeError, match="wait_for_pending_save"):
        saver.add_frame({"frame": 1})

    assert saver.wait_for_pending_save()
    saver.add_frame({"frame": 1})
    assert dataset.frames == [{"frame": 1}]
    dataset.release_save.set()
    saver.shutdown()


def test_only_one_uncollected_save_can_be_submitted() -> None:
    dataset = _BlockingDataset()
    saver = AsyncEpisodeSaver(dataset)
    saver.add_frame({"frame": 0})
    saver.submit_save_episode()

    with pytest.raises(RuntimeError, match="already pending"):
        saver.submit_save_episode()

    dataset.release_save.set()
    saver.wait_for_pending_save()
    saver.shutdown()


def test_empty_episode_is_rejected_before_worker_submission() -> None:
    dataset = _BlockingDataset()
    saver = AsyncEpisodeSaver(dataset)

    with pytest.raises(RuntimeError, match="no pending frames"):
        saver.submit_save_episode()

    assert not dataset.save_started.is_set()
    saver.shutdown()


def test_save_exception_is_collected_and_permanently_blocks_recording() -> None:
    dataset = _BlockingDataset()
    dataset.save_error = ValueError("encoding failed")
    saver = AsyncEpisodeSaver(dataset)
    saver.add_frame({"frame": 0})
    saver.submit_save_episode()
    dataset.release_save.set()

    with pytest.raises(ValueError, match="encoding failed"):
        saver.wait_for_pending_save()

    with pytest.raises(AsyncEpisodeSaveError, match="cannot safely record"):
        saver.add_frame({"frame": 1})
    with pytest.raises(AsyncEpisodeSaveError, match="cannot safely continue"):
        saver.wait_for_pending_save()
    with pytest.raises(AsyncEpisodeSaveError, match="failed"):
        saver.shutdown()


def test_shutdown_waits_for_pending_save() -> None:
    dataset = _BlockingDataset()
    saver = AsyncEpisodeSaver(dataset)
    saver.add_frame({"frame": 0})
    saver.submit_save_episode(parallel_encoding=True)

    release_thread = threading.Thread(target=dataset.release_save.set)
    release_thread.start()
    saver.shutdown()
    release_thread.join(timeout=1)

    assert dataset.save_finished.is_set()
    assert dataset.parallel_encoding_values == [True]
    saver.shutdown()
