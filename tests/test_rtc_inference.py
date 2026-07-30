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

"""Lifecycle and concurrency tests for the asynchronous RTC inference engine."""

import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

import lerobot.rollout.inference.rtc as rtc_module
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference.rtc import RTCInferenceEngine


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _make_engine(monkeypatch, predict_action_chunk, *, fps: float = 30.0):
    monkeypatch.setattr(
        rtc_module,
        "build_dataset_frame",
        lambda _features, obs, prefix: dict(obs),
    )
    monkeypatch.setattr(
        rtc_module,
        "prepare_observation_for_inference",
        lambda obs, _device, _task, _robot_type: obs,
    )

    policy = MagicMock()
    policy.config = MagicMock(action_feature_names=None)
    policy.predict_action_chunk.side_effect = predict_action_chunk

    preprocessor = MagicMock()
    preprocessor.steps = []
    preprocessor.side_effect = lambda batch: batch

    postprocessor = MagicMock()
    postprocessor.steps = []
    postprocessor.side_effect = lambda actions: actions

    robot = MagicMock()
    robot.robot_type = "test_robot"
    robot.action_features = {}

    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot,
        rtc_config=RTCConfig(enabled=True, execution_horizon=10),
        hw_features={},
        task="test task",
        fps=fps,
        device="cpu",
        rtc_queue_threshold=30,
    )
    return engine, policy


def test_resume_waits_for_fresh_observation_and_resets_first_chunk_delay(monkeypatch):
    calls = []
    second_call_started = threading.Event()

    def predict_action_chunk(batch, *, inference_delay, prev_chunk_left_over):
        calls.append((batch["state"].clone(), inference_delay, prev_chunk_left_over))
        if len(calls) == 2:
            second_call_started.set()
        time.sleep(0.02)
        offset = 100.0 * (len(calls) - 1)
        return (torch.arange(40, dtype=torch.float32) + offset).reshape(1, 40, 1)

    engine, _policy = _make_engine(monkeypatch, predict_action_chunk, fps=1000.0)
    engine.start()
    try:
        engine.notify_observation({"state": torch.tensor([1.0])})
        engine.resume()
        assert _wait_until(lambda: engine.action_queue is not None and engine.action_queue.qsize() == 40)
        assert calls[0][1] == 0

        engine.pause()
        engine.resume()

        # resume must not reuse the observation held before pause.
        assert not second_call_started.wait(0.05)
        assert len(calls) == 1

        engine.notify_observation({"state": torch.tensor([2.0])})
        assert second_call_started.wait(1.0)
        assert _wait_until(lambda: engine.action_queue is not None and engine.action_queue.qsize() == 40)

        torch.testing.assert_close(calls[1][0], torch.tensor([2.0]))
        # The previous generation's measured latency cannot skip actions in
        # the first chunk after resume.
        assert calls[1][1] == 0
        assert calls[1][2] is None
        torch.testing.assert_close(engine.get_action(None), torch.tensor([100.0]))
    finally:
        engine.stop()


@pytest.mark.parametrize("lifecycle_method", ["pause", "reset"])
def test_in_flight_result_is_discarded_across_lifecycle_generation(monkeypatch, lifecycle_method):
    inference_started = threading.Event()
    release_inference = threading.Event()

    def predict_action_chunk(_batch, *, inference_delay, prev_chunk_left_over):
        inference_started.set()
        assert release_inference.wait(1.0)
        return torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)

    engine, _policy = _make_engine(monkeypatch, predict_action_chunk)
    engine.start()
    reset_thread = None
    try:
        engine.notify_observation({"state": torch.tensor([1.0])})
        engine.resume()
        assert inference_started.wait(1.0)

        generation_before = engine._generation
        if lifecycle_method == "pause":
            engine.pause()
        else:
            reset_thread = threading.Thread(target=engine.reset)
            reset_thread.start()
            assert _wait_until(lambda: engine._generation > generation_before)

        assert engine._generation > generation_before
        release_inference.set()

        if reset_thread is not None:
            reset_thread.join(timeout=1.0)
            assert not reset_thread.is_alive()

        assert _wait_until(lambda: engine.action_queue is not None)
        assert engine.action_queue.qsize() == 0
        assert engine.get_action(None) is None
    finally:
        release_inference.set()
        if reset_thread is not None:
            reset_thread.join(timeout=1.0)
        engine.stop()
