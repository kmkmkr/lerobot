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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_keyboard_controls_must_be_distinct():
    from lerobot.rollout import DAggerKeyboardConfig

    with pytest.raises(ValueError, match="distinct keys"):
        DAggerKeyboardConfig(correction="backspace")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.correction_persistence == "background"
    assert cfg.input_device == "keyboard"
    assert cfg.keyboard.discard == "backspace"
    assert cfg.resume_blend_duration_s == 2.0
    assert cfg.max_action_velocity is None
    assert cfg.web_ui.enabled is False
    assert cfg.web_ui.port == 8000
    assert cfg.web_ui.preview_fps == 5.0


def test_dagger_config_rejects_invalid_correction_persistence():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="correction_persistence"):
        DAggerStrategyConfig(correction_persistence="deferred")

    with pytest.raises(ValueError, match="record_autonomous=False"):
        DAggerStrategyConfig(record_autonomous=True, correction_persistence="synchronous")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": -1},
        {"port": 65536},
        {"port": True},
        {"preview_fps": 0.0},
        {"preview_fps": float("nan")},
        {"preview_fps": True},
        {"jpeg_quality": 0},
        {"jpeg_quality": 101},
        {"jpeg_quality": True},
    ],
)
def test_dagger_web_ui_config_rejects_invalid_values(kwargs):
    from lerobot.rollout import DAggerWebUIConfig

    with pytest.raises(ValueError, match="DAgger web_ui"):
        DAggerWebUIConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resume_blend_duration_s": float("nan")},
        {"resume_blend_duration_s": float("inf")},
        {"max_action_velocity": float("nan")},
        {"max_action_velocity": float("inf")},
    ],
)
def test_dagger_safety_limits_must_be_finite(kwargs):
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="must be finite"):
        DAggerStrategyConfig(**kwargs)


def test_dagger_corrections_only_forces_non_streaming_encoding():
    from lerobot.rollout import DAggerStrategyConfig, RolloutConfig

    dataset = MagicMock()
    dataset.repo_id = "local/test"
    dataset.streaming_encoding = True
    dataset.video = True
    dataset.video_encoding_batch_size = 1
    dataset.num_episodes = 1
    dataset.single_task = "task"
    policy = MagicMock()
    policy.device = "cpu"

    RolloutConfig(
        robot=MagicMock(),
        teleop=MagicMock(),
        policy=policy,
        strategy=DAggerStrategyConfig(num_episodes=1),
        dataset=dataset,
        device="cpu",
    )

    assert dataset.streaming_encoding is False
    assert dataset.video_encoding_batch_size == 2


@pytest.mark.parametrize("robot_type", ["openarm_follower", "bi_openarm_follower"])
def test_openarm_continuous_dagger_fails_before_hardware_connection(robot_type):
    from lerobot.rollout import DAggerStrategyConfig, RolloutConfig

    dataset = MagicMock()
    dataset.repo_id = "local/test"
    dataset.streaming_encoding = True
    dataset.num_episodes = 1
    dataset.single_task = "task"
    policy = MagicMock()
    policy.device = "cpu"
    robot = MagicMock()
    robot.type = robot_type

    with pytest.raises(ValueError, match="record_autonomous=False"):
        RolloutConfig(
            robot=robot,
            teleop=MagicMock(),
            policy=policy,
            strategy=DAggerStrategyConfig(num_episodes=1, record_autonomous=True),
            dataset=dataset,
            device="cpu",
        )


def test_inference_config_types():
    from lerobot.rollout import RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


def test_dagger_continuous_feedback_routes_measured_follower_observation():
    from lerobot.rollout.strategies.dagger import _send_continuous_feedback

    observation = {"joint_1.pos": 12.5}
    teleop = MagicMock()
    teleop.requires_continuous_feedback = True

    _send_continuous_feedback(teleop, observation)

    teleop.send_feedback.assert_called_once_with(observation)


def test_dagger_skips_continuous_feedback_for_other_teleoperators():
    from lerobot.rollout.strategies.dagger import _send_continuous_feedback

    teleop = MagicMock()
    teleop.requires_continuous_feedback = False

    _send_continuous_feedback(teleop, {"joint_1.pos": 12.5})

    teleop.send_feedback.assert_not_called()


def test_dagger_openarm_transitions_keep_bilateral_feedback_torque_enabled():
    from lerobot.rollout import DAggerStrategy
    from lerobot.rollout.strategies import DAggerPhase

    teleop = MagicMock()
    teleop.feedback_features = {"joint_1.pos": float}
    teleop.requires_continuous_feedback = True
    ctx = MagicMock()
    ctx.hardware.teleop = teleop
    engine = MagicMock()
    interpolator = MagicMock()
    action = {"joint_1.pos": 10.0}

    with patch("lerobot.rollout.strategies.dagger.teleop_smooth_move_to") as smooth_move:
        DAggerStrategy._apply_transition(
            DAggerPhase.AUTONOMOUS,
            DAggerPhase.PAUSED,
            engine,
            interpolator,
            ctx,
            action,
        )
        DAggerStrategy._apply_transition(
            DAggerPhase.PAUSED,
            DAggerPhase.CORRECTING,
            engine,
            interpolator,
            ctx,
            action,
        )
        DAggerStrategy._apply_transition(
            DAggerPhase.CORRECTING,
            DAggerPhase.PAUSED,
            engine,
            interpolator,
            ctx,
            action,
        )
        DAggerStrategy._apply_transition(
            DAggerPhase.PAUSED,
            DAggerPhase.AUTONOMOUS,
            engine,
            interpolator,
            ctx,
            action,
        )

    smooth_move.assert_not_called()
    teleop.disable_torque.assert_not_called()
    teleop.enable_torque.assert_not_called()
    engine.pause.assert_called_once_with()
    engine.reset.assert_called_once_with()
    engine.resume.assert_not_called()
    interpolator.reset.assert_called_once_with()


def test_dagger_generic_actuated_teleop_keeps_existing_handover_behavior():
    from lerobot.rollout import DAggerStrategy
    from lerobot.rollout.strategies import DAggerPhase

    teleop = MagicMock()
    teleop.feedback_features = {"joint_1.pos": float}
    teleop.requires_continuous_feedback = False
    ctx = MagicMock()
    ctx.hardware.teleop = teleop
    engine = MagicMock()
    interpolator = MagicMock()
    action = {"joint_1.pos": 10.0}

    with patch("lerobot.rollout.strategies.dagger.teleop_smooth_move_to") as smooth_move:
        DAggerStrategy._apply_transition(
            DAggerPhase.AUTONOMOUS,
            DAggerPhase.PAUSED,
            engine,
            interpolator,
            ctx,
            action,
        )
        DAggerStrategy._apply_transition(
            DAggerPhase.PAUSED,
            DAggerPhase.CORRECTING,
            engine,
            interpolator,
            ctx,
            action,
        )
        DAggerStrategy._apply_transition(
            DAggerPhase.CORRECTING,
            DAggerPhase.PAUSED,
            engine,
            interpolator,
            ctx,
            action,
        )
        DAggerStrategy._apply_transition(
            DAggerPhase.PAUSED,
            DAggerPhase.AUTONOMOUS,
            engine,
            interpolator,
            ctx,
            action,
        )

    smooth_move.assert_called_once_with(teleop, action)
    assert teleop.disable_torque.call_count == 2
    teleop.enable_torque.assert_called_once_with()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_dagger_intervention_connects_teleop_before_coordinated_startup():
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.context import _connect_rollout_hardware

    events: list[str] = []
    robot = MagicMock()
    robot.name = "bi_openarm_follower"
    robot.connect.side_effect = lambda: events.append("robot.connect")
    robot.prepare_for_intervention_deployment.side_effect = lambda teleop: events.append(
        "robot.prepare_intervention"
    )
    robot.get_observation.side_effect = lambda: events.append("robot.get_observation") or {"joint_1.pos": 1.0}
    teleop = MagicMock()
    teleop.connect.side_effect = lambda: events.append("teleop.connect")
    cfg = MagicMock()
    cfg.strategy = DAggerStrategyConfig()
    cfg.robot.type = "bi_openarm_follower"
    cfg.teleop.type = "bi_openarm_leader"

    with (
        patch("lerobot.rollout.context.make_robot_from_config", return_value=robot),
        patch("lerobot.rollout.context.make_teleoperator_from_config", return_value=teleop),
    ):
        raw_robot, wrapper, connected_teleop, initial_position = _connect_rollout_hardware(cfg)

    assert events == [
        "robot.connect",
        "teleop.connect",
        "robot.prepare_intervention",
        "robot.get_observation",
    ]
    assert raw_robot is robot
    assert wrapper.inner is robot
    assert connected_teleop is teleop
    assert initial_position == {"joint_1.pos": 1.0}
    robot.prepare_for_intervention_deployment.assert_called_once_with(teleop)
    robot.prepare_for_policy_deployment.assert_not_called()


def test_dagger_intervention_startup_error_disconnects_both_roles():
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.context import _connect_rollout_hardware

    robot = MagicMock()
    robot.name = "bi_openarm_follower"
    robot.is_connected = True
    robot.prepare_for_intervention_deployment.side_effect = RuntimeError("tracking error")
    teleop = MagicMock()
    teleop.is_connected = True
    cfg = MagicMock()
    cfg.strategy = DAggerStrategyConfig()
    cfg.robot.type = "bi_openarm_follower"
    cfg.teleop.type = "bi_openarm_leader"

    with (
        patch("lerobot.rollout.context.make_robot_from_config", return_value=robot),
        patch("lerobot.rollout.context.make_teleoperator_from_config", return_value=teleop),
        pytest.raises(RuntimeError, match="tracking error"),
    ):
        _connect_rollout_hardware(cfg)

    teleop.disconnect.assert_called_once_with()
    robot.disconnect.assert_called_once_with()


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}


def test_teardown_prefers_robot_specific_policy_deployment_return():
    from lerobot.rollout import BaseStrategyConfig, HardwareContext
    from lerobot.rollout.strategies.core import RolloutStrategy

    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    robot = MagicMock()
    robot.is_connected = True
    robot.finish_policy_deployment.return_value = True
    wrapper = MagicMock()
    wrapper.inner = robot
    hardware = HardwareContext(
        robot_wrapper=wrapper,
        teleop=None,
        initial_position={"joint_1.pos": 10.0},
    )

    strategy = TestStrategy(BaseStrategyConfig())
    strategy._teardown_hardware(hardware)

    robot.finish_policy_deployment.assert_called_once_with()
    wrapper.get_observation.assert_not_called()
    robot.disconnect.assert_called_once_with()


def test_dagger_teardown_uses_coordinated_intervention_return():
    from lerobot.rollout import DAggerStrategyConfig, HardwareContext
    from lerobot.rollout.strategies.core import RolloutStrategy

    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    robot = MagicMock()
    robot.is_connected = True
    robot.finish_intervention_deployment.return_value = True
    wrapper = MagicMock()
    wrapper.inner = robot
    teleop = MagicMock()
    teleop.is_connected = True
    hardware = HardwareContext(robot_wrapper=wrapper, teleop=teleop, initial_position={})

    strategy = TestStrategy(DAggerStrategyConfig())
    strategy._teardown_hardware(hardware)

    robot.finish_intervention_deployment.assert_called_once_with(teleop)
    robot.finish_policy_deployment.assert_not_called()
    robot.disconnect.assert_called_once_with()
    teleop.disconnect.assert_called_once_with()


def test_teardown_disconnects_when_robot_specific_return_fails():
    from lerobot.rollout import BaseStrategyConfig, HardwareContext
    from lerobot.rollout.strategies.core import RolloutStrategy

    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    robot = MagicMock()
    robot.is_connected = True
    robot.finish_policy_deployment.side_effect = RuntimeError("tracking error")
    wrapper = MagicMock()
    wrapper.inner = robot
    teleop = MagicMock()
    teleop.is_connected = True
    hardware = HardwareContext(robot_wrapper=wrapper, teleop=teleop, initial_position={})

    strategy = TestStrategy(BaseStrategyConfig())
    with pytest.raises(RuntimeError, match="tracking error"):
        strategy._teardown_hardware(hardware)

    robot.disconnect.assert_called_once_with()
    teleop.disconnect.assert_called_once_with()


def test_context_build_failure_after_connect_secures_and_disconnects_hardware():
    import lerobot.rollout.context as context_module

    robot = MagicMock()
    robot.is_connected = True
    teleop = MagicMock()
    teleop.is_connected = True

    @context_module._cleanup_connected_hardware_on_build_error
    def fail_after_connect():
        context_module._pending_context_hardware.set((robot, teleop))
        raise RuntimeError("dataset setup failed")

    with pytest.raises(RuntimeError, match="dataset setup failed"):
        fail_after_connect()

    robot.secure_intervention_after_fault.assert_called_once_with(teleop)
    teleop.disconnect.assert_called_once_with()
    robot.disconnect.assert_called_once_with()


def test_task_ready_observation_failure_secures_intervention_before_disconnect():
    from types import SimpleNamespace

    import lerobot.rollout.context as context_module
    from lerobot.rollout import DAggerStrategyConfig

    calls: list[str] = []
    robot = MagicMock()
    robot.name = "bi_openarm_follower"
    robot.is_connected = True
    robot.get_observation.side_effect = RuntimeError("initial observation failed")
    robot.secure_intervention_after_fault.side_effect = lambda _teleop: calls.append("secure")
    robot.disconnect.side_effect = lambda: calls.append("robot.disconnect")
    teleop = MagicMock()
    teleop.is_connected = True
    teleop.disconnect.side_effect = lambda: calls.append("teleop.disconnect")
    cfg = SimpleNamespace(
        robot=SimpleNamespace(type="bi_openarm_follower"),
        strategy=DAggerStrategyConfig(num_episodes=1),
        teleop=SimpleNamespace(type="bi_openarm_leader"),
    )

    with (
        patch.object(context_module, "make_robot_from_config", return_value=robot),
        patch.object(context_module, "make_teleoperator_from_config", return_value=teleop),
        pytest.raises(RuntimeError, match="initial observation failed"),
    ):
        context_module._connect_rollout_hardware(cfg)

    robot.prepare_for_intervention_deployment.assert_called_once_with(teleop)
    assert calls == ["secure", "teleop.disconnect", "robot.disconnect"]


def test_context_connect_system_exit_disconnects_a_partially_connected_can_bus():
    from types import SimpleNamespace

    import lerobot.rollout.context as context_module
    from lerobot.rollout import BaseStrategyConfig

    robot = MagicMock()
    robot.name = "partial-robot"
    robot.connect.side_effect = SystemExit("interrupted during camera startup")
    robot.is_connected = False
    robot.bus.is_connected = True
    robot.cameras = {}
    cfg = SimpleNamespace(
        robot=SimpleNamespace(type="partial-robot"),
        strategy=BaseStrategyConfig(),
        teleop=None,
    )

    with (
        patch.object(context_module, "make_robot_from_config", return_value=robot),
        pytest.raises(SystemExit, match="camera startup"),
    ):
        context_module._connect_rollout_hardware(cfg)

    robot.disconnect.assert_called_once_with()


def test_bimanual_disconnect_cleans_live_can_when_full_arm_state_is_false():
    from types import SimpleNamespace

    from lerobot.utils.bimanual import BimanualMixin

    device = object.__new__(BimanualMixin)
    left_disconnect = MagicMock()
    right_disconnect = MagicMock()
    device.left_arm = SimpleNamespace(
        is_connected=False,
        bus=SimpleNamespace(is_connected=True),
        cameras={},
        disconnect=left_disconnect,
    )
    device.right_arm = SimpleNamespace(
        is_connected=False,
        bus=SimpleNamespace(is_connected=False),
        cameras={},
        disconnect=right_disconnect,
    )

    assert device.any_arm_connected
    device.disconnect()

    left_disconnect.assert_called_once_with()
    right_disconnect.assert_not_called()


def test_bimanual_live_connection_check_does_not_swallow_keyboard_interrupt():
    from types import SimpleNamespace

    from lerobot.utils.bimanual import BimanualMixin

    class InterruptingArm:
        bus = SimpleNamespace(is_connected=True)
        cameras = {}

        @property
        def is_connected(self):
            raise KeyboardInterrupt("connection check interrupted")

    with pytest.raises(KeyboardInterrupt, match="connection check interrupted"):
        BimanualMixin._arm_has_live_connection(InterruptingArm())


def test_keyboard_interrupt_with_teardown_failure_surfaces_teardown_error():
    from types import SimpleNamespace

    import lerobot.scripts.lerobot_rollout as rollout_module

    strategy = MagicMock()
    strategy.run.side_effect = KeyboardInterrupt()
    strategy.teardown.side_effect = RuntimeError("hardware teardown failed")
    ctx = SimpleNamespace(hardware=SimpleNamespace(control_fault=None))
    cfg = SimpleNamespace(
        display_data=False,
        strategy=SimpleNamespace(type="dagger"),
        robot=SimpleNamespace(type="bi_openarm_follower"),
        fps=30.0,
        duration=0.0,
    )

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=MagicMock()),
        ),
        patch.object(rollout_module, "create_strategy", return_value=strategy),
        patch.object(rollout_module, "build_rollout_context", return_value=ctx),
        pytest.raises(RuntimeError, match="hardware teardown failed"),
    ):
        rollout_module.rollout.__wrapped__(cfg)

    assert isinstance(ctx.hardware.control_fault, KeyboardInterrupt)


def test_interrupt_during_post_build_logging_still_tears_down_connected_context():
    from types import SimpleNamespace

    import lerobot.scripts.lerobot_rollout as rollout_module

    strategy = MagicMock()
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)
    ctx = SimpleNamespace(hardware=hardware)
    cfg = SimpleNamespace(
        display_data=False,
        strategy=SimpleNamespace(type="dagger"),
        robot=SimpleNamespace(type="bi_openarm_follower"),
        fps=30.0,
        duration=0.0,
    )

    def interrupt_after_context(message, *_args, **_kwargs):
        if message == "Rollout strategy: %s":
            raise KeyboardInterrupt("post-build interrupt")

    def complete_teardown(_ctx):
        hardware.teardown_complete = True

    strategy.teardown.side_effect = complete_teardown

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=MagicMock()),
        ),
        patch.object(rollout_module, "create_strategy", return_value=strategy),
        patch.object(rollout_module, "build_rollout_context", return_value=ctx),
        patch.object(rollout_module.logger, "info", side_effect=interrupt_after_context),
        pytest.raises(KeyboardInterrupt, match="post-build interrupt") as exc_info,
    ):
        rollout_module.rollout.__wrapped__(cfg)

    assert hardware.control_fault is exc_info.value
    strategy.setup.assert_not_called()
    strategy.teardown.assert_called_once_with(ctx)


def test_interrupt_after_context_adoption_uses_handoff_for_teardown():
    from types import SimpleNamespace

    import lerobot.scripts.lerobot_rollout as rollout_module

    strategy = MagicMock()
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)
    ctx = SimpleNamespace(hardware=hardware)
    cfg = SimpleNamespace(
        display_data=False,
        strategy=SimpleNamespace(type="dagger"),
        robot=SimpleNamespace(type="bi_openarm_follower"),
        fps=30.0,
        duration=0.0,
    )

    def build_context(_cfg, _shutdown_event, *, context_ready_callback=None):
        assert context_ready_callback is not None
        context_ready_callback(ctx)
        raise KeyboardInterrupt("context handoff interrupt")

    def complete_teardown(_ctx):
        hardware.teardown_complete = True

    strategy.teardown.side_effect = complete_teardown

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=MagicMock()),
        ),
        patch.object(rollout_module, "create_strategy", return_value=strategy),
        patch.object(rollout_module, "build_rollout_context", side_effect=build_context),
        pytest.raises(KeyboardInterrupt, match="context handoff interrupt") as exc_info,
    ):
        rollout_module.rollout.__wrapped__(cfg)

    assert hardware.control_fault is exc_info.value
    strategy.setup.assert_not_called()
    strategy.teardown.assert_called_once_with(ctx)


def test_interrupt_immediately_before_teardown_call_still_completes_hardware_cleanup():
    from types import SimpleNamespace

    import lerobot.scripts.lerobot_rollout as rollout_module

    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)
    ctx = SimpleNamespace(hardware=hardware)
    interrupt = KeyboardInterrupt("pre-teardown interrupt")

    class InterruptBeforeTeardownStrategy:
        def __init__(self):
            self.teardown_accesses = 0

        def setup(self, _ctx):
            pass

        def run(self, _ctx):
            pass

        @property
        def teardown(self):
            self.teardown_accesses += 1
            if self.teardown_accesses == 1:
                raise interrupt

            def complete_teardown(_ctx):
                hardware.teardown_complete = True

            return complete_teardown

    strategy = InterruptBeforeTeardownStrategy()
    cfg = SimpleNamespace(
        display_data=False,
        strategy=SimpleNamespace(type="dagger"),
        robot=SimpleNamespace(type="bi_openarm_follower"),
        fps=30.0,
        duration=0.0,
    )

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=MagicMock()),
        ),
        patch.object(rollout_module, "create_strategy", return_value=strategy),
        patch.object(rollout_module, "build_rollout_context", return_value=ctx),
        pytest.raises(KeyboardInterrupt, match="pre-teardown interrupt") as exc_info,
    ):
        rollout_module.rollout.__wrapped__(cfg)

    assert exc_info.value is interrupt
    assert hardware.control_fault is interrupt
    assert hardware.teardown_complete
    assert strategy.teardown_accesses == 2


def test_interrupt_during_teardown_retries_in_fault_mode_until_hardware_is_complete():
    from types import SimpleNamespace

    import lerobot.scripts.lerobot_rollout as rollout_module

    strategy = MagicMock()
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)
    ctx = SimpleNamespace(hardware=hardware)
    cfg = SimpleNamespace(
        display_data=False,
        strategy=SimpleNamespace(type="dagger"),
        robot=SimpleNamespace(type="bi_openarm_follower"),
        fps=30.0,
        duration=0.0,
    )
    interrupt = KeyboardInterrupt("teardown interrupt")

    def teardown_side_effect(_ctx):
        if strategy.teardown.call_count == 1:
            raise interrupt
        hardware.teardown_complete = True

    strategy.teardown.side_effect = teardown_side_effect

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=MagicMock()),
        ),
        patch.object(rollout_module, "create_strategy", return_value=strategy),
        patch.object(rollout_module, "build_rollout_context", return_value=ctx),
        pytest.raises(KeyboardInterrupt, match="teardown interrupt") as exc_info,
    ):
        rollout_module.rollout.__wrapped__(cfg)

    assert exc_info.value is interrupt
    assert hardware.control_fault is interrupt
    assert hardware.teardown_complete
    assert strategy.teardown.call_count == 2
