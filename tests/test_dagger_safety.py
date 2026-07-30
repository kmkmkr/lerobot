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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout import DAggerStrategyConfig, HardwareContext
from lerobot.rollout.strategies.core import RolloutStrategy
from lerobot.rollout.strategies.dagger import (
    DAGGER_INTERVENTION_PHASE_HOOK_VERSION,
    DAggerEvents,
    DAggerPhase,
    DAggerStrategy,
    PolicyActionRateLimiter,
    _init_dagger_keyboard,
    _pause_dagger_control_before_loop_exit,
)


def test_dagger_keyboard_prefers_terminal_and_routes_space_and_tab() -> None:
    events = DAggerEvents()
    listener = object()
    captured: dict[str, object] = {}

    def create_listener(dispatch, **kwargs):
        captured["dispatch"] = dispatch
        captured.update(kwargs)
        return listener

    with patch("lerobot.rollout.strategies.dagger.create_key_listener", side_effect=create_listener):
        result = _init_dagger_keyboard(events, DAggerStrategyConfig().keyboard)

    assert result is listener
    assert captured["prefer_terminal"] is True
    dispatch = captured["dispatch"]
    assert callable(dispatch)

    dispatch("space")
    assert events.consume_transition() == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)
    dispatch("tab")
    assert events.consume_transition() == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)


def test_corrections_only_dead_keyboard_listener_faults_before_robot_commands() -> None:
    strategy = DAggerStrategy(DAggerStrategyConfig(num_episodes=1))
    strategy._listener = object()
    engine = MagicMock()
    strategy._engine = engine
    interpolator = MagicMock()
    interpolator.get_control_interval.return_value = 1 / 30
    strategy._interpolator = interpolator
    strategy._episode_saver = MagicMock()
    strategy._events = DAggerEvents()
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = False
    robot = MagicMock()
    hardware = SimpleNamespace(
        robot_wrapper=robot,
        teleop=MagicMock(),
        control_fault=None,
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                fps=30,
                interpolation_multiplier=1,
                dataset=SimpleNamespace(single_task="task"),
                task="task",
                play_sounds=False,
                duration=0,
                use_torch_compile=False,
            ),
            shutdown_event=shutdown_event,
        ),
        hardware=hardware,
        data=SimpleNamespace(
            dataset=MagicMock(),
            dataset_features={},
            ordered_action_keys=[],
        ),
    )

    with (
        patch("lerobot.rollout.strategies.dagger.key_listener_is_alive", return_value=False),
        pytest.raises(RuntimeError, match="keyboard input listener stopped") as exc_info,
    ):
        strategy._run_corrections_only(ctx)

    assert hardware.control_fault is exc_info.value
    robot.get_observation.assert_not_called()
    robot.send_action.assert_not_called()
    engine.pause.assert_called_once_with()


def test_policy_action_rate_limiter_blends_and_limits_by_elapsed_time() -> None:
    now = [0.0]
    limiter = PolicyActionRateLimiter(2.0, 10.0, clock=lambda: now[0])
    limiter.reset({"joint_1.pos": 0.0}, ["joint_1.pos"])

    now[0] = 0.1
    first = limiter({"joint_1.pos": 100.0})
    assert first["joint_1.pos"] == pytest.approx(0.725)

    now[0] = 0.2
    second = limiter({"joint_1.pos": 100.0})
    assert second["joint_1.pos"] == pytest.approx(1.725)

    now[0] = 5.0
    limiter.mark_hold()
    now[0] = 5.1
    after_wait = limiter({"joint_1.pos": 100.0})
    assert after_wait["joint_1.pos"] == pytest.approx(2.725)


def test_policy_action_rate_limiter_does_not_consume_blend_while_holding() -> None:
    now = [0.0]
    limiter = PolicyActionRateLimiter(2.0, None, clock=lambda: now[0])
    limiter.reset({"joint_1.pos": 0.0}, ["joint_1.pos"])

    now[0] = 5.0
    limiter.mark_hold()
    now[0] = 5.1
    first = limiter({"joint_1.pos": 100.0})

    assert first["joint_1.pos"] == pytest.approx(0.725)


def test_policy_action_rate_limiter_requires_every_finite_anchor() -> None:
    limiter = PolicyActionRateLimiter(2.0, 10.0)

    with pytest.raises(RuntimeError, match="joint_2.pos"):
        limiter.reset(
            {"joint_1.pos": 0.0, "joint_2.pos": float("nan")},
            ["joint_1.pos", "joint_2.pos"],
        )


def test_policy_action_rate_limiter_rejects_non_finite_target() -> None:
    limiter = PolicyActionRateLimiter(2.0, 10.0)
    limiter.reset({"joint_1.pos": 0.0}, ["joint_1.pos"])

    with pytest.raises(ValueError, match="non-finite target"):
        limiter({"joint_1.pos": float("nan")})


def test_dagger_resume_notifies_fresh_observation_before_engine_resume() -> None:
    calls: list[str] = []
    strategy = DAggerStrategy(DAggerStrategyConfig())
    strategy._engine = MagicMock()
    strategy._engine.notify_observation.side_effect = lambda _obs: calls.append("notify")
    strategy._engine.resume.side_effect = lambda: calls.append("resume")
    strategy._interpolator = MagicMock()
    strategy._interpolator.needs_new_action.return_value = True
    processors = SimpleNamespace(robot_observation_processor=lambda obs: {"joint_1.pos": obs["joint_1.pos"]})
    ctx = SimpleNamespace(
        processors=processors,
        data=SimpleNamespace(ordered_action_keys=["joint_1.pos"]),
    )

    result = strategy._resume_from_fresh_observation(ctx, {"joint_1.pos": 12.0})

    assert result == {"joint_1.pos": 12.0}
    assert calls == ["notify", "resume"]


def test_dagger_notifies_inner_robot_of_phase_changes_in_safe_order() -> None:
    assert DAGGER_INTERVENTION_PHASE_HOOK_VERSION == 2
    calls: list[str] = []

    class PhaseAwareRobot:
        def set_intervention_phase(self, old_phase: str, new_phase: str) -> None:
            calls.append(f"phase:{old_phase}->{new_phase}")

    engine = MagicMock()
    engine.pause.side_effect = lambda: calls.append("engine.pause")
    engine.reset.side_effect = lambda: calls.append("engine.reset")
    interpolator = MagicMock()
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(
            robot_wrapper=SimpleNamespace(inner=PhaseAwareRobot()),
            teleop=SimpleNamespace(feedback_features={}),
        )
    )

    assert not DAggerStrategy._apply_transition(
        DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED, engine, interpolator, ctx, None
    )
    assert not DAggerStrategy._apply_transition(
        DAggerPhase.PAUSED, DAggerPhase.CORRECTING, engine, interpolator, ctx, None
    )
    assert not DAggerStrategy._apply_transition(
        DAggerPhase.CORRECTING, DAggerPhase.PAUSED, engine, interpolator, ctx, None
    )
    assert DAggerStrategy._apply_transition(
        DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS, engine, interpolator, ctx, None
    )

    assert calls == [
        "engine.pause",
        "phase:autonomous->paused",
        "phase:paused->correcting",
        "phase:correcting->paused",
        "phase:paused->autonomous",
        "engine.reset",
    ]
    engine.resume.assert_not_called()


def test_dagger_external_continuous_feedback_skips_blocking_follower_handover() -> None:
    teleop = SimpleNamespace(
        feedback_features={},
        requires_continuous_feedback=True,
    )
    robot = SimpleNamespace(inner=object())
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=teleop),
    )

    with patch("lerobot.rollout.strategies.dagger.follower_smooth_move_to") as smooth_move:
        assert not DAggerStrategy._apply_transition(
            DAggerPhase.PAUSED,
            DAggerPhase.CORRECTING,
            MagicMock(),
            MagicMock(),
            ctx,
            {"joint_1.pos": 1.0},
        )

    smooth_move.assert_not_called()


def test_dagger_exit_pauses_phase_aware_robot_before_slow_teardown() -> None:
    calls: list[str] = []

    class PhaseAwareRobot:
        def set_intervention_phase(self, old_phase: str, new_phase: str) -> None:
            calls.append(f"phase:{old_phase}->{new_phase}")

    engine = MagicMock()
    engine.pause.side_effect = lambda: calls.append("engine.pause")
    events = DAggerEvents()
    events.phase = DAggerPhase.CORRECTING

    _pause_dagger_control_before_loop_exit(
        engine,
        events,
        SimpleNamespace(inner=PhaseAwareRobot()),
    )

    assert calls == ["engine.pause", "phase:correcting->paused"]
    assert events.phase is DAggerPhase.PAUSED


def test_dagger_exit_attempts_phase_hold_when_engine_pause_fails() -> None:
    calls: list[str] = []
    pause_error = RuntimeError("engine pause failed")

    class PhaseAwareRobot:
        def set_intervention_phase(self, old_phase: str, new_phase: str) -> None:
            calls.append(f"phase:{old_phase}->{new_phase}")

    engine = MagicMock()
    engine.pause.side_effect = pause_error
    events = DAggerEvents()
    events.phase = DAggerPhase.AUTONOMOUS

    with pytest.raises(RuntimeError, match="engine pause failed") as exc_info:
        _pause_dagger_control_before_loop_exit(
            engine,
            events,
            SimpleNamespace(inner=PhaseAwareRobot()),
        )

    assert exc_info.value is pause_error
    assert calls == ["phase:autonomous->paused"]
    assert events.phase is DAggerPhase.PAUSED


def test_dagger_exit_reissues_idempotent_paused_hook() -> None:
    robot = SimpleNamespace(set_intervention_phase=MagicMock())
    events = DAggerEvents()
    events.phase = DAggerPhase.PAUSED

    _pause_dagger_control_before_loop_exit(
        MagicMock(),
        events,
        SimpleNamespace(inner=robot),
    )

    robot.set_intervention_phase.assert_called_once_with("paused", "paused")


def test_dagger_phase_hook_is_optional_and_propagates_failures() -> None:
    engine = MagicMock()
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(
            robot_wrapper=SimpleNamespace(inner=object()),
            teleop=SimpleNamespace(feedback_features={}),
        )
    )

    assert not DAggerStrategy._apply_transition(
        DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED, engine, MagicMock(), ctx, None
    )

    fault = RuntimeError("phase hook failed")
    robot = SimpleNamespace(set_intervention_phase=MagicMock(side_effect=fault))
    ctx.hardware.robot_wrapper = SimpleNamespace(inner=robot)
    with pytest.raises(RuntimeError, match="phase hook failed") as exc_info:
        DAggerStrategy._apply_transition(
            DAggerPhase.PAUSED, DAggerPhase.CORRECTING, engine, MagicMock(), ctx, None
        )

    assert exc_info.value is fault


@pytest.mark.parametrize(
    ("event", "new_phase"),
    [("correction", DAggerPhase.CORRECTING), ("pause_resume", DAggerPhase.AUTONOMOUS)],
)
def test_dagger_events_keep_paused_transition_pending_until_episode_save_is_ready(
    event: str, new_phase: DAggerPhase
) -> None:
    events = DAggerEvents()
    events.phase = DAggerPhase.PAUSED
    events.request_transition(event)

    assert events.consume_transition(paused_transition_ready=False) is None
    assert events.phase is DAggerPhase.PAUSED
    assert events.consume_transition(paused_transition_ready=True) == (
        DAggerPhase.PAUSED,
        new_phase,
    )


def test_dagger_hold_action_zeros_stale_dynamic_targets() -> None:
    action = {
        "joint_1.pos": 4.0,
        "joint_1.vel": 30.0,
        "joint_1.torque": 2.0,
    }

    assert DAggerStrategy._hold_action(action) == {
        "joint_1.pos": 4.0,
        "joint_1.vel": 0.0,
        "joint_1.torque": 0.0,
    }


def test_dagger_measured_hold_replaces_last_target_and_zeros_dynamics() -> None:
    hold = DAggerStrategy._measured_hold_action(
        {
            "joint_1.pos": 20.0,
            "joint_1.vel": 30.0,
            "joint_1.torque": 2.0,
        },
        {"joint_1.pos": 4.5},
        ["joint_1.pos"],
    )

    assert hold == {
        "joint_1.pos": 4.5,
        "joint_1.vel": 0.0,
        "joint_1.torque": 0.0,
    }


def test_dagger_marks_background_inference_failure_as_control_fault() -> None:
    engine = SimpleNamespace(failed=True)
    ctx = SimpleNamespace(hardware=SimpleNamespace(control_fault=None))

    with pytest.raises(RuntimeError, match="background thread") as exc_info:
        DAggerStrategy._raise_if_engine_failed(engine, ctx)

    assert ctx.hardware.control_fault is exc_info.value


def test_dagger_event_only_shutdown_is_a_control_fault() -> None:
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = True
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(shutdown_event=shutdown_event),
        hardware=SimpleNamespace(control_fault=None),
    )

    with pytest.raises(KeyboardInterrupt, match="shutdown signal") as exc_info:
        DAggerStrategy._raise_if_shutdown_requested(ctx)

    assert ctx.hardware.control_fault is exc_info.value


def test_fault_teardown_skips_return_motion_and_secures_in_place() -> None:
    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    calls: list[str] = []
    robot = MagicMock()
    robot.is_connected = True
    robot.secure_intervention_after_fault.side_effect = lambda _teleop: calls.append("secure")
    robot.disconnect.side_effect = lambda: calls.append("robot.disconnect")
    wrapper = MagicMock()
    wrapper.inner = robot
    teleop = MagicMock()
    teleop.is_connected = True
    teleop.disconnect.side_effect = lambda: calls.append("teleop.disconnect")
    hardware = HardwareContext(
        robot_wrapper=wrapper,
        teleop=teleop,
        control_fault=ValueError("feedback fault"),
    )
    strategy = TestStrategy(DAggerStrategyConfig())
    strategy._engine = MagicMock()
    strategy._engine.stop.side_effect = lambda: calls.append("engine.stop")

    strategy._teardown_hardware(hardware)

    assert calls == ["engine.stop", "secure", "robot.disconnect", "teleop.disconnect"]
    robot.finish_intervention_deployment.assert_not_called()
    assert hardware.teardown_complete


def test_fault_teardown_secures_live_leader_when_followers_are_disconnected() -> None:
    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    calls: list[str] = []
    robot = MagicMock()
    robot.is_connected = False
    robot.bus.is_connected = False
    robot.cameras = {}
    robot.secure_intervention_after_fault.side_effect = lambda _teleop: calls.append("secure")
    robot.disconnect.side_effect = lambda: calls.append("robot.disconnect")
    wrapper = MagicMock()
    wrapper.inner = robot
    teleop = MagicMock()
    teleop.is_connected = True
    teleop.disconnect.side_effect = lambda: calls.append("teleop.disconnect")
    hardware = HardwareContext(
        robot_wrapper=wrapper,
        teleop=teleop,
        control_fault=RuntimeError("follower disconnected first"),
    )
    strategy = TestStrategy(DAggerStrategyConfig())
    strategy._engine = MagicMock()
    strategy._engine.stop.side_effect = lambda: calls.append("engine.stop")

    strategy._teardown_hardware(hardware)

    assert calls == ["engine.stop", "secure", "teleop.disconnect"]
    robot.secure_intervention_after_fault.assert_called_once_with(teleop)
    robot.finish_intervention_deployment.assert_not_called()
    robot.disconnect.assert_not_called()
    teleop.disconnect.assert_called_once_with()
    assert hardware.teardown_complete


def test_interrupt_before_coordinated_return_secures_pose_before_disconnect() -> None:
    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    calls: list[str] = []
    interrupt = KeyboardInterrupt("interrupt before coordinated return")
    robot = MagicMock()
    robot.is_connected = True

    def interrupt_return(_teleop):
        calls.append("return")
        raise interrupt

    robot.finish_intervention_deployment.side_effect = interrupt_return
    robot.secure_intervention_after_fault.side_effect = lambda _teleop: calls.append("secure")
    robot.disconnect.side_effect = lambda: calls.append("robot.disconnect")
    wrapper = MagicMock()
    wrapper.inner = robot
    teleop = MagicMock()
    teleop.is_connected = True
    teleop.disconnect.side_effect = lambda: calls.append("teleop.disconnect")
    hardware = HardwareContext(robot_wrapper=wrapper, teleop=teleop)
    strategy = TestStrategy(DAggerStrategyConfig())

    with pytest.raises(KeyboardInterrupt, match="before coordinated return") as exc_info:
        strategy._teardown_hardware(hardware)

    assert exc_info.value is interrupt
    assert hardware.control_fault is interrupt
    assert calls == ["return", "secure", "robot.disconnect", "teleop.disconnect"]
    assert hardware.teardown_complete


def test_core_teardown_detects_single_arm_live_can_when_camera_state_is_false() -> None:
    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    calls: list[str] = []
    robot = MagicMock()
    robot.is_connected = False
    robot.bus.is_connected = True
    robot.cameras = {}
    robot.secure_after_fault.side_effect = lambda: calls.append("secure")
    robot.disconnect.side_effect = lambda: calls.append("robot.disconnect")
    wrapper = MagicMock()
    wrapper.inner = robot
    hardware = HardwareContext(
        robot_wrapper=wrapper,
        teleop=None,
        control_fault=RuntimeError("camera disconnected while CAN stayed live"),
    )
    strategy = TestStrategy(DAggerStrategyConfig())

    strategy._teardown_hardware(hardware)

    assert calls == ["secure", "robot.disconnect"]
    robot.disconnect.assert_called_once_with()
    assert hardware.teardown_complete


def test_engine_stop_failure_prohibits_return_motion() -> None:
    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    robot = MagicMock()
    robot.is_connected = True
    wrapper = MagicMock()
    wrapper.inner = robot
    teleop = MagicMock()
    teleop.is_connected = True
    hardware = HardwareContext(
        robot_wrapper=wrapper,
        teleop=teleop,
        initial_position={"joint_1.pos": 0.0},
    )
    strategy = TestStrategy(DAggerStrategyConfig())
    strategy._engine = MagicMock()
    strategy._engine.stop.side_effect = RuntimeError("engine did not stop")

    with pytest.raises(RuntimeError, match="engine did not stop"):
        strategy._teardown_hardware(hardware)

    assert isinstance(hardware.control_fault, RuntimeError)
    robot.secure_intervention_after_fault.assert_called_once_with(teleop)
    robot.finish_intervention_deployment.assert_not_called()
    robot.disconnect.assert_called_once_with()
    teleop.disconnect.assert_called_once_with()


def test_disconnect_interrupt_keeps_teardown_retryable_and_still_closes_teleop() -> None:
    class TestStrategy(RolloutStrategy):
        def setup(self, ctx):
            pass

        def run(self, ctx):
            pass

        def teardown(self, ctx):
            pass

    robot = MagicMock()
    robot.is_connected = True
    robot.disconnect.side_effect = KeyboardInterrupt("robot disconnect interrupted")
    wrapper = MagicMock()
    wrapper.inner = robot
    teleop = MagicMock()
    teleop.is_connected = True
    hardware = HardwareContext(
        robot_wrapper=wrapper,
        teleop=teleop,
        control_fault=RuntimeError("control fault"),
    )
    strategy = TestStrategy(DAggerStrategyConfig())

    with pytest.raises(KeyboardInterrupt, match="robot disconnect interrupted"):
        strategy._teardown_hardware(hardware)

    robot.secure_intervention_after_fault.assert_called_once_with(teleop)
    robot.disconnect.assert_called_once_with()
    teleop.disconnect.assert_called_once_with()
    assert not hardware.teardown_complete


def test_dagger_teardown_secures_hardware_before_dataset_finalize() -> None:
    calls: list[str] = []
    strategy = DAggerStrategy(DAggerStrategyConfig())
    strategy._teardown_hardware = MagicMock(side_effect=lambda *_args, **_kwargs: calls.append("hardware"))
    dataset = MagicMock()
    dataset.has_pending_frames.return_value = False
    dataset.finalize.side_effect = lambda: calls.append("dataset.finalize")
    cfg = SimpleNamespace(
        play_sounds=False,
        return_to_initial_position=True,
        dataset=SimpleNamespace(push_to_hub=False),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg),
        hardware=MagicMock(),
        data=SimpleNamespace(dataset=dataset),
    )

    strategy.teardown(ctx)

    assert calls == ["hardware", "dataset.finalize"]


def test_dagger_teardown_marks_shutdown_signal_before_hardware_teardown() -> None:
    strategy = DAggerStrategy(DAggerStrategyConfig())
    observed_faults: list[BaseException | None] = []
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)

    def complete_hardware_teardown(*_args, **_kwargs):
        observed_faults.append(hardware.control_fault)
        hardware.teardown_complete = True

    strategy._teardown_hardware = MagicMock(side_effect=complete_hardware_teardown)
    dataset = MagicMock()
    dataset.has_pending_frames.return_value = False
    cfg = SimpleNamespace(
        play_sounds=False,
        return_to_initial_position=True,
        dataset=SimpleNamespace(push_to_hub=False),
    )
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = True
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=shutdown_event),
        hardware=hardware,
        data=SimpleNamespace(dataset=dataset),
    )

    strategy.teardown(ctx)

    assert len(observed_faults) == 1
    assert isinstance(observed_faults[0], KeyboardInterrupt)
    assert hardware.control_fault is observed_faults[0]


def test_dagger_listener_interrupt_becomes_fault_before_hardware_teardown() -> None:
    strategy = DAggerStrategy(DAggerStrategyConfig())
    interrupt = KeyboardInterrupt("listener interrupt")
    strategy._listener = MagicMock()
    strategy._listener.stop.side_effect = interrupt
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)
    observed_faults: list[BaseException | None] = []

    def hardware_teardown(*_args, **_kwargs):
        observed_faults.append(hardware.control_fault)
        hardware.teardown_complete = True

    strategy._teardown_hardware = MagicMock(side_effect=hardware_teardown)
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = False
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                play_sounds=False,
                return_to_initial_position=True,
                dataset=None,
            ),
            shutdown_event=shutdown_event,
        ),
        hardware=hardware,
        data=SimpleNamespace(dataset=None),
    )

    with pytest.raises(KeyboardInterrupt, match="listener interrupt") as exc_info:
        strategy.teardown(ctx)

    assert exc_info.value is interrupt
    assert hardware.control_fault is interrupt
    assert observed_faults == [interrupt]
    assert hardware.teardown_complete


def test_dagger_teardown_skips_stop_for_dead_listener_and_tears_down_hardware() -> None:
    strategy = DAggerStrategy(DAggerStrategyConfig())
    listener = MagicMock()
    strategy._listener = listener
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)

    def hardware_teardown(*_args, **_kwargs):
        hardware.teardown_complete = True

    strategy._teardown_hardware = MagicMock(side_effect=hardware_teardown)
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = False
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                play_sounds=False,
                return_to_initial_position=True,
                dataset=None,
            ),
            shutdown_event=shutdown_event,
        ),
        hardware=hardware,
        data=SimpleNamespace(dataset=None),
    )

    with patch("lerobot.rollout.strategies.dagger.key_listener_is_alive", return_value=False):
        strategy.teardown(ctx)

    listener.stop.assert_not_called()
    strategy._teardown_hardware.assert_called_once_with(
        hardware,
        return_to_initial_position=True,
    )
    assert hardware.teardown_complete


def test_dagger_hardware_interrupt_retries_once_in_fault_mode() -> None:
    strategy = DAggerStrategy(DAggerStrategyConfig())
    interrupt = KeyboardInterrupt("hardware interrupt")
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)
    observed_faults: list[BaseException | None] = []

    def hardware_teardown(*_args, **_kwargs):
        observed_faults.append(hardware.control_fault)
        if len(observed_faults) == 1:
            raise interrupt
        hardware.teardown_complete = True

    strategy._teardown_hardware = MagicMock(side_effect=hardware_teardown)
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = False
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                play_sounds=False,
                return_to_initial_position=True,
                dataset=None,
            ),
            shutdown_event=shutdown_event,
        ),
        hardware=hardware,
        data=SimpleNamespace(dataset=None),
    )

    with pytest.raises(KeyboardInterrupt, match="hardware interrupt") as exc_info:
        strategy.teardown(ctx)

    assert exc_info.value is interrupt
    assert hardware.control_fault is interrupt
    assert observed_faults == [None, interrupt]
    assert hardware.teardown_complete
    assert strategy._teardown_hardware.call_count == 2


def test_dagger_incomplete_hardware_teardown_prohibits_dataset_io() -> None:
    strategy = DAggerStrategy(DAggerStrategyConfig())
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)
    strategy._teardown_hardware = MagicMock(
        side_effect=(RuntimeError("first disconnect failed"), RuntimeError("retry failed"))
    )
    saver = MagicMock()
    saver.save_pending = True
    strategy._episode_saver = saver
    dataset = MagicMock()
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = False
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                play_sounds=False,
                return_to_initial_position=True,
                dataset=SimpleNamespace(push_to_hub=False),
            ),
            shutdown_event=shutdown_event,
        ),
        hardware=hardware,
        data=SimpleNamespace(dataset=dataset),
    )

    with pytest.raises(RuntimeError, match="Hardware teardown remains incomplete"):
        strategy.teardown(ctx)

    assert strategy._teardown_hardware.call_count == 2
    assert isinstance(hardware.control_fault, RuntimeError)
    saver.shutdown.assert_not_called()
    dataset.finalize.assert_not_called()


def test_dagger_teardown_waits_for_async_save_before_inspecting_dataset() -> None:
    calls: list[str] = []
    strategy = DAggerStrategy(DAggerStrategyConfig())
    strategy._teardown_hardware = MagicMock(side_effect=lambda *_args, **_kwargs: calls.append("hardware"))
    saver = MagicMock()
    saver.save_pending = True
    saver.shutdown.side_effect = lambda: calls.append("saver.shutdown")
    strategy._episode_saver = saver
    dataset = MagicMock()
    dataset.has_pending_frames.side_effect = AssertionError("raced with pending save")
    dataset.finalize.side_effect = lambda: calls.append("dataset.finalize")
    cfg = SimpleNamespace(
        play_sounds=False,
        return_to_initial_position=True,
        dataset=SimpleNamespace(push_to_hub=False),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg),
        hardware=MagicMock(),
        data=SimpleNamespace(dataset=dataset),
    )

    strategy.teardown(ctx)

    assert calls == ["hardware", "saver.shutdown", "dataset.finalize"]
    dataset.has_pending_frames.assert_not_called()


def test_dagger_teardown_shuts_down_poisoned_saver_after_submit_failure() -> None:
    calls: list[str] = []
    strategy = DAggerStrategy(DAggerStrategyConfig())
    strategy._teardown_hardware = MagicMock(side_effect=lambda *_args, **_kwargs: calls.append("hardware"))
    saver = MagicMock()
    saver.save_pending = False
    saver.submit_save_episode.side_effect = RuntimeError("save worker poisoned")
    saver.shutdown.side_effect = lambda: calls.append("saver.shutdown")
    strategy._episode_saver = saver
    dataset = MagicMock()
    dataset.has_pending_frames.return_value = True
    dataset.finalize.side_effect = lambda: calls.append("dataset.finalize")
    cfg = SimpleNamespace(
        play_sounds=False,
        return_to_initial_position=True,
        dataset=SimpleNamespace(push_to_hub=False),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg),
        hardware=MagicMock(),
        data=SimpleNamespace(dataset=dataset),
    )

    with pytest.raises(RuntimeError, match="save worker poisoned"):
        strategy.teardown(ctx)

    assert calls == ["hardware", "saver.shutdown", "dataset.finalize"]
    saver.shutdown.assert_called_once_with()


def test_corrections_only_stop_exits_without_collecting_pending_save() -> None:
    strategy = DAggerStrategy(DAggerStrategyConfig(num_episodes=10))
    engine = MagicMock()
    engine.failed = False
    strategy._engine = engine
    interpolator = MagicMock()
    interpolator.get_control_interval.return_value = 1 / 30
    strategy._interpolator = interpolator
    events = MagicMock()
    events.stop_recording.is_set.return_value = True
    strategy._events = events
    saver = MagicMock()
    saver.save_pending = True
    saver.save_in_progress = True
    strategy._episode_saver = saver
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = False
    robot = MagicMock()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                fps=30,
                interpolation_multiplier=1,
                dataset=SimpleNamespace(single_task="task"),
                task="task",
                play_sounds=False,
                duration=0,
                use_torch_compile=False,
            ),
            shutdown_event=shutdown_event,
        ),
        hardware=SimpleNamespace(
            robot_wrapper=robot,
            teleop=MagicMock(),
            control_fault=None,
        ),
        data=SimpleNamespace(
            dataset=MagicMock(),
            dataset_features={},
            ordered_action_keys=[],
        ),
    )

    strategy._run_corrections_only(ctx)

    saver.wait_for_pending_save.assert_not_called()
    robot.get_observation.assert_not_called()
    engine.pause.assert_called_once_with()


def test_dagger_teardown_suppresses_push_after_async_persistence_failure() -> None:
    strategy = DAggerStrategy(DAggerStrategyConfig())
    hardware = SimpleNamespace(control_fault=None, teardown_complete=False)

    def hardware_teardown(*_args, **_kwargs):
        hardware.teardown_complete = True

    strategy._teardown_hardware = MagicMock(side_effect=hardware_teardown)
    saver = MagicMock()
    saver.save_pending = True
    saver.shutdown.side_effect = RuntimeError("save failed")
    strategy._episode_saver = saver
    strategy._needs_push.set()
    dataset = MagicMock()
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = False
    cfg = SimpleNamespace(
        play_sounds=False,
        return_to_initial_position=True,
        dataset=SimpleNamespace(
            push_to_hub=True,
            tags=["dagger"],
            private=True,
        ),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=shutdown_event),
        hardware=hardware,
        data=SimpleNamespace(dataset=dataset),
    )

    with (
        patch("lerobot.rollout.strategies.dagger.safe_push_to_hub") as push,
        pytest.raises(RuntimeError, match="save failed"),
    ):
        strategy.teardown(ctx)

    dataset.finalize.assert_called_once_with()
    push.assert_not_called()
    assert isinstance(hardware.control_fault, RuntimeError)
