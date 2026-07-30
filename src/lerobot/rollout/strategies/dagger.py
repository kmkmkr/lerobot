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

"""DAgger rollout strategy: Human-in-the-Loop data collection.

Implements the RaC paradigm (Recovery and Correction) for interactive
imitation learning.  Alternates between autonomous policy execution and
human intervention via teleoperator.

Input is controlled via either a keyboard or foot pedal, selected by
the ``input_device`` config field.  Each device exposes three actions:

    1. **pause_resume** — Toggle policy execution (AUTONOMOUS <-> PAUSED).
    2. **correction**   — Toggle correction recording (PAUSED <-> CORRECTING).
    3. **upload**        — Push dataset to hub on demand (corrections-only mode).
    ESC (keyboard only) — Stop session.

Recording modes:
    ``record_autonomous=True``:  Sentry-like continuous recording with
        time-based episode rotation.  Both autonomous and correction
        frames are recorded; corrections tagged ``intervention=True``.
    ``record_autonomous=False``: Only correction windows are recorded.
        Each correction (start to stop) becomes one episode.

Teleoperator handover:
    Teleoperators that declare ``requires_continuous_feedback`` (the OpenArm
    bilateral leaders) receive the measured follower observation on every
    control tick in every phase.  Their torque remains enabled across DAgger
    transitions so leader PD, gravity compensation, friction compensation,
    and force reflection stay active during policy execution and intervention.

    On AUTONOMOUS → PAUSED, actuated teleops (those with non-empty
    ``feedback_features``, e.g. SO-101, OpenArmMini) are smoothly driven to
    the follower's last position via ``send_feedback`` so the operator takes
    over without a jerk.  Non-actuated teleops cannot be driven,
    so on PAUSED → CORRECTING the follower is instead slid to the teleop's
    current pose before the correction begins.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

import numpy as np

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import create_key_listener, key_listener_is_alive
from lerobot.utils.pedal import start_pedal_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..async_dataset import AsyncEpisodeSaver
from ..configs import DAggerKeyboardConfig, DAggerPedalConfig, DAggerStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, estimate_max_episode_seconds, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


def _teleop_requires_continuous_feedback(teleop) -> bool:
    """Return whether a teleoperator needs follower feedback on every control tick."""
    return bool(getattr(teleop, "requires_continuous_feedback", False))


def _send_continuous_feedback(teleop, observation: dict[str, Any]) -> None:
    """Maintain feedback control for teleoperators such as the OpenArm leader."""
    if _teleop_requires_continuous_feedback(teleop):
        teleop.send_feedback(observation)


class PolicyActionRateLimiter:
    """Blend and velocity-limit policy targets after autonomous handover."""

    def __init__(
        self,
        blend_duration_s: float,
        max_velocity: float | None,
        *,
        clock=time.perf_counter,
    ) -> None:
        self._blend_duration_s = blend_duration_s
        self._max_velocity = max_velocity
        self._clock = clock
        self._anchor: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._started_at: float | None = None
        self._last_time: float | None = None

    def reset(self, observation: dict[str, Any], action_keys: list[str]) -> None:
        """Anchor a new handover to a fresh measured observation."""
        anchor: dict[str, float] = {}
        for key in action_keys:
            if key not in observation:
                continue
            value_array = np.asarray(observation[key])
            if value_array.size != 1:
                continue
            value = float(value_array.item())
            if np.isfinite(value):
                anchor[key] = value

        missing_keys = [key for key in action_keys if key not in anchor]
        if (self._blend_duration_s > 0 or self._max_velocity is not None) and missing_keys:
            raise RuntimeError(
                "DAgger safe autonomous handover requires a finite measured value for every "
                f"policy action key; missing or invalid: {missing_keys}"
            )

        now = self._clock()
        self._anchor = anchor
        self._last = anchor.copy()
        self._started_at = now
        self._last_time = now

    def mark_hold(self) -> None:
        """Freeze blend progress and velocity time while no policy target exists."""
        if self._last_time is not None and self._started_at is not None:
            now = self._clock()
            self._started_at += max(0.0, now - self._last_time)
            self._last_time = now

    def __call__(self, action: dict[str, float]) -> dict[str, float]:
        if self._started_at is None or self._last_time is None:
            return dict(action)

        now = self._clock()
        elapsed = max(0.0, now - self._started_at)
        dt = max(0.0, now - self._last_time)
        if self._blend_duration_s == 0:
            blend = 1.0
        else:
            ratio = min(1.0, elapsed / self._blend_duration_s)
            blend = ratio * ratio * (3.0 - 2.0 * ratio)

        limited = {key: float(value) for key, value in action.items()}
        for key, value in limited.items():
            if not np.isfinite(value):
                raise ValueError(f"DAgger policy produced a non-finite target for {key}: {value}")

        missing_targets = [key for key in self._anchor if key not in limited]
        if missing_targets:
            raise ValueError(f"DAgger policy action is missing anchored targets: {missing_targets}")

        for key, anchor in self._anchor.items():
            target = limited[key]
            target = anchor + blend * (target - anchor)
            if self._max_velocity is not None:
                max_delta = self._max_velocity * dt
                previous = self._last[key]
                target = min(max(target, previous - max_delta), previous + max_delta)
            limited[key] = target
            self._last[key] = target

        self._last_time = now
        return limited


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


class DAggerPhase(enum.Enum):
    """Observable phases of a DAgger episode."""

    AUTONOMOUS = "autonomous"  # Policy driving
    PAUSED = "paused"  # Engine paused, teleop aligned, awaiting input
    CORRECTING = "correcting"  # Human driving via teleop, recording interventions


def _set_robot_intervention_phase(
    robot_wrapper, old_phase: DAggerPhase, new_phase: DAggerPhase
) -> None:
    """Notify robots that opt into DAgger intervention phase changes."""
    robot = getattr(robot_wrapper, "inner", robot_wrapper)
    set_phase = getattr(robot, "set_intervention_phase", None)
    if callable(set_phase):
        set_phase(old_phase.value, new_phase.value)


# Valid (current_phase, event) -> next_phase
_DAGGER_TRANSITIONS: dict[tuple[DAggerPhase, str], DAggerPhase] = {
    (DAggerPhase.AUTONOMOUS, "pause_resume"): DAggerPhase.PAUSED,
    (DAggerPhase.PAUSED, "pause_resume"): DAggerPhase.AUTONOMOUS,
    (DAggerPhase.PAUSED, "correction"): DAggerPhase.CORRECTING,
    (DAggerPhase.CORRECTING, "correction"): DAggerPhase.PAUSED,
}


class DAggerEvents:
    """Thread-safe container for DAgger input device events.

    The keyboard/pedal threads write transition requests; the main loop
    consumes them.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = DAggerPhase.AUTONOMOUS
        self._pending_transition: str | None = None

        # Session-level flags
        self.stop_recording = Event()
        self.upload_requested = Event()

    # -- Thread-safe phase access ------------------------------------------

    @property
    def phase(self) -> DAggerPhase:
        """Current phase of the DAgger state machine."""
        with self._lock:
            return self._phase

    @phase.setter
    def phase(self, value: DAggerPhase) -> None:
        with self._lock:
            self._phase = value

    def request_transition(self, event: str) -> None:
        """Request a phase transition (called from keyboard/pedal threads).

        Only enqueues the request if it corresponds to a valid transition
        from the current phase, preventing impossible state changes.
        """
        with self._lock:
            if (self._phase, event) in _DAGGER_TRANSITIONS:
                self._pending_transition = event

    def consume_transition(
        self, *, paused_transition_ready: bool = True
    ) -> tuple[DAggerPhase, DAggerPhase] | None:
        """Consume a pending transition (called from main loop)."""
        with self._lock:
            if self._pending_transition is None:
                return None
            key = (self._phase, self._pending_transition)
            if self._phase == DAggerPhase.PAUSED and not paused_transition_ready:
                return None
            self._pending_transition = None
            new_phase = _DAGGER_TRANSITIONS.get(key)
            if new_phase is None:
                return None
            old_phase = self._phase
            self._phase = new_phase
            return old_phase, new_phase

    def reset(self) -> None:
        """Reset all transient state for a fresh session."""
        with self._lock:
            self._phase = DAggerPhase.AUTONOMOUS
            self._pending_transition = None
        self.upload_requested.clear()


# ---------------------------------------------------------------------------
# Input device handlers
# ---------------------------------------------------------------------------


def _init_dagger_keyboard(events: DAggerEvents, cfg: DAggerKeyboardConfig):
    """Initialise a keyboard listener for DAgger's 3 controls.

    Backend selection is delegated to :func:`create_key_listener`. DAgger prefers the
    controlling POSIX terminal even when X11 is available, then falls back to a usable
    pynput backend on platforms without one. Returns the
    listener (exposing ``stop()``) or ``None`` when no keyboard backend is usable.
    """
    # Map config key names to DAgger event names.
    key_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction: "correction",
    }

    def dispatch(name: str) -> None:
        """Apply a resolved key name to the DAgger events."""
        if name == "esc":
            logger.info("Stop recording...")
            events.stop_recording.set()
            return
        if name in key_to_event:
            events.request_transition(key_to_event[name])
        if name == cfg.upload:
            events.upload_requested.set()

    return create_key_listener(
        dispatch,
        prefer_terminal=True,
        controls_help=(
            f"pause_resume='{cfg.pause_resume}', correction='{cfg.correction}', "
            f"upload='{cfg.upload}', ESC=stop"
        ),
    )


def _init_dagger_pedal(events: DAggerEvents, cfg: DAggerPedalConfig):
    """Initialise foot pedal listener with DAgger 3-pedal controls.

    Returns the pedal listener thread (or ``None`` if evdev is unavailable).
    """
    code_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction: "correction",
    }

    def on_press(code: str) -> None:
        if code in code_to_event:
            events.request_transition(code_to_event[code])
        if code == cfg.upload:
            events.upload_requested.set()

    logger.info("Initializing DAgger foot pedal listener (device=%s)", cfg.device_path)
    return start_pedal_listener(on_press, device_path=cfg.device_path)


# ---------------------------------------------------------------------------
# DAgger Strategy
# ---------------------------------------------------------------------------


class DAggerStrategy(RolloutStrategy):
    """Human-in-the-Loop data collection with intervention tagging.

    State machine::

        AUTONOMOUS --(key1)--> PAUSED --(key2)--> CORRECTING --(key2)--> PAUSED
                               --(key1)--> AUTONOMOUS

    Recording modes:
        ``record_autonomous=True``: Sentry-like continuous recording with
            time-based episode rotation.  Intervention frames tagged True.
        ``record_autonomous=False``: Only correction windows recorded.
            Each correction = one episode.  Upload on demand via key3.
    """

    config: DAggerStrategyConfig

    def __init__(self, config: DAggerStrategyConfig):
        super().__init__(config)
        self._listener = None
        self._pedal_thread = None
        self._events = DAggerEvents()
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()

        self._episode_saver: AsyncEpisodeSaver | None = None
        self._rate_limiter = PolicyActionRateLimiter(
            blend_duration_s=config.resume_blend_duration_s,
            max_velocity=config.max_action_velocity,
        )

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine and input device listener."""
        self._init_engine(ctx)
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dagger-push")
        target_mb = self.config.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        self._episode_duration_s = estimate_max_episode_seconds(
            ctx.data.dataset_features, ctx.runtime.cfg.fps, target_size_mb=target_mb
        )

        if not self.config.record_autonomous:
            if ctx.data.dataset is None:
                raise RuntimeError("DAgger corrections-only mode requires a writable dataset")
            self._episode_saver = AsyncEpisodeSaver(
                ctx.data.dataset, dataset_lock=self._episode_lock, thread_name_prefix="dagger-save"
            )

        if self.config.input_device == "keyboard":
            self._listener = _init_dagger_keyboard(self._events, self.config.keyboard)
            if self._listener is None:
                raise RuntimeError("DAgger cannot start because no keyboard input backend is available")
            self._raise_if_keyboard_listener_failed()
        else:
            self._pedal_thread = _init_dagger_pedal(self._events, self.config.pedal)
            if self._pedal_thread is None:
                raise RuntimeError("DAgger cannot start because no pedal input backend is available")

        record_mode = "all frames (sentry-like)" if self.config.record_autonomous else "corrections only"
        logger.info(
            "DAgger strategy ready (input=%s, episodes=%d, record=%s, episode_duration=%.0fs)",
            self.config.input_device,
            self.config.num_episodes,
            record_mode,
            self._episode_duration_s,
        )

    def run(self, ctx: RolloutContext) -> None:
        """Run DAgger episodes with human-in-the-loop intervention."""
        if self.config.record_autonomous:
            self._run_continuous(ctx)
        else:
            self._run_corrections_only(ctx)

    def _raise_if_keyboard_listener_failed(self) -> None:
        """Stop control if the configured keyboard backend is no longer receiving input."""
        if self.config.input_device != "keyboard":
            return
        if self._listener is not None and key_listener_is_alive(self._listener):
            return
        raise RuntimeError(
            "DAgger keyboard input listener stopped unexpectedly; "
            "stopping control before sending additional robot commands"
        )

    def _resume_from_fresh_observation(
        self,
        ctx: RolloutContext,
        observation: dict[str, Any],
    ) -> dict:
        """Publish current state before allowing the reset engine to run."""
        self._cached_obs_processed = None
        obs_processed = self._process_observation_and_notify(ctx.processors, observation)
        self._rate_limiter.reset(
            observation,
            list(ctx.data.ordered_action_keys),
        )
        self._engine.resume()
        logger.info("Autonomous inference resumed from a fresh measured observation")
        return obs_processed

    @staticmethod
    def _hold_action(action: dict[str, Any]) -> dict[str, Any]:
        """Hold position without replaying stale velocity or torque targets."""
        return {key: 0.0 if key.endswith((".vel", ".torque")) else value for key, value in action.items()}

    @classmethod
    def _measured_hold_action(
        cls,
        previous_action: dict[str, Any] | None,
        observation: dict[str, Any],
        action_keys: list[str],
    ) -> dict[str, Any]:
        """Snapshot a finite measured position hold at a PAUSED transition."""
        hold = cls._hold_action(previous_action or {})
        missing: list[str] = []
        for key in action_keys:
            if not key.endswith(".pos"):
                continue
            if key not in observation:
                missing.append(key)
                continue
            value_array = np.asarray(observation[key])
            if value_array.size != 1:
                missing.append(key)
                continue
            value = float(value_array.item())
            if not np.isfinite(value):
                missing.append(key)
                continue
            hold[key] = value

        if missing:
            raise RuntimeError(
                f"DAgger cannot establish a measured PAUSED hold; missing or invalid positions: {missing}"
            )
        return hold

    @staticmethod
    def _raise_if_engine_failed(engine, ctx: RolloutContext) -> None:
        """Turn a background inference-thread failure into a control fault."""
        if not engine.failed:
            return
        error = RuntimeError("The inference engine failed in its background thread")
        ctx.hardware.control_fault = error
        raise error

    @staticmethod
    def _raise_if_shutdown_requested(ctx: RolloutContext) -> None:
        """Treat an event-only shutdown as a control fault, never an orderly return."""
        if not ctx.runtime.shutdown_event.is_set():
            return
        error = KeyboardInterrupt("Rollout interrupted by a shutdown signal")
        ctx.hardware.control_fault = error
        raise error

    @staticmethod
    def _record_teardown_error(
        ctx: RolloutContext,
        teardown_errors: list[BaseException],
        error: BaseException,
        phase: str,
    ) -> None:
        """Promote every teardown failure to a no-return control fault."""
        existing_fault = ctx.hardware.control_fault
        if existing_fault is None:
            ctx.hardware.control_fault = error
        elif existing_fault is not error:
            existing_fault.add_note(f"Additional {phase} teardown error: {error!r}")
        if not any(recorded is error for recorded in teardown_errors):
            teardown_errors.append(error)

    @staticmethod
    def _promote_shutdown_signal(ctx: RolloutContext) -> None:
        """Latch an event-only signal before any shutdown motion is considered."""
        shutdown_event = getattr(ctx.runtime, "shutdown_event", None)
        if shutdown_event is not None and shutdown_event.is_set() and ctx.hardware.control_fault is None:
            ctx.hardware.control_fault = KeyboardInterrupt("Rollout interrupted by a shutdown signal")
            logger.warning("Shutdown signal detected; prohibiting coordinated return motion")

    def _teardown_hardware_best_effort(
        self,
        ctx: RolloutContext,
        teardown_errors: list[BaseException],
    ) -> None:
        """Retry once in fault mode if teardown is interrupted before completion."""
        for attempt in range(1, 3):
            try:
                # Re-check immediately before each attempt. A signal may have
                # arrived after teardown began but before hardware was reached.
                self._promote_shutdown_signal(ctx)
                self._teardown_hardware(
                    ctx.hardware,
                    return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
                )
            except BaseException as error:
                self._record_teardown_error(
                    ctx,
                    teardown_errors,
                    error,
                    f"hardware attempt {attempt}",
                )
                logger.exception("Hardware teardown attempt %d failed", attempt)
                if attempt == 1:
                    # Core teardown marks completion in its innermost finally.
                    # Clear it because an asynchronous BaseException may have
                    # interrupted teleoperator disconnect just before that mark.
                    ctx.hardware.teardown_complete = False
                    logger.critical("Retrying hardware teardown in no-return fault mode")
                    continue
            break

        if not bool(getattr(ctx.hardware, "teardown_complete", False)):
            error = RuntimeError(
                "Hardware teardown remains incomplete after fault-mode retries; "
                "dataset persistence and finalization are prohibited"
            )
            self._record_teardown_error(
                ctx,
                teardown_errors,
                error,
                "hardware completion",
            )
            if len(teardown_errors) > 1:
                raise error from teardown_errors[-2]
            raise error

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop listeners, finalise the dataset, and disconnect hardware."""
        teardown_errors: list[BaseException] = []
        play_sounds = ctx.runtime.cfg.play_sounds
        try:
            try:
                logger.info("Stopping DAgger recording")
                log_say("Stopping DAgger recording", play_sounds)
            except BaseException as error:
                self._record_teardown_error(ctx, teardown_errors, error, "announcement")
                logger.exception("Failed to announce DAgger shutdown; continuing hardware teardown")

            if self._listener is not None:
                if key_listener_is_alive(self._listener):
                    logger.info("Stopping keyboard listener")
                    try:
                        self._listener.stop()
                    except BaseException as error:
                        self._record_teardown_error(ctx, teardown_errors, error, "input listener")
                        logger.exception("Failed to stop keyboard listener; continuing hardware teardown")
                else:
                    logger.warning("Keyboard listener is already stopped; skipping listener.stop()")
        finally:
            # Covers an asynchronous BaseException delivered between the guarded
            # pre-hardware operations above. Hardware still gets a fault-mode
            # secure/disconnect attempt before the exception can propagate.
            active_error = sys.exception()
            if active_error is not None:
                self._record_teardown_error(ctx, teardown_errors, active_error, "pre-hardware")
            self._teardown_hardware_best_effort(ctx, teardown_errors)

        if self._episode_saver is not None:
            logger.info("Waiting for pending episode persistence after hardware shutdown...")
            saver_error: BaseException | None = None
            save_was_pending = False
            save_submitted = False
            try:
                save_was_pending = self._episode_saver.save_pending
                if not save_was_pending and ctx.data.dataset is not None:
                    # save_episode mutates the episode buffer. Never inspect it
                    # concurrently with the background worker.
                    with self._episode_saver.dataset_lock:
                        has_pending_frames = ctx.data.dataset.has_pending_frames()
                    if has_pending_frames:
                        self._episode_saver.submit_save_episode()
                        save_submitted = True
            except BaseException as error:
                saver_error = error
                logger.exception("Asynchronous episode persistence failed")
            try:
                self._episode_saver.shutdown()
            except BaseException as error:
                if saver_error is None:
                    saver_error = error
                else:
                    saver_error.add_note(f"Additional episode saver shutdown error: {error!r}")
                logger.exception("Failed to shut down asynchronous episode persistence")
            finally:
                self._episode_saver = None
            if saver_error is None and (save_was_pending or save_submitted):
                self._needs_push.set()
            if saver_error is not None:
                self._record_teardown_error(ctx, teardown_errors, saver_error, "episode saver")
        elif ctx.data.dataset is not None and ctx.data.dataset.has_pending_frames():
            try:
                with self._episode_lock:
                    ctx.data.dataset.save_episode()
                self._needs_push.set()
            except BaseException as error:
                self._record_teardown_error(ctx, teardown_errors, error, "episode save")
                logger.exception("Failed to save final in-progress episode")
        # Flush any queued/running push cleanly
        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            try:
                self._push_executor.shutdown(wait=True)
            except BaseException as error:
                self._record_teardown_error(ctx, teardown_errors, error, "push executor")
                logger.exception("Failed to shut down DAgger push executor")
            finally:
                self._push_executor = None

        if ctx.data.dataset is not None:
            try:
                logger.info("Finalizing dataset...")
                ctx.data.dataset.finalize()
                if (
                    not teardown_errors
                    and ctx.hardware.control_fault is None
                    and self._needs_push.is_set()
                    and ctx.runtime.cfg.dataset
                    and ctx.runtime.cfg.dataset.push_to_hub
                ):
                    logger.info("Pushing final dataset to hub...")
                    if safe_push_to_hub(
                        ctx.data.dataset,
                        tags=ctx.runtime.cfg.dataset.tags,
                        private=ctx.runtime.cfg.dataset.private,
                    ):
                        logger.info("Dataset uploaded to hub")
                        log_say("Dataset uploaded to hub", play_sounds)
            except BaseException as error:
                self._record_teardown_error(ctx, teardown_errors, error, "dataset finalization")
                logger.exception("Dataset finalization or final push failed")

        if teardown_errors:
            first_error = teardown_errors[0]
            for additional_error in teardown_errors[1:]:
                first_error.add_note(f"Additional teardown error: {additional_error!r}")
            raise first_error

        logger.info("DAgger strategy teardown complete")

    # ------------------------------------------------------------------
    # Continuous recording mode (record_autonomous=True)
    # ------------------------------------------------------------------

    def _run_continuous(self, ctx: RolloutContext) -> None:
        """Sentry-like continuous recording with intervention tagging.

        Episodes are auto-rotated every ``episode_time_s`` seconds and
        uploaded in the background every ``upload_every_n_episodes`` episodes.
        Both autonomous and correction frames are recorded; corrections are
        tagged with ``intervention=True``.
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride = max(1, cfg.interpolation_multiplier)
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        resume_pending = True

        last_action: dict[str, Any] | None = None
        record_tick = 0
        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        episode_duration_s = self._episode_duration_s
        logger.info("DAgger continuous recording started (episode_duration=%.0fs)", episode_duration_s)

        with contextlib.nullcontext():
            try:
                while not events.stop_recording.is_set() and not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    self._raise_if_keyboard_listener_failed()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # Process transitions
                    entered_paused = False
                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        entered_paused = new_phase == DAggerPhase.PAUSED
                        resume_pending = self._apply_transition(
                            old_phase,
                            new_phase,
                            engine,
                            interpolator,
                            ctx,
                            last_action,
                        )

                    phase = events.phase
                    obs = robot.get_observation()
                    if entered_paused:
                        last_action = self._measured_hold_action(
                            last_action, obs, list(ctx.data.ordered_action_keys)
                        )

                    # Read the leader before returning follower feedback so OpenArm
                    # compensation can reuse the fresh leader state. Feedback is
                    # then maintained once per tick in every DAgger phase.
                    teleop_action = teleop.get_action() if phase == DAggerPhase.CORRECTING else None
                    _send_continuous_feedback(teleop, obs)

                    # --- CORRECTING: human teleop control ---
                    # TODO(Steven): teleop runs at the same FPS as the policy. To
                    # decouple the two, sample teleop at its native rate and
                    # interpolate to the control loop's tick rate.
                    if phase == DAggerPhase.CORRECTING:
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        assert teleop_action is not None
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)
                        if record_tick % record_stride == 0:
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            frame = {
                                **obs_frame,
                                **action_frame,
                                "task": task_str,
                                "intervention": np.array([True], dtype=bool),
                            }
                            dataset.add_frame(frame)
                        record_tick += 1

                    # --- PAUSED: hold position ---
                    elif phase == DAggerPhase.PAUSED:
                        if last_action:
                            robot.send_action(self._hold_action(last_action))

                    # --- AUTONOMOUS: policy control ---
                    else:
                        if resume_pending:
                            obs_processed = self._resume_from_fresh_observation(ctx, obs)
                            resume_pending = False
                        else:
                            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        warmup_was_flushed = self._warmup_flushed
                        if self._handle_warmup(
                            cfg.use_torch_compile,
                            loop_start,
                            control_interval,
                            resume_after_reset=False,
                        ):
                            self._rate_limiter.mark_hold()
                            if last_action:
                                robot.send_action(self._hold_action(last_action))
                            continue
                        if not warmup_was_flushed and self._warmup_flushed:
                            obs_processed = self._resume_from_fresh_observation(ctx, obs)

                        action_dict = send_next_action(
                            obs_processed, obs, ctx, interpolator, action_filter=self._rate_limiter
                        )
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))
                            if record_tick % record_stride == 0:
                                obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                                action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                                frame = {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([False], dtype=bool),
                                }
                                dataset.add_frame(frame)
                            record_tick += 1
                        else:
                            self._rate_limiter.mark_hold()
                            if last_action:
                                robot.send_action(self._hold_action(last_action))

                    # Episode rotation derived from the video file-size target.
                    # Saving is deferred while a correction is ongoing so the
                    # episode boundary lands on a clean autonomous frame.
                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= episode_duration_s and phase != DAggerPhase.CORRECTING:
                        with self._episode_lock:
                            dataset.save_episode()
                        episodes_since_push += 1
                        self._needs_push.set()
                        logger.info(
                            "Episode saved (total: %d, elapsed: %.1fs)",
                            dataset.num_episodes,
                            elapsed,
                        )
                        log_say(f"Episode {dataset.num_episodes} saved", play_sounds)

                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0

                        episode_start = time.perf_counter()

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                        )

            except Exception as error:
                ctx.hardware.control_fault = error
                raise
            finally:
                logger.info("DAgger continuous control loop ended — pausing engine")
                active_error = sys.exception()
                try:
                    engine.pause()
                except BaseException as pause_error:
                    if active_error is None:
                        ctx.hardware.control_fault = pause_error
                        raise
                    active_error.add_note(f"Additional engine pause error: {pause_error!r}")
                    logger.exception("Engine pause also failed; preserving the control-loop error")

        self._raise_if_engine_failed(engine, ctx)
        self._raise_if_shutdown_requested(ctx)

    # ------------------------------------------------------------------
    # Corrections-only mode (record_autonomous=False)
    # ------------------------------------------------------------------

    def _run_corrections_only(self, ctx: RolloutContext) -> None:
        """Record only human correction windows.  Each correction = one episode.

        The policy runs autonomously without recording.  When the user
        pauses and starts a correction, frames are recorded with
        ``intervention=True``.  Stopping the correction saves the episode.
        The dataset can be uploaded on demand via the upload key/pedal.
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features
        saver = self._episode_saver
        if saver is None:
            raise RuntimeError("DAgger corrections-only episode saver was not initialized")

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride = max(1, cfg.interpolation_multiplier)
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        resume_pending = True

        last_action: dict[str, Any] | None = None
        start_time = time.perf_counter()
        record_tick = 0
        recorded = 0
        logger.info(
            "DAgger corrections-only recording started (target: %d episodes)", self.config.num_episodes
        )

        with contextlib.nullcontext():
            try:
                while (
                    recorded < self.config.num_episodes
                    and not events.stop_recording.is_set()
                    and not ctx.runtime.shutdown_event.is_set()
                ):
                    loop_start = time.perf_counter()

                    self._raise_if_keyboard_listener_failed()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # Collect a completed background result without blocking the
                    # feedback loop. Space and Tab stay pending while saving.
                    if saver.save_pending and not saver.save_in_progress:
                        if saver.wait_for_pending_save():
                            self._needs_push.set()
                        logger.info("Previous correction persistence completed")

                    # Process transitions
                    entered_paused = False
                    transition = events.consume_transition(paused_transition_ready=not saver.save_pending)
                    if transition is not None:
                        old_phase, new_phase = transition
                        entered_paused = new_phase == DAggerPhase.PAUSED
                        resume_pending = self._apply_transition(
                            old_phase,
                            new_phase,
                            engine,
                            interpolator,
                            ctx,
                            last_action,
                        )

                        # Persist in the background while PAUSED hold and feedback
                        # continue at the control-loop frequency.
                        if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                            saver.submit_save_episode()
                            recorded += 1
                            logger.info(
                                "Correction %d/%d queued for persistence",
                                recorded,
                                self.config.num_episodes,
                            )
                            log_say(f"Correction {recorded} queued", play_sounds)

                    # On-demand upload
                    if events.upload_requested.is_set():
                        events.upload_requested.clear()
                        logger.info("Upload requested by user")
                        self._background_push(dataset, cfg)

                    phase = events.phase
                    obs = robot.get_observation()
                    if entered_paused:
                        last_action = self._measured_hold_action(
                            last_action, obs, list(ctx.data.ordered_action_keys)
                        )

                    # Read the leader before returning follower feedback so OpenArm
                    # compensation can reuse the fresh leader state. Feedback is
                    # then maintained once per tick in every DAgger phase.
                    teleop_action = teleop.get_action() if phase == DAggerPhase.CORRECTING else None
                    _send_continuous_feedback(teleop, obs)

                    # --- CORRECTING: human teleop control + recording ---
                    # TODO(Steven): teleop runs at the same FPS as the policy. To
                    # decouple the two, sample teleop at its native rate and
                    # interpolate to the control loop's tick rate.
                    if phase == DAggerPhase.CORRECTING:
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        assert teleop_action is not None
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)

                        if record_tick % record_stride == 0:
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            saver.add_frame(
                                {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([True], dtype=bool),
                                }
                            )
                        record_tick += 1

                    # --- PAUSED: hold position ---
                    elif phase == DAggerPhase.PAUSED:
                        if last_action:
                            robot.send_action(self._hold_action(last_action))

                    # --- AUTONOMOUS: policy control (no recording) ---
                    else:
                        if resume_pending:
                            obs_processed = self._resume_from_fresh_observation(ctx, obs)
                            resume_pending = False
                        else:
                            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        warmup_was_flushed = self._warmup_flushed
                        if self._handle_warmup(
                            cfg.use_torch_compile,
                            loop_start,
                            control_interval,
                            resume_after_reset=False,
                        ):
                            self._rate_limiter.mark_hold()
                            if last_action:
                                robot.send_action(self._hold_action(last_action))
                            continue
                        if not warmup_was_flushed and self._warmup_flushed:
                            obs_processed = self._resume_from_fresh_observation(ctx, obs)

                        action_dict = send_next_action(
                            obs_processed, obs, ctx, interpolator, action_filter=self._rate_limiter
                        )
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))
                        else:
                            self._rate_limiter.mark_hold()
                            if last_action:
                                robot.send_action(self._hold_action(last_action))

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                        )

            except Exception as error:
                ctx.hardware.control_fault = error
                raise
            finally:
                logger.info("DAgger corrections-only loop ended — pausing engine")
                active_error = sys.exception()
                try:
                    engine.pause()
                except BaseException as pause_error:
                    if active_error is None:
                        ctx.hardware.control_fault = pause_error
                        raise
                    active_error.add_note(f"Additional engine pause error: {pause_error!r}")
                    logger.exception("Engine pause also failed; preserving the control-loop error")

        self._raise_if_engine_failed(engine, ctx)
        self._raise_if_shutdown_requested(ctx)

    # ------------------------------------------------------------------
    # State-machine transition side-effects
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transition(
        old_phase: DAggerPhase,
        new_phase: DAggerPhase,
        engine,
        interpolator,
        ctx: RolloutContext,
        prev_action: dict | None,
    ) -> bool:
        """Execute side-effects for a validated phase transition, including smooth handovers.

        AUTONOMOUS -> PAUSED (actuated teleop):
            Pause the engine, then drive the leader arm to the follower's last
            commanded position so the operator takes over without a jerk.
            Continuous-feedback teleops already track measured follower state,
            so they do not need this blocking handover.

        PAUSED -> CORRECTING (non-actuated teleop):
            Slide the follower to the teleop's current pose so the robot meets
            the operator's hand rather than jumping to it on the first frame.

        CORRECTING -> PAUSED (actuated teleop):
            Re-enable torque to hold position after correction.
            This will be potentially useful if cancelling the correction recording

        PAUSED -> AUTONOMOUS:
            Reset the inference engine; the caller resumes it only after publishing a fresh observation.

        Continuous-feedback teleops keep torque enabled for all transitions;
        the main loop supplies measured follower feedback in every phase.

        Robots exposing ``set_intervention_phase(old_phase, new_phase)`` are
        notified at these safe boundaries.  The AUTONOMOUS notification occurs
        before inference restarts, so hardware may hold until its next action.
        """
        teleop = ctx.hardware.teleop
        robot = ctx.hardware.robot_wrapper
        supports_feedback = teleop_supports_feedback(teleop)
        continuous_feedback = _teleop_requires_continuous_feedback(teleop)

        logger.info("Phase transition: %s -> %s", old_phase.value, new_phase.value)
        if old_phase == DAggerPhase.AUTONOMOUS and new_phase == DAggerPhase.PAUSED:
            logger.info("Pausing engine - robot holds position")
            engine.pause()
            _set_robot_intervention_phase(robot, old_phase, new_phase)

            if supports_feedback and not continuous_feedback and prev_action is not None:
                # TODO(Maxime): prev_action is in robot action key space (output of robot_action_processor).
                # send_feedback expects teleop feedback key space. For homogeneous setups (e.g. SO-101
                # leader + SO-101 follower) the keys are identical so this works. If the processor pipeline
                # does non-trivial key renaming (e.g. a rename_map on action keys), the interpolation in
                # teleop_smooth_move_to silently no-ops and the arm doesn't move.
                logger.info("Smooth handover: moving leader arm to follower position")
                teleop_smooth_move_to(teleop, prev_action)

        elif old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
            _set_robot_intervention_phase(robot, old_phase, new_phase)
            logger.info("Entering correction mode - human teleop control")
            if not supports_feedback and prev_action is not None:
                logger.info("Smooth handover: sliding follower to teleop position")
                obs = robot.get_observation()
                teleop_action = teleop.get_action()
                processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                target = ctx.processors.robot_action_processor((processed, obs))
                follower_smooth_move_to(robot, prev_action, target)

            # unlock the teleop for human control
            if supports_feedback and not continuous_feedback:
                teleop.disable_torque()

        elif old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
            _set_robot_intervention_phase(robot, old_phase, new_phase)
            if supports_feedback and not continuous_feedback:
                teleop.enable_torque()

        elif new_phase == DAggerPhase.AUTONOMOUS:
            _set_robot_intervention_phase(robot, old_phase, new_phase)
            logger.info("Resuming autonomous mode - resetting engine and interpolator")
            interpolator.reset()
            engine.reset()

            # release teleop before resuming the policy
            if supports_feedback and not continuous_feedback:
                teleop.disable_torque()
            return True

        return False

    # ------------------------------------------------------------------
    # Background push (shared by both modes)
    # ------------------------------------------------------------------

    def _background_push(self, _dataset, _cfg) -> None:
        """Defer upload until hardware is down and dataset writers are finalized.

        Network upload and open Parquet/video writers must never contend with
        the hardware loop. Teardown performs the requested push synchronously.
        """
        self._needs_push.set()
        logger.info("Dataset upload queued for after hardware shutdown and finalization")
