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

"""Rollout strategy ABC and shared action-dispatch helper."""

from __future__ import annotations

import abc
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data

from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)


def _device_has_live_connection(device) -> bool:
    """Detect complete and partial hardware connections during teardown."""
    if device is None:
        return False
    any_arm_connected = getattr(type(device), "any_arm_connected", None)
    if isinstance(any_arm_connected, property):
        try:
            if bool(device.any_arm_connected):
                return True
        except Exception:
            pass
    try:
        if bool(device.is_connected):
            return True
    except Exception:
        pass
    bus = getattr(device, "bus", None)
    try:
        if bus is not None and bool(bus.is_connected):
            return True
    except Exception:
        pass
    cameras = getattr(device, "cameras", {})
    try:
        return any(bool(camera.is_connected) for camera in cameras.values())
    except Exception:
        return False


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Attach the inference engine and action interpolator, then start the backend.

        Creates an :class:`ActionInterpolator` from the config's
        ``interpolation_multiplier`` and starts the inference engine.
        Call this from ``setup()`` so strategies share identical
        initialisation without duplicating code.
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        logger.info("Starting inference engine...")
        self._engine.reset()
        self._engine.start()
        self._warmup_flushed = False
        self._cached_obs_processed = None
        logger.info("Inference engine started")

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        """Run the observation processor and notify the engine — throttled to policy ticks.

        Callers are responsible for calling ``robot.get_observation()`` every loop
        iteration so ``obs_raw`` stays fresh for the action post-processor.  This
        helper gates only the comparatively expensive bits — the processor pipeline
        and ``engine.notify_observation`` — to fire when the interpolator signals
        it needs a new action (once per ``interpolation_multiplier`` ticks).  On
        interpolated ticks the cached ``obs_processed`` is reused.

        With ``interpolation_multiplier == 1`` this is equivalent to the unthrottled
        path: ``needs_new_action()`` is True every tick.

        The cache is implicitly invalidated whenever ``interpolator.reset()`` is
        called (warmup completion, DAgger phase transitions back to AUTONOMOUS),
        because reset makes ``needs_new_action()`` return True on the next call.
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(
        self,
        use_torch_compile: bool,
        loop_start: float,
        control_interval: float,
        *,
        resume_after_reset: bool = True,
    ) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            self._warmup_flushed = True
            if resume_after_reset:
                engine.resume()
        return False

    def _secure_hardware_after_fault(self, robot, teleop) -> None:
        """Use the strongest available no-motion safety hook."""
        secure_intervention = getattr(robot, "secure_intervention_after_fault", None)
        if (
            self.config.type == "dagger"
            and teleop is not None
            and callable(secure_intervention)
        ):
            secure_intervention(teleop)
            return

        secure_after_fault = getattr(robot, "secure_after_fault", None)
        if callable(secure_after_fault):
            secure_after_fault()
            return

        logger.warning("Robot has no specialized fault-hold hook; disconnecting in place.")

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """Secure and disconnect hardware before any dataset finalization.

        A control fault prohibits automatic return motion. Specialized robots
        may hold their measured pose; generic hardware disconnects in place.
        """
        if hw.teardown_complete:
            return

        engine_stop_error: BaseException | None = None
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            try:
                self._engine.stop()
            except BaseException as error:
                engine_stop_error = error
                logger.exception("Failed to stop inference engine; continuing hardware shutdown")
        if engine_stop_error is not None and hw.control_fault is None:
            hw.control_fault = engine_stop_error

        robot = hw.robot_wrapper.inner
        teleop = hw.teleop
        robot_connected = _device_has_live_connection(robot)
        teleop_connected = _device_has_live_connection(teleop)
        disconnect_complete = True
        if (
            self.config.type == "dagger"
            and robot_connected != teleop_connected
            and hw.control_fault is None
        ):
            hw.control_fault = RuntimeError(
                "DAgger hardware connectivity became asymmetric during teardown"
            )
            logger.critical(
                "Follower/teleoperator connectivity mismatch; prohibiting coordinated return motion"
            )
        try:
            # A follower may already have closed its CAN buses while one or
            # both leaders are still live. The coordinated OpenArm fault hook
            # remains responsible for every device that is still reachable;
            # do not gate it solely on the follower connection state.
            if (
                not robot_connected
                and teleop_connected
                and hw.control_fault is not None
                and self.config.type == "dagger"
            ):
                logger.critical(
                    "Control fault detected with only the teleoperator still connected; "
                    "securing the remaining intervention hardware in place."
                )
                self._secure_hardware_after_fault(robot, teleop)
            if robot_connected:
                try:
                    if hw.control_fault is not None:
                        logger.critical(
                            "Control fault detected (%s); automatic return motion is prohibited. "
                            "Securing hardware at its current pose.",
                            type(hw.control_fault).__name__,
                        )
                        self._secure_hardware_after_fault(robot, teleop)
                    elif return_to_initial_position:
                        finish_intervention_deployment = getattr(
                            robot, "finish_intervention_deployment", None
                        )
                        if (
                            self.config.type == "dagger"
                            and teleop is not None
                            and callable(finish_intervention_deployment)
                        ):
                            specialized_return = bool(finish_intervention_deployment(teleop))
                        else:
                            finish_policy_deployment = getattr(robot, "finish_policy_deployment", None)
                            specialized_return = (
                                bool(finish_policy_deployment())
                                if callable(finish_policy_deployment)
                                else False
                            )
                        if not specialized_return and hw.initial_position:
                            logger.info("Returning robot to initial position before shutdown...")
                            self._return_to_initial_position(hw)
                    else:
                        logger.info(
                            "Skipping shutdown return motion (disabled by config); leaving robot in final pose."
                        )
                except BaseException as error:
                    if hw.control_fault is None:
                        hw.control_fault = error
                    logger.critical(
                        "Hardware shutdown motion/control was interrupted; "
                        "securing the current pose before disconnect"
                    )
                    try:
                        self._secure_hardware_after_fault(robot, teleop)
                    except BaseException as secure_error:
                        error.add_note(
                            f"Additional current-pose safety hook error: {secure_error!r}"
                        )
                        logger.exception(
                            "Current-pose safety hook also failed; continuing disconnect"
                        )
                    raise
                finally:
                    logger.info("Disconnecting robot...")
                    try:
                        robot.disconnect()
                    except BaseException:
                        disconnect_complete = False
                        raise
        finally:
            try:
                teleop_connected = _device_has_live_connection(teleop)
                if teleop_connected:
                    logger.info("Disconnecting teleoperator...")
                    try:
                        teleop.disconnect()
                    except BaseException:
                        disconnect_complete = False
                        raise
            finally:
                # A caller may safely retry when an asynchronous interruption
                # or strict disconnect failure left a device unresolved.
                hw.teardown_complete = disconnect_complete

        if engine_stop_error is not None:
            raise engine_stop_error

    @staticmethod
    def _return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> None:
        """Smoothly interpolate the robot back to its initial position."""
        robot = hw.robot_wrapper
        target = hw.initial_position
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {}
                for k in current_pos:
                    interp[k] = current_pos[k] * (1 - t) + target[k] * t
                robot.send_action(interp)
                precise_sleep(1 / fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log observation/action telemetry to the visualization backend if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_visualization_data(
            cfg.display_mode,
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires."""

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: save dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """Push dataset to hub, skipping if no episodes have been saved.

    Returns ``True`` if the push was attempted, ``False`` if skipped.
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """Conservatively estimate how many seconds of video will exceed *target_size_mb*.

    Each camera produces its own video file, so the episode duration is
    driven by the **slowest** camera to fill ``target_size_mb`` — i.e.
    the one with the fewest pixels per frame (lowest bitrate).

    Uses a deliberately **low** bits-per-pixel estimate so the computed
    duration is *longer* than reality.  By the time the timer fires the
    actual video file is guaranteed to have crossed the target size,
    which aligns episode boundaries with the dataset's video-file
    chunking — each ``push_to_hub`` uploads complete files rather than
    re-uploading a still-growing one.

    The estimate ignores codec-specific settings (CRF, preset) on purpose:
    we only need a rough lower bound on bitrate, not a precise prediction.

    Falls back to 300 s (5 min) when no video features are present.
    """
    # 0.1 bits-per-pixel is a *low* estimate for CRF-30 streaming video of
    # robot footage (real-world is typically 0.1 – 0.3 bpp).  Under-
    # estimating the bitrate over-estimates the time → the episode will be
    # *larger* than target_size_mb when we save, which is what we want.
    conservative_bpp = 0.1

    # Collect per-camera pixel counts — each camera has its own video file.
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C) — bits-per-pixel is a per-spatial-pixel metric,
            # so we exclude the channel dimension from the count.
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # Use the smallest camera: it produces the lowest bitrate and therefore
    # takes the longest to reach the target — the conservative choice.
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # Guard against division by zero just in case
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# Shared action-dispatch helper
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
    action_filter: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys

    if interpolator.needs_new_action():
        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    if action_filter is not None:
        action_dict = action_filter(action_dict)
    processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
    ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict
