#!/usr/bin/env python

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

"""Policy deployment engine with pluggable rollout strategies.

``lerobot-rollout`` is the single CLI for running trained policies on
real robots.

Strategies
----------
    --strategy.type=base       Autonomous rollout, no recording
    --strategy.type=sentry     Continuous recording with auto-upload
    --strategy.type=highlight  Ring buffer + keystroke save
    --strategy.type=dagger     Human-in-the-loop (DAgger / RaC)
    --strategy.type=episodic   Episode-oriented recording with reset phases

Inference backends
------------------
    --inference.type=sync      One policy call per control tick (default)
    --inference.type=rtc       Real-Time Chunking for slow VLA models

Usage examples
--------------
::

    # Base mode — quick evaluation with sync inference
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=30

    # Base mode — RTC inference for slow VLAs (Pi0, Pi0.5, SmolVLA)
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/pi0_base \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=10 \\
        --inference.rtc.max_guidance_weight=10.0 \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --task="pick up cube" --duration=60

    # Sentry mode — continuous recording with periodic upload
    lerobot-rollout \\
        --strategy.type=sentry \\
        --strategy.upload_every_n_episodes=5 \\
        --policy.path=lerobot/pi0_base \\
        --inference.type=rtc \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_sentry_data \\
        --dataset.single_task="patrol" --duration=3600

    # Highlight mode — ring buffer, press 's' to save, 'h' to push
    lerobot-rollout \\
        --strategy.type=highlight \\
        --strategy.ring_buffer_seconds=30 \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_highlight_data \\
        --dataset.single_task="pick up cube"

    # DAgger mode — human-in-the-loop corrections only
    lerobot-rollout \\
        --strategy.type=dagger \\
        --strategy.num_episodes=20 \\
        --policy.path=outputs/pretrain/checkpoints/last/pretrained_model \\
        --robot.type=bi_openarm_follower \\
        --teleop.type=openarm_mini \\
        --dataset.repo_id=user/rollout_hil_data \\
        --dataset.single_task="Fold the T-shirt"

    # DAgger mode — continuous recording with RTC inference
    lerobot-rollout \\
        --strategy.type=dagger \\
        --strategy.record_autonomous=true \\
        --strategy.num_episodes=50 \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=10 \\
        --policy.path=user/my_pi0_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --teleop.type=so101_leader \\
        --teleop.port=/dev/ttyACM1 \\
        --dataset.repo_id=user/rollout_dagger_rtc_data \\
        --dataset.single_task="Grasp the block"

    # With Rerun visualization and torch.compile
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=60 \\
        --display_data=true \\
        --use_torch_compile=true

    # Episodic mode — episode-oriented recording with reset phases
    lerobot-rollout \\
        --strategy.type=episodic \\
        --policy.path=user/my_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --teleop.type=so100_leader \\
        --teleop.port=/dev/ttyACM1 \\
        --dataset.repo_id=user/rollout_episodic_data \\
        --dataset.num_episodes=20 \\
        --dataset.single_task="Grab the cube"

    # Resume a previous sentry recording session
    lerobot-rollout \\
        --strategy.type=sentry \\
        --policy.path=user/my_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_sentry_data \\
        --dataset.single_task="patrol" \\
        --resume=true

    # Rollout with custom video encoding parameters
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=60 \\
        --display_data=true \\
        --dataset.rgb_encoder.vcodec=h264 \\
        --dataset.rgb_encoder.preset=fast \\
        --dataset.rgb_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2}

    # Stream to Foxglove instead of Rerun:
    # add --display_mode=foxglove, then connect the Foxglove app to ws://127.0.0.1:8765.
"""

import logging

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

logger = logging.getLogger(__name__)


@parser.wrap()
def rollout(cfg: RolloutConfig):
    """Main entry point for policy deployment."""
    init_logging()

    if cfg.display_data:
        logger.info(
            "Initializing %s visualization (ip=%s, port=%s)",
            cfg.display_mode,
            cfg.display_ip,
            cfg.display_port,
        )
        init_visualization(cfg.display_mode, session_name="rollout", ip=cfg.display_ip, port=cfg.display_port)

    # A rollout signal must interrupt model loading/startup/control immediately.
    # The context and strategy teardown paths convert that BaseException into a
    # no-return-motion fault cleanup.
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False, raise_on_first_signal=True)
    shutdown_event = signal_handler.shutdown_event

    strategy = create_strategy(cfg.strategy)
    ctx = None
    context_handoff = []
    primary_error: BaseException | None = None
    try:
        logger.info("Building rollout context...")
        ctx = build_rollout_context(
            cfg,
            shutdown_event,
            context_ready_callback=context_handoff.append,
        )

        # Keep post-build work inside the lifecycle guard. Once context
        # construction returns, this function owns connected hardware and every
        # BaseException must reach strategy teardown.
        logger.info("Rollout strategy: %s", cfg.strategy.type)
        logger.info(
            "Robot: %s | FPS: %.0f | Duration: %s",
            cfg.robot.type if cfg.robot else "?",
            cfg.fps,
            f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
        )

        strategy.setup(ctx)
        logger.info("Rollout setup complete, starting rollout...")
        strategy.run(ctx)
    except KeyboardInterrupt as error:
        primary_error = error
        if ctx is None and context_handoff:
            ctx = context_handoff[-1]
        if ctx is not None:
            ctx.hardware.control_fault = error
        logger.info("Interrupted by user")
        raise
    except BaseException as error:
        primary_error = error
        if ctx is None and context_handoff:
            ctx = context_handoff[-1]
        if ctx is not None:
            ctx.hardware.control_fault = error
        raise
    finally:
        try:
            if ctx is None and context_handoff:
                ctx = context_handoff[-1]
            if ctx is not None:
                strategy.teardown(ctx)
        except BaseException as teardown_error:
            # This also catches a signal immediately before the teardown call.
            if ctx is None and context_handoff:
                ctx = context_handoff[-1]
            if ctx is None:
                raise
            # A first shutdown signal can arrive while teardown itself is
            # running. Convert it (and any other teardown BaseException)
            # into a no-return control fault, then retry if hardware cleanup
            # did not reach its terminal state.
            if ctx.hardware.control_fault is None:
                ctx.hardware.control_fault = teardown_error
            if not bool(getattr(ctx.hardware, "teardown_complete", False)):
                logger.critical(
                    "Teardown was interrupted before hardware shutdown completed; retrying in fault mode"
                )
                try:
                    strategy.teardown(ctx)
                except BaseException as retry_error:
                    teardown_error.add_note(f"Additional teardown retry error: {retry_error!r}")
                    logger.exception("Fault-mode teardown retry also failed")

            if primary_error is None:
                raise
            if isinstance(primary_error, KeyboardInterrupt):
                teardown_error.add_note(
                    "The rollout was interrupted while this teardown failure occurred"
                )
                raise teardown_error from primary_error
            primary_error.add_note(f"Additional teardown error: {teardown_error!r}")
            logger.exception("Teardown also failed; preserving the original rollout exception")
        if cfg.display_data:
            try:
                shutdown_visualization(cfg.display_mode)
            except BaseException as visualization_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Additional visualization shutdown error: {visualization_error!r}")
                logger.exception("Visualization shutdown failed; preserving the rollout error")

    logger.info("Rollout finished")


def main():
    """CLI entry point for ``lerobot-rollout``."""
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
