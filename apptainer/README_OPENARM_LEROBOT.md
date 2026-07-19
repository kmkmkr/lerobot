# LeRobot OpenArm Apptainer Environment

This setup keeps fixed native dependencies in Apptainer and keeps Python
dependencies editable in the repository-local `.venv` managed by `uv`.

The container includes:

- Ubuntu 24.04 and CUDA 12.8.1 user-space libraries
- Python 3.12, `uv`, build tools, ffmpeg, camera utilities, and CAN tools
- OpenArm CAN packages from `ppa:openarm/main`

It intentionally does not install LeRobot Python dependencies at image build
time. Run `uv sync` inside the container after binding this repository.

In this project, task demonstrations are collected with the native bilateral
controller in `dora-openarm-data-collection`. LeRobot is used for training and
policy deployment. The LeRobot teleoperation and recording commands below are
useful diagnostics or alternatives, but they do not replace the bilateral Dora
collection path.

See [OPENARM_FOLLOWER_CONTROL_COMPARISON.md](OPENARM_FOLLOWER_CONTROL_COMPARISON.md)
for a comparison of follower PD gains, gravity/friction compensation, and joint
limits across bilateral collection, policy inference, and their startup/shutdown
trajectory motions.

## Build

From the LeRobot repository root:

```bash
apptainer build --fakeroot --force apptainer/lerobot_openarm.sif apptainer/lerobot_openarm.def
```

If `--fakeroot` is unavailable:

```bash
sudo apptainer build --force apptainer/lerobot_openarm.sif apptainer/lerobot_openarm.def
```

## Sync Python Dependencies

Install only the Python groups needed for OpenArm data collection:

```bash
./apptainer/openarm_lerobot_exec.sh \
  uv sync --locked --extra core_scripts --extra openarms
```

Add extras when needed, for example:

```bash
# RealSense cameras
./apptainer/openarm_lerobot_exec.sh \
  uv sync --locked --extra core_scripts --extra openarms --extra intelrealsense

# Training dependencies
./apptainer/openarm_lerobot_exec.sh \
  uv sync --locked --extra core_scripts --extra openarms --extra training
```

Quick check:

```bash
./apptainer/openarm_lerobot_exec.sh \
  uv run --no-sync python -c "import torch, can; print(torch.__version__, torch.cuda.is_available()); print(can.__version__)"
```

For hosts without an NVIDIA GPU, disable `--nv`:

```bash
LEROBOT_OPENARM_NV=0 ./apptainer/openarm_lerobot_exec.sh uv run --no-sync lerobot-info --help
```

The wrapper does not bind the host `/dev` tree by default. CAN interfaces are
host network interfaces, not `/dev` nodes, and the tested RealSense D435 is
available through Apptainer without a full `/dev` bind. Verify it with:

```bash
./apptainer/openarm_lerobot_exec.sh \
  uv run --no-sync lerobot-find-cameras realsense
```

Do not set `LEROBOT_OPENARM_BIND_DEV=1` for this RealSense setup. Binding the
entire host `/dev` tree made device nodes such as `/dev/urandom` inaccessible
inside the container and caused PyTorch to fail with
`RuntimeError: Unable to open /dev/urandom`. Use the full-device bind only as a
last-resort diagnostic for hardware that is otherwise invisible.

## CAN Setup

SocketCAN interfaces belong to the host network namespace. Do not run Apptainer
with `--net` for hardware teleoperation.

Configure the four OpenArm CAN FD interfaces:

```bash
sudo ./apptainer/openarm_lerobot_exec.sh \
  openarm-can-configure-socketcan-4-arms -fd
```

Alternatively, use LeRobot's CAN helper after `uv sync`:

```bash
sudo ./apptainer/openarm_lerobot_exec.sh \
  .venv/bin/lerobot-setup-can --mode=setup --interfaces=can0,can1,can2,can3
```

Test motor communication only after supporting every arm safely:

```bash
sudo ./apptainer/openarm_lerobot_exec.sh \
  .venv/bin/lerobot-setup-can --mode=test --interfaces=can0,can1,can2,can3
```

This is not a passive bus probe: `--mode=test` briefly enables and then disables
each motor while checking for its response.

If raw CAN sockets are permitted for your normal user, `sudo` is not required.
When using `sudo`, prefer `.venv/bin/<command>` or `uv run --no-sync` so the
root process does not modify the local Python environment.

## CAN Mapping

The following mapping was confirmed with unilateral LeRobot teleoperation on
this workstation. Recheck it after changing USB-CAN adapters or cables:

| Arm | Leader CAN (`teleop.*`) | Follower CAN (`robot.*`) |
| --- | --- | --- |
| right | `can0` | `can2` |
| left | `can1` | `can3` |

LeRobot does not discover these roles from the motor. A port passed under
`robot.*` is configured as the torque-enabled follower, while a port passed
under `teleop.*` is configured as the manually moved leader. If moving a
physical follower makes a physical leader move, stop immediately and correct
the port assignments.

## OpenArm Motor-Zero Calibration

OpenArm v1/v1.1 uses the zero stored in each Damiao motor as its joint coordinate
frame. Stop every controller, support the arms, and run the official OpenArm
zero-position calibration once per physical arm:

```bash
# Leaders
./apptainer/openarm_lerobot_exec.sh \
  openarm-can-zero-position-calibration --canport can0 --arm-side right_arm
./apptainer/openarm_lerobot_exec.sh \
  openarm-can-zero-position-calibration --canport can1 --arm-side left_arm

# Followers
./apptainer/openarm_lerobot_exec.sh \
  openarm-can-zero-position-calibration --canport can2 --arm-side right_arm
./apptainer/openarm_lerobot_exec.sh \
  openarm-can-zero-position-calibration --canport can3 --arm-side left_arm
```

Do not run `lerobot-calibrate` for a full-size OpenArm v1 leader or follower.
The current integration uses the official motor-zero frame directly, does not
rewrite zero positions during `connect()`, and ignores legacy LeRobot OpenArm
calibration JSON files.

## Deploy A Policy From The Task-Ready Pose

`lerobot-rollout` can reuse the exact CSV profile exported for bilateral
teleoperation. When `robot.deployment_trajectory_profile` is set, the policy
deployment lifecycle is:

1. Connect both followers and move them smoothly to exact motor zero.
2. Replay `left_arm.csv` and `right_arm.csv` together to the task-ready pose.
3. Re-send the final position through the policy-inference gain fields, whose
   defaults equal the trajectory and native bilateral gains, then start inference.
4. At shutdown, blend both followers back to the recorded task-ready pose,
   replay both CSV files in reverse, and finish at exact motor zero.
5. If any shutdown joint differs from its task-ready position by more than the
   warning threshold, ask `yes/no` before moving. A non-interactive session
   chooses `no` and disables the motors in place.

### Test Only The Home-Motion Round Trip

Before starting a full rollout, run the same startup and shutdown motion code
without loading a policy, dataset, or camera. This command performs exactly one
cycle: current pose to exact motor zero, forward CSV replay to the task-ready
pose, a short hold, reverse CSV replay, and return to exact motor zero.

```bash
unset LEROBOT_OPENARM_BIND_DEV
PROFILE=/workspace/openarm_startup_trajectories/task_ready_20260718_180030

./apptainer/openarm_lerobot_exec.sh .venv/bin/python -m \
  lerobot.scripts.lerobot_openarm_home_motion_test \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can3 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can2 \
  --robot.right_arm_config.side=right \
  --robot.id=openarm_v1_follower \
  --robot.deployment_trajectory_profile="$PROFILE" \
  --home_hold_s=3.0
```

The script asks `yes/no` before connecting to the CAN interfaces and moving. It
uses `/dev/tty` when available and otherwise falls back to the interactive
standard input already forwarded by Apptainer; do not bind the complete host
`/dev` tree for this prompt. Use `--confirm_before_motion=false` only for an
intentionally unattended test.
It invokes `prepare_for_policy_deployment()` and
`finish_policy_deployment()` directly, so its gains, compensation, trajectory
validation, interpolation, tracking-error handling, and shutdown warning are
the same as `lerobot-rollout`. It always attempts to disable and disconnect
both arms after success or failure and rejects configurations that disable this
safety behavior. If the process is interrupted, it disables and disconnects
the arms without initiating another recovery motion.

After reinstalling or syncing the editable project, the equivalent console
entry point is `lerobot-openarm-home-motion-test`.

The Apptainer wrapper automatically binds the sibling directory
`../openarm_teleop/config/startup_trajectories` read-only at
`/workspace/openarm_startup_trajectories`. Override the host directory when the
repositories are laid out differently:

```bash
LEROBOT_OPENARM_TRAJECTORY_ROOT=/absolute/path/to/startup_trajectories \
  ./apptainer/openarm_lerobot_exec.sh <command>
```

The following base rollout uses the profile collected on 2026-07-18. Camera
names must match the converted Dora dataset and policy checkpoint. Per-arm
camera key `wrist` becomes `left_wrist` or `right_wrist` in the bimanual
observation:

```bash
POLICY_PATH=/absolute/path/to/pretrained_model
PROFILE=/workspace/openarm_startup_trajectories/task_ready_20260718_180030

./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-rollout \
  --strategy.type=base \
  --policy.path="$POLICY_PATH" \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can3 \
  --robot.left_arm_config.side=left \
  --robot.left_arm_config.cameras='{wrist: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}}' \
  --robot.right_arm_config.port=can2 \
  --robot.right_arm_config.side=right \
  --robot.right_arm_config.cameras='{wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' \
  --robot.cameras='{front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}' \
  --robot.id=openarm_v1_follower \
  --robot.deployment_trajectory_profile="$PROFILE" \
  --return_to_initial_position=true \
  --task="Pick up the red cube and place it in the tray" \
  --duration=60 \
  --display_data=false
```

`trajectory_position_kp` and `trajectory_position_kd` apply only to the zero
and CSV motions. `position_kp` and `position_kd` apply to policy actions. Set
them independently only for temporary diagnostics. All four defaults are kept
equal to the native bilateral follower because it is the canonical controller
configuration, so the normal rollout command should not override them.
Likewise, `max_relative_target` limits policy actions but is deliberately not
applied to the validated zero/CSV deployment trajectory, which already has
joint-limit, velocity, exact-target clipping, and tracking-error checks.

LeRobot also applies the native follower's gravity and friction feed-forward
terms during policy inference and trajectory motions. The wrapper generates a
v10 bimanual URDF from the sibling native `openarm_teleop.sif` on every launch,
then binds it read-only into the LeRobot container. This ensures the Python
gravity model consumes the same OpenArm Description 1.0.4 model as the native
KDL controller. Override the source image or provide an already generated host
URDF when needed:

```bash
LEROBOT_OPENARM_TELEOP_IMAGE=/absolute/path/to/openarm_teleop.sif \
  ./apptainer/openarm_lerobot_exec.sh <command>

LEROBOT_OPENARM_DYNAMICS_URDF=/absolute/path/to/v10_bimanual.urdf \
  ./apptainer/openarm_lerobot_exec.sh <command>
```

`LEROBOT_OPENARM_ENABLE_COMPENSATION=0` disables both gravity and friction
compensation. It is intended only for diagnosis; the normal deployment default
is enabled.

Relevant motion settings and defaults are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `robot.deployment_control_frequency_hz` | `50` | CSV interpolation/control rate |
| `robot.startup_zero_pose_duration_s` | `2.2` | Current pose to exact zero |
| `robot.startup_trajectory_speed` | `1.0` | Forward CSV speed scale |
| `robot.startup_trajectory_blend_s` | `1.0` | Exact zero to first CSV sample |
| `robot.shutdown_task_pose_blend_s` | `10.0` | Final policy pose to task-ready pose |
| `robot.shutdown_replay_speed` | `0.25` | Reverse CSV speed scale |
| `robot.shutdown_zero_transition_s` | `1.0` | Recorded-time transition from first sample to exact zero |
| `robot.shutdown_task_pose_warn_deg` | `28.6479` | Confirmation threshold, equivalent to `0.5 rad` |
| `robot.deployment_tracking_error_deg` | `20.0535` | Abort-and-disable threshold, equivalent to `0.35 rad` |

The zero transition is part of the reverse trajectory and is therefore also
scaled by `shutdown_replay_speed`. Set `--return_to_initial_position=false` to
skip the complete shutdown return and disable the followers in their final
pose.

## Teleoperate Without Recording

This is LeRobot unilateral leader-follower teleoperation. It mirrors leader
joint positions to follower joint targets and does not provide bilateral force
feedback.

```bash
./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-teleoperate \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can3 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can2 \
  --robot.right_arm_config.side=right \
  --robot.id=openarm_v1_follower \
  --teleop.type=bi_openarm_leader \
  --teleop.left_arm_config.port=can1 \
  --teleop.right_arm_config.port=can0 \
  --teleop.id=openarm_v1_leader \
  --fps=30 \
  --display_data=false
```

Start with conservative limits if this is a first run:

```bash
  --robot.left_arm_config.max_relative_target=1.0 \
  --robot.right_arm_config.max_relative_target=1.0
```

Align each leader and follower in a similar safe pose before connecting. The
relative-target limit is useful for a first motion test, but it is not a
zero-position or pose-alignment procedure.

## Record A LeRobot Dataset

Use `--dataset.push_to_hub=false` while testing. LeRobot stamps the repository
ID with a timestamp when creating a new local dataset. The following command was
validated with an Intel RealSense D435 at 640x480 and 30 FPS. Replace the camera
ID with the `Id` printed by `lerobot-find-cameras realsense` if the camera is
changed.

```bash
ROOT="./data/openarm_v1_bimanual_$(date +%Y%m%d_%H%M%S)"

./apptainer/openarm_lerobot_exec.sh \
  .venv/bin/lerobot-record \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can3 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can2 \
  --robot.right_arm_config.side=right \
  --robot.id=openarm_v1_follower \
  --robot.cameras='{front: {type: intelrealsense, serial_number_or_name: "152222070204", width: 640, height: 480, fps: 30, use_depth: false}}' \
  --teleop.type=bi_openarm_leader \
  --teleop.left_arm_config.port=can1 \
  --teleop.right_arm_config.port=can0 \
  --teleop.id=openarm_v1_leader \
  --dataset.repo_id="${HF_USER:-local}/openarm_v1_bimanual" \
  --dataset.root="$ROOT" \
  --dataset.single_task="Pick up the red cube and place it in the tray" \
  --dataset.fps=30 \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=false \
  --play_sounds=false
```

The successful RealSense check and recording command do not use
`LEROBOT_OPENARM_BIND_DEV=1`.

To collect joint-only data without a camera, omit `--robot.cameras`. For an
OpenCV camera, replace the RealSense configuration, for example:

```bash
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}'
```

## Notes On Bilateral Control

OpenArm's native bilateral controller provides two-way force feedback. LeRobot's
current OpenArm leader implementation exposes no `feedback_features`, and
`send_feedback` is not implemented for `openarm_leader`. For LeRobot-format data
collection, use the LeRobot unilateral path above. If force-feedback teleop is
required during collection, bridge or port the OpenArm bilateral controller so
the commanded/sent follower actions and observations are also written through
`LeRobotDataset`.
