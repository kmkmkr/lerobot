# LeRobot OpenArm Apptainer Environment

This setup keeps fixed native dependencies in Apptainer and keeps Python
dependencies editable in the repository-local `.venv` managed by `uv`.

The container includes:

- Ubuntu 24.04 and CUDA 12.8.1 user-space libraries
- Python 3.12, `uv`, build tools, ffmpeg, camera utilities, and CAN tools
- OpenArm CAN packages from `ppa:openarm/main`

It intentionally does not install LeRobot Python dependencies at image build
time. Run `uv sync` inside the container after binding this repository.

In this project, the native bilateral controller in
`dora-openarm-data-collection` remains the canonical collection path. LeRobot
is used for training and policy deployment, and its teleoperation/recording
commands below now provide bilateral OpenArm leader feedback using the same
role-specific PD gains and compensation defaults. The LeRobot path remains a
lower-frequency diagnostic or alternative to the native controller.

See [OPENARM_FOLLOWER_CONTROL_COMPARISON.md](OPENARM_FOLLOWER_CONTROL_COMPARISON.md)
for a comparison of leader/follower PD gains, gravity/friction compensation,
and joint limits across bilateral collection, policy inference, and their
startup/shutdown trajectory motions.

### Native 1 kHz bridge for DAgger

For full-size OpenArm leader intervention, the superproject also provides an
out-of-tree `lerobot-openarm-rt` plugin and a fail-closed launcher. In this
mode, the two native `bilateral_control` processes retain exclusive ownership
of all four CAN interfaces and run each left/right leader-follower control pair
at the native 1 kHz period. LeRobot communicates over two local Unix sockets
and remains responsible for policy inference, 30 Hz cameras, phase events, and
dataset recording. It never opens an OpenArm CAN interface in this mode.

The native processes implement the phase behavior directly:

- `autonomous`: leader follows measured follower feedback; follower tracks a
  rate-limited policy target and holds its measured pose until the first fresh
  target after every resume.
- `paused`: follower holds its measured pose and no policy target is accepted.
- `correcting`: follower follows the measured leader inside the 1 kHz loop;
  the native role-specific bilateral gains and compensation remain active.
- `fault`: a healthy native side rejects new client motion targets and enters
  measured hold. A native control-loop, measurement, or motor-command failure
  may disable torque and exit; no fault initiates an automatic return trajectory.

This avoids implementing a 1 kHz torque loop in Python and keeps the native
Dora controller as the source of truth. Only a small, generic opt-in
`set_intervention_phase(old_phase, new_phase)` notification hook lives in the
LeRobot checkout; the production RT path requires its version-2 transition and
loop-exit PAUSED boundaries. Plugin discovery, the socket protocol, and
OpenArm-specific state handling stay outside LeRobot. This separation is
intended to reduce conflicts when merging future upstream LeRobot updates.

For current checkpoint metadata and converted Dora datasets, the launcher
explicitly selects `robot.torque_observation_mode=zero` for the legacy raw
channel. `measured` exposes native motor torque in N.m at the raw Robot API.
The current rollout feature aggregation still routes only `.pos` and cameras
to the policy and dataset, and the supplied GR00T pack step consumes the 16-D
position state. Persisting velocity or measured torque requires a deliberately
separated processor/context schema and a consistently trained checkpoint. See
the superproject guide at
[`../../docs/OPENARM_RT_BRIDGE.md`](../../docs/OPENARM_RT_BRIDGE.md) before
hardware use. The bridge implementation has software tests, but still requires
a supported-arm, low-risk physical validation before normal operation.

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
under `teleop.*` is configured as the torque-enabled bilateral leader. Verify
all four adapter-to-arm assignments before connecting; a swapped role or side
can send feedback to the wrong physical device.

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
teleoperation. For a non-intervention strategy, when
`robot.deployment_trajectory_profile` is set, the policy deployment lifecycle
is:

1. Connect both followers and move them smoothly to exact motor zero.
2. Replay `left_arm.csv` and `right_arm.csv` together to the task-ready pose.
3. Re-send the final position through the policy-inference gain fields, whose
   defaults equal the trajectory and native bilateral gains, then start inference.
4. At shutdown, blend both followers back to the recorded task-ready pose,
   replay both CSV files in reverse, and finish at exact motor zero.
5. If any shutdown joint differs from its task-ready position by more than the
   warning threshold, ask `yes/no` before moving. A non-interactive session
   chooses `no` and disables the motors in place.
6. If the shutdown return itself fails, stop the motion and keep both followers
   torque-enabled at their current pose by default. The subsequent cleanup
   closes CAN and cameras without disabling torque, preventing the arms from
   free-falling.

OpenArm DAgger extends this lifecycle to both leaders as described below. The
standalone home-motion test remains follower-only.

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
the same as `lerobot-rollout`. It requires torque-disable-on-disconnect at the
start. After normal completion, startup failure, or interruption, it disables
and disconnects both arms. A shutdown-return failure is handled differently:
it refreshes a current-position hold command and disconnects CAN without
disabling torque, so the arms remain energized after the command exits. Support
both arms before removing power or manually disabling torque.

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

短時間検証
```
cd ~/gitrepo/openarm-bilateral-teleop-lerobot-deploy/lerobot

POLICY_PATH=/home/mkj/gitrepo/openarm-bilateral-teleop-lerobot-deploy/checkpoint/pi05_lora_21437/pretrained_model
PROFILE=/workspace/openarm_startup_trajectories/task_ready_20260718_180030

./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --policy.n_action_steps=3 \
  --device=cuda \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can3 \
  --robot.left_arm_config.side=left \
  --robot.left_arm_config.max_relative_target=2.0 \
  --robot.left_arm_config.cameras='{wrist: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}}' \
  --robot.right_arm_config.port=can2 \
  --robot.right_arm_config.side=right \
  --robot.right_arm_config.max_relative_target=2.0 \
  --robot.right_arm_config.cameras='{wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' \
  --robot.cameras='{front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}' \
  --robot.id=openarm_v1_follower \
  --robot.deployment_trajectory_profile="$PROFILE" \
  --return_to_initial_position=true \
  --task="Pick up the large green gear and place it in the red recessed area of the blue board." \
  --fps=30 \
  --duration=1.0 \
  --display_data=true \
  --play_sounds=false
```

`trajectory_position_kp` and `trajectory_position_kd` apply only to the zero
and CSV motions. `position_kp` and `position_kd` apply to policy actions. Set
them independently only for temporary diagnostics. All four defaults are kept
equal to the native bilateral follower because it is the canonical controller
configuration, so the normal rollout command should not override them.
Likewise, `max_relative_target` limits policy actions but is deliberately not
applied to the validated zero/CSV deployment trajectory, which already has
joint-limit, velocity, exact-target clipping, and tracking-error checks.

## Run DAgger With OpenArm Bilateral Intervention

DAgger starts in autonomous policy mode. For a keyboard-controlled session,
press Space to pause policy execution, Tab to start or finish a human
correction, Space from the paused phase to resume the policy, and Escape to
stop the session. Enter queues a Hub upload for after hardware shutdown and
dataset finalization. Space after finishing a correction means
`PAUSED -> AUTONOMOUS`; it does not start a new episode. The default
`record_autonomous=false` stores each Tab-to-Tab correction window as one
episode. Episode persistence runs in a single background worker while PAUSED
feedback and measured-position hold commands continue. If Space or Tab is
pressed before persistence completes, the request remains pending. With 10
target episodes, `video_encoding_batch_size=11` defers AV1 encoding to
`dataset.finalize()`, after hardware has been secured and disconnected. On
Escape, duration expiry, or reaching the target episode count, hardware
cleanup begins without waiting for a pending save; only after disconnect does
teardown wait for the saver. A persistence or teardown error suppresses the
final Hub push so a partial dataset is not published.
`record_autonomous=true` is intentionally rejected for both single-arm and
bimanual OpenArm until continuous-mode persistence is fully removed from the
hardware control loop.

Start from the full `lerobot-rollout` command above, retain its policy,
follower, camera, deployment-trajectory, task, and runtime arguments, replace
`--strategy.type=base`, and add the following arguments:

```bash
ROOT="./data/rollout_openarm_dagger_$(date +%Y%m%d_%H%M%S)"
```

The CLI fragment to append is:

```text
--strategy.type=dagger
--strategy.record_autonomous=false
--strategy.num_episodes=10
--strategy.input_device=keyboard
--strategy.resume_blend_duration_s=2.0
--strategy.max_action_velocity=10.0
--robot.left_arm_config.use_velocity_and_torque=true
--robot.right_arm_config.use_velocity_and_torque=true
--teleop.type=bi_openarm_leader
--teleop.left_arm_config.port=can1
--teleop.left_arm_config.use_velocity_and_torque=true
--teleop.right_arm_config.port=can0
--teleop.right_arm_config.use_velocity_and_torque=true
--teleop.id=openarm_v1_leader
--dataset.repo_id=${HF_USER:-local}/rollout_openarm_dagger
--dataset.root=$ROOT
--dataset.single_task="Pick up the red cube and place it in the tray"
--dataset.fps=30
--dataset.num_episodes=10
--dataset.push_to_hub=false
--dataset.streaming_encoding=false
--dataset.video_encoding_batch_size=11
--dataset.num_image_writer_threads_per_camera=2
--dataset.encoder_threads=1
```

For `openarm_leader` and `bi_openarm_leader`, DAgger sends the measured
follower observation back to the leader in autonomous, paused, and correcting
phases. It does not run the generic blocking leader handover or disable/re-enable
leader torque at phase boundaries. Therefore the native-aligned leader PD,
gravity compensation, friction compensation, and bilateral force response stay
active both while the policy drives the follower and while the person moves the
leader to correct it. The follower continues to use its native-aligned PD and
compensation for policy and correction actions.

Enable `use_velocity_and_torque=true` on both followers and both leaders as in
the fragment above. Measured follower velocity then reaches the leader feedback
controller, and measured leader velocity reaches the follower MIT target. If it
is omitted, target velocity defaults to zero and the bilateral leader can feel
unnecessarily heavy while the person moves it.

For `bi_openarm_follower`, OpenArm DAgger requires
`robot.deployment_trajectory_profile`. It connects both followers and both
leaders before startup motion. Each role blends from its own measured pose to
exact motor zero, then all four devices receive the same side-specific CSV
targets through the task-ready pose. Followers use their trajectory PD and
compensation; leaders use the native-aligned leader PD and compensation. At the
handover to inference, the measured follower pose becomes the leader reference,
so continuous bilateral feedback starts without a reference jump.

Every start or `PAUSED -> AUTONOMOUS` transition resets the RTC queue and
invalidates the previous observation generation. The main loop publishes a
fresh measured observation before resuming inference; results started before a
pause/reset are discarded. The first new chunk is consumed from step zero, and
policy targets are blended from the measured handover pose for
`resume_blend_duration_s` while `max_action_velocity` limits every control-tick
step. An empty inference queue keeps sending a zero-velocity position hold.

With the default `return_to_initial_position=true`, a normal orderly shutdown
blends all four devices back to the task-ready pose and replays the same
reversed CSV to exact motor zero before disconnecting. If a control, feedback,
or recording fault occurs, that return trajectory is prohibited: all four
devices are secured at freshly measured positions before CAN disconnect and
dataset finalization. Only a device whose position command, torque-enable, and
post-enable command all succeed is left holding across disconnect; a failed
device is explicitly disabled and reported. A startup replay error disables all
four devices. A failure during an otherwise normal shutdown return uses the
same current-position hold. The warning confirmation evaluates both leader and
follower joints.

Do not set `teleop.{left,right}_arm_config.manual_control=true` for this DAgger
mode: that is a torque-disabled diagnostic and deliberately opts out of
continuous bilateral feedback and coordinated replay. Support all four arms,
verify that each device has a collision-free path from its current pose through
motor zero, test at conservative follower relative-action limits, and perform a
fresh physical validation before collecting task data.

LeRobot also applies the native role-specific gravity and friction feed-forward
terms: to leaders during bilateral teleoperation/recording, and to followers
during teleoperation, policy inference, and trajectory motions. The wrapper
generates a v10 bimanual URDF from the sibling native `openarm_teleop.sif` on
every launch, then binds it read-only into the LeRobot container. This ensures
the Python gravity model consumes the same OpenArm Description 1.0.4 model as
the native KDL controller. Override the source image or provide an already
generated host URDF when needed:

```bash
LEROBOT_OPENARM_TELEOP_IMAGE=/absolute/path/to/openarm_teleop.sif \
  ./apptainer/openarm_lerobot_exec.sh <command>

LEROBOT_OPENARM_DYNAMICS_URDF=/absolute/path/to/v10_bimanual.urdf \
  ./apptainer/openarm_lerobot_exec.sh <command>
```

`LEROBOT_OPENARM_ENABLE_COMPENSATION=0` disables both gravity and friction
compensation for both leaders and followers. It is intended only for diagnosis;
the normal bilateral and deployment defaults are enabled.

Relevant motion settings and defaults are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `strategy.resume_blend_duration_s` | `2.0` | Fresh measured pose to policy target smoothstep duration |
| `strategy.max_action_velocity` | `None` (`dagger.sh`: `10.0`) | Per-second autonomous action limit; degrees/s for OpenArm positions |
| `robot.{left,right}_arm_config.use_velocity_and_torque` | `false` (`dagger.sh`: `true`) | Expose follower velocity for leader feedback |
| `teleop.{left,right}_arm_config.use_velocity_and_torque` | `false` (`dagger.sh`: `true`) | Send measured leader velocity to follower MIT control |
| `teleop.{left,right}_arm_config.feedback_position_limit_tolerance_deg` | `1.5` | Clamp only small measured boundary overshoot; larger values fault |
| `robot.deployment_control_frequency_hz` | `50` | CSV interpolation/control rate |
| `robot.startup_zero_pose_duration_s` | `2.2` | Current pose to exact zero |
| `robot.startup_trajectory_speed` | `1.0` | Forward CSV speed scale |
| `robot.startup_trajectory_blend_s` | `1.0` | Exact zero to first CSV sample |
| `robot.shutdown_task_pose_blend_s` | `10.0` | Final policy pose to task-ready pose |
| `robot.shutdown_replay_speed` | `0.25` | Reverse CSV speed scale |
| `robot.shutdown_zero_transition_s` | `1.0` | Recorded-time transition from first sample to exact zero |
| `robot.shutdown_task_pose_warn_deg` | `28.6479` | Confirmation threshold, equivalent to `0.5 rad` |
| `robot.deployment_tracking_error_deg` | `20.0535` | Motion-abort threshold, equivalent to `0.35 rad` |
| `robot.deployment_start_limit_tolerance_deg` | `1.5` | Clamp the observed small gripper sensor overshoot (up to 1.5 deg); larger deviations fault |
| `robot.hold_position_on_shutdown_error` | `true` | Keep torque enabled at the current pose if the shutdown return fails |

The zero transition is part of the reverse trajectory and is therefore also
scaled by `shutdown_replay_speed`. Set `--return_to_initial_position=false` to
skip the complete shutdown return and disable connected arms in their final
pose; in OpenArm DAgger this means both leaders and both followers. Set
`--robot.hold_position_on_shutdown_error=false` only when the former
abort-and-disable behavior is preferable to holding after a shutdown-return
error.

## Teleoperate Without Recording

This is LeRobot bilateral leader-follower teleoperation. Each cycle reads the
follower observation, reads the leader action, sends the follower pose back to
the leader with the native leader PD/compensation defaults, and sends the leader
pose to the follower with the native follower defaults.

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

Support all arms and align each leader/follower pair in a similar safe pose
before connecting. Unlike the native Dora startup flow, this command does not
move all devices through the synchronized zero/CSV startup trajectory. The
relative-target limit is useful for a first follower motion test, but it is not
a zero-position or pose-alignment procedure.

For a torque-disabled leader diagnostic, set both
`--teleop.{left,right}_arm_config.manual_control=true`. This removes the leader
feedback features, PD, and compensation; it is not bilateral operation.

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

`openarm_leader` and `bi_openarm_leader` expose follower position feedback and
issue leader MIT commands with the effective native leader `Kp`, `Kd`, gravity,
and friction values. In bimanual mode, a feedback/compensation failure disables
both leaders. The dynamics URDF for both sides is validated before either
leader is enabled.

The standard LeRobot OpenArm dataset commands are position-only by default, so
the MIT reference velocity is `0`; if a follower observation supplies `.vel`,
`send_feedback` uses it. The loop also runs at the configured dataset/teleop
FPS (typically 30 Hz), not the native controller's high-rate control loop.
Consequently the native Dora controller remains the reference for collection
feel and hardware validation even though the default gains and feed-forward
equations are synchronized numerically.
