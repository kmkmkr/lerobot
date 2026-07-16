# LeRobot OpenArm Apptainer Environment

This setup keeps fixed native dependencies in Apptainer and keeps Python
dependencies editable in the repository-local `.venv` managed by `uv`.

The container includes:

- Ubuntu 24.04 and CUDA 12.8.1 user-space libraries
- Python 3.12, `uv`, build tools, ffmpeg, camera utilities, and CAN tools
- OpenArm CAN packages from `ppa:openarm/main`

It intentionally does not install LeRobot Python dependencies at image build
time. Run `uv sync` inside the container after binding this repository.

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
