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

The wrapper does not bind the host `/dev` tree by default because that can
interfere with `uv`'s Python interpreter checks on some Apptainer setups. CAN
interfaces are host network interfaces, not `/dev` nodes. If a USB camera or
other device is not visible, opt in per command:

```bash
LEROBOT_OPENARM_BIND_DEV=1 ./apptainer/openarm_lerobot_exec.sh v4l2-ctl --list-devices
```

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

Test communication:

```bash
sudo ./apptainer/openarm_lerobot_exec.sh \
  .venv/bin/lerobot-setup-can --mode=test --interfaces=can0,can1,can2,can3
```

If raw CAN sockets are permitted for your normal user, `sudo` is not required.
When using `sudo`, prefer `.venv/bin/<command>` or `uv run --no-sync` so the
root process does not modify the local Python environment.

## CAN Mapping

The existing bilateral setup in `../openarm_teleop` documents `right: leader
can0 / follower can2` and `left: leader can1 / follower can3`. On this
LeRobot setup, the first hardware test showed those functional roles are
reversed. Use this observed mapping unless the cables are changed:

| Arm | Leader CAN (`teleop.*`) | Follower CAN (`robot.*`) |
| --- | --- | --- |
| right | `can2` | `can0` |
| left | `can3` | `can1` |

If moving a follower arm makes a leader arm move, swap the `robot.*` and
`teleop.*` ports.

## Calibrate

Calibrate each physical arm once. The IDs below keep calibration files stable
between runs.

```bash
# Followers
sudo ./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-calibrate \
  --robot.type=openarm_follower \
  --robot.port=can1 \
  --robot.side=left \
  --robot.id=openarm_v1_follower_left

sudo ./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-calibrate \
  --robot.type=openarm_follower \
  --robot.port=can0 \
  --robot.side=right \
  --robot.id=openarm_v1_follower_right

# Leaders
sudo ./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-calibrate \
  --teleop.type=openarm_leader \
  --teleop.port=can3 \
  --teleop.id=openarm_v1_leader_left

sudo ./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-calibrate \
  --teleop.type=openarm_leader \
  --teleop.port=can2 \
  --teleop.id=openarm_v1_leader_right
```

If calibration files were created while the ports were reversed, rerun
calibration with the ports above and choose recalibration when prompted.

## Teleoperate Without Recording

This is LeRobot unilateral leader-follower teleoperation. It mirrors leader
joint positions to follower joint targets and does not provide bilateral force
feedback.

```bash
sudo ./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-teleoperate \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can1 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can0 \
  --robot.right_arm_config.side=right \
  --robot.id=openarm_v1_follower \
  --teleop.type=bi_openarm_leader \
  --teleop.left_arm_config.port=can3 \
  --teleop.right_arm_config.port=can2 \
  --teleop.id=openarm_v1_leader \
  --fps=30 \
  --display_data=false
```

Start with conservative limits if this is a first run:

```bash
  --robot.left_arm_config.max_relative_target=5 \
  --robot.right_arm_config.max_relative_target=5
```

## Record A LeRobot Dataset

Use `--dataset.push_to_hub=false` while testing. LeRobot stamps the repository
ID with a timestamp when creating a new local dataset.

```bash
sudo ./apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-record \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can1 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can0 \
  --robot.right_arm_config.side=right \
  --robot.id=openarm_v1_follower \
  --teleop.type=bi_openarm_leader \
  --teleop.left_arm_config.port=can3 \
  --teleop.right_arm_config.port=can2 \
  --teleop.id=openarm_v1_leader \
  --dataset.repo_id=${HF_USER:-local}/openarm_v1_bimanual_test \
  --dataset.root=./data/openarm_v1_bimanual_test \
  --dataset.single_task="Teleoperate OpenArm v1 bimanual setup" \
  --dataset.fps=30 \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=false \
  --play_sounds=false
```

Add cameras to the robot config when ready, for example:

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
