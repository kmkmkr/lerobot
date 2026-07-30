#!/usr/bin/env bash

../dora-openarm-data-collection/apptainer/set_camera_manual_exposure.sh 
set -euo pipefail

RIGHT_CAN="${RIGHT_CAN:-can2}"
LEFT_CAN="${LEFT_CAN:-can3}"

LEFT_WRIST_CAMERA="${LEFT_WRIST_CAMERA:-/dev/video8}"
FRONT_CAMERA="${FRONT_CAMERA:-/dev/video6}"
RIGHT_WRIST_CAMERA="${RIGHT_WRIST_CAMERA:-/dev/video0}"
PROFILE="/workspace/openarm_startup_trajectories/task_ready_20260718_180030"
POLICY_PATH="/home/mkj/gitrepo/openarm-bilateral-teleop-lerobot-deploy/checkpoint/pi05_openarm_rel_cam_20260726/checkpoints/060000/pretrained_model"
TASK='Pick up the large green gear and place it in the red recessed area of the blue board.'
DAGGER_DATASET_ROOT="${DAGGER_DATASET_ROOT:-./data/rollout_openarm_dagger_$(date +%Y%m%d_%H%M%S)}"


apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-rollout \
    --robot.deployment_trajectory_profile="$PROFILE" \
    --return_to_initial_position=true \
    --dataset.push_to_hub=false \
    --policy.path="$POLICY_PATH" \
    --policy.dtype=bfloat16 \
    --device=cuda \
    --inference.type=rtc \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port="${LEFT_CAN}" \
    --robot.left_arm_config.side=left \
    --robot.left_arm_config.max_relative_target=10.0 \
    --robot.left_arm_config.use_velocity_and_torque=true \
    --robot.left_arm_config.cameras="{wrist: {type: opencv, index_or_path: ${LEFT_WRIST_CAMERA}, width: 640, height: 480, fps: 30, fourcc: YUYV}}" \
    --robot.right_arm_config.port="${RIGHT_CAN}" \
    --robot.right_arm_config.side=right \
    --robot.right_arm_config.max_relative_target=10.0 \
    --robot.right_arm_config.use_velocity_and_torque=true \
    --robot.right_arm_config.cameras="{wrist: {type: opencv, index_or_path: ${RIGHT_WRIST_CAMERA}, width: 640, height: 480, fps: 30, fourcc: YUYV}}" \
    --robot.cameras="{front: {type: opencv, index_or_path: ${FRONT_CAMERA}, width: 640, height: 480, fps: 30, fourcc: YUYV}}" \
    --task="$TASK" \
    --fps=30 \
    --play_sounds=false \
    --strategy.type=dagger \
    --strategy.record_autonomous=false \
    --strategy.num_episodes=2 \
    --strategy.input_device=keyboard \
    --strategy.resume_blend_duration_s=2.0 \
    --strategy.max_action_velocity=10.0 \
    --teleop.type=bi_openarm_leader \
    --teleop.left_arm_config.port=can1 \
    --teleop.left_arm_config.use_velocity_and_torque=true \
    --teleop.right_arm_config.port=can0 \
    --teleop.right_arm_config.use_velocity_and_torque=true \
    --teleop.id=openarm_v1_leader \
    --dataset.repo_id=${HF_USER:-nkmurst}/rollout_openarm_dagger \
    --dataset.root="$DAGGER_DATASET_ROOT" \
    --dataset.single_task="$TASK" \
    --dataset.fps=30 \
    --dataset.num_episodes=2 \
    --dataset.streaming_encoding=false \
    --dataset.video_encoding_batch_size=2 \
    --dataset.num_image_writer_threads_per_camera=3 \
    --dataset.encoder_threads=1
