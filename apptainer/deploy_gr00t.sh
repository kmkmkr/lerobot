
../dora-openarm-data-collection/apptainer/set_camera_manual_exposure.sh 
RIGHT_CAN="${RIGHT_CAN:-can2}"
LEFT_CAN="${LEFT_CAN:-can3}"

LEFT_WRIST_CAMERA="${LEFT_WRIST_CAMERA:-/dev/video8}"
FRONT_CAMERA="${FRONT_CAMERA:-/dev/video6}"
RIGHT_WRIST_CAMERA="${RIGHT_WRIST_CAMERA:-/dev/video0}"
PROFILE="/workspace/openarm_startup_trajectories/task_ready_20260718_180030"
POLICY_PATH="/home/mkj/gitrepo/openarm-bilateral-teleop-lerobot-deploy/checkpoint/groot_fullft_rel_21576/045000/pretrained_model"

apptainer/openarm_lerobot_exec.sh .venv/bin/lerobot-rollout \
  --robot.deployment_trajectory_profile="$PROFILE" \
  --return_to_initial_position=true \
  --task='Pick up the large green gear and place it in the red recessed area of the blue board.' \
  --dataset.repo_id "nkmurst/rollout_openarm-inference_grootN17" \
  --dataset.push_to_hub=True \
  --strategy.type=base \
  --policy.path="${POLICY_PATH}" \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --device=cuda \
  --inference.type=rtc \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port="${LEFT_CAN}" \
  --robot.left_arm_config.side=left \
  --robot.left_arm_config.max_relative_target=10.0 \
  --robot.left_arm_config.cameras="{wrist: {type: opencv, index_or_path: ${LEFT_WRIST_CAMERA}, width: 640, height: 480, fps: 30, fourcc: YUYV}}" \
  --robot.right_arm_config.port="${RIGHT_CAN}" \
  --robot.right_arm_config.side=right \
  --robot.right_arm_config.max_relative_target=10.0 \
  --robot.right_arm_config.cameras="{wrist: {type: opencv, index_or_path: ${RIGHT_WRIST_CAMERA}, width: 640, height: 480, fps: 30, fourcc: YUYV}}" \
  --robot.cameras="{front: {type: opencv, index_or_path: ${FRONT_CAMERA}, width: 640, height: 480, fps: 30, fourcc: YUYV}}" \
  --task='Pick up the large green gear and place it in the red recessed area of the blue board.' \
  --fps=30 \
  --play_sounds=false \
  --duration=60 \
  --return_to_initial_position=true \
  --strategy.type=episodic \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=0 \
