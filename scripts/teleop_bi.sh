LEFT_L_PORT=/dev/ttyACM2

RIGHT_L_PORT=/dev/ttyACM3

LEFT_F_PORT=/dev/ttyACM1

RIGHT_F_PORT=/dev/ttyACM0

REPO_ID="nkmurst/bi_so101_test"
SINGLE_TASK="South America"
DATASET_NUM_EPISODES=50
DATASET_EPISODE_TIME_S=180
DATASET_RESET_TIME_S=3

TOP_CAM_ID=/dev/video0
LEFT_ARM_CAM_ID=/dev/video2
RIGHT_ARM_CAM_ID=/dev/video4
TOP_CAM_WIDTH=640
TOP_CAM_HEIGHT=480
TOP_CAM_FPS=24
ARM_CAM_WIDTH=$TOP_CAM_WIDTH
ARM_CAM_HEIGHT=$TOP_CAM_HEIGHT
LEFT_ARM_CAM_FPS=180
RIGHT_ARM_CAM_FPS=180
FPS=$TOP_CAM_FPS

TOP_FORMAT=MJPG
LEFT_ARM_FORMAT=MJPG
RIGHT_ARM_FORMAT=MJPG

CAMERA_LEFT="{top: {type: opencv, index_or_path: $TOP_CAM_ID, width: $TOP_CAM_WIDTH, height: $TOP_CAM_HEIGHT, fps: $TOP_CAM_FPS}, left: {type: opencv, index_or_path: $LEFT_ARM_CAM_ID, width: $ARM_CAM_WIDTH, height: $ARM_CAM_HEIGHT, fps: $LEFT_ARM_CAM_FPS, fourcc: $LEFT_ARM_FORMAT}}"
CAMERA_RIGHT="{top: {type: opencv, index_or_path: $TOP_CAM_ID, width: $TOP_CAM_WIDTH, height: $TOP_CAM_HEIGHT, fps: $TOP_CAM_FPS}, right: {type: opencv, index_or_path: $RIGHT_ARM_CAM_ID, width: $ARM_CAM_WIDTH, height: $ARM_CAM_HEIGHT, fps: $RIGHT_ARM_CAM_FPS, fourcc: $RIGHT_ARM_FORMAT}}"

CAMERA_ALL="{top: {type: opencv, index_or_path: $TOP_CAM_ID, width: $TOP_CAM_WIDTH, height: $TOP_CAM_HEIGHT, fps: $TOP_CAM_FPS, fourcc: $TOP_FORMAT}, left: {type: opencv, index_or_path: $LEFT_ARM_CAM_ID, width: $ARM_CAM_WIDTH, height: $ARM_CAM_HEIGHT, fps: $LEFT_ARM_CAM_FPS, fourcc: $LEFT_ARM_FORMAT}, right: {type: opencv, index_or_path: $RIGHT_ARM_CAM_ID, width: $ARM_CAM_WIDTH, height: $ARM_CAM_HEIGHT, fps: $RIGHT_ARM_CAM_FPS, fourcc: $RIGHT_ARM_FORMAT}}"

# Rerunビューアの接続先 (コンテナ外でrerunを起動した場合は127.0.0.1:9876を指定)
# 未指定の場合はコンテナ内でrerun GUIを起動しようとするため、コンテナ環境では失敗する
DISPLAY_IP=127.0.0.1
DISPLAY_PORT=9876

source .venv/bin/activate

lerobot-record \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=$LEFT_F_PORT \
    --robot.right_arm_config.port=$RIGHT_F_PORT \
    --robot.id=bi_follower_arm \
    --teleop.type=bi_so_leader \
    --teleop.left_arm_config.port=$LEFT_L_PORT \
    --teleop.right_arm_config.port=$RIGHT_L_PORT \
    --teleop.id=bi_leader_arm \
    --dataset.repo_id=$REPO_ID \
    --dataset.single_task="$SINGLE_TASK" \
    --dataset.num_episodes="$DATASET_NUM_EPISODES" \
    --dataset.episode_time_s="$DATASET_EPISODE_TIME_S" \
    --dataset.reset_time_s="$DATASET_RESET_TIME_S" \
    --dataset.fps=$FPS \
    --dataset.push_to_hub=False \
    --robot.left_arm_config.cameras="$CAMERA_ALL" \
    --display_data=true \
    --display_ip=$DISPLAY_IP \
    --display_port=$DISPLAY_PORT \
    --display_compressed_images=true \
    --play_sounds=false