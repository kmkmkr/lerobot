#!/bin/bash
#SBATCH --gpus=1
#SBATCH -J bash
#SBATCH --time=12:00:00
echo $CUDA_VISIBLE_DEVICES

HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
/home/sota.nakamura/gorgeous/lerobot/.venv/bin/python \
-m lerobot.rl.convert_imitation_to_rl_dataset \
  --input-repo-id nkmurst/gorgeous_south_america_1 \
  --input-root /home/sota.nakamura/.cache/huggingface/hub/datasets--nkmurst--gorgeous_south_america_1/snapshots/13457d61215127f492938c8e2aeec8cf452be136 \
  --output-repo-id nkmurst/gorgeous_south_america_1_terminal_rl \
  --output-root /home/sota.nakamura/.cache/huggingface/lerobot/nkmurst/gorgeous_south_america_1_terminal_rl \
  --trim-tail-frames 72 \
  --min-remaining-frames 10 \
  --image-writer-threads 4 \
  --vcodec h264 \
  --encoder-threads 1 \
  --encoder-threads 2 \
  --parallel-video-encoding \
  --push-to-hub
