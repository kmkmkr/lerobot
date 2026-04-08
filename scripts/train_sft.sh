#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH -J bash
#SBATCH --time=24:00:00


set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

VENV_DIR="${VENV_DIR:-${PWD}/.venv}"
CACHE_ROOT="${CACHE_ROOT:-/data/sota.nakamura}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_HOME}/assets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${CACHE_ROOT}/lerobot}"

DATASET_REPO_ID="${DATASET_REPO_ID:-nkmurst/gorgeous-south-america-30hz}"
DATASET_NAME="${DATASET_NAME:-${DATASET_REPO_ID##*/}}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-100000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
N_ACTION_STEPS="${N_ACTION_STEPS:-100}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"

JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CACHE_ROOT}/outputs/train/${JOB_NAME}}"
POLICY_REPO_ID="${POLICY_REPO_ID:-${DATASET_REPO_ID%/*}/${JOB_NAME}}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_ASSETS_CACHE}" "${HF_LEROBOT_HOME}" "$(dirname "${OUTPUT_DIR}")"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Python environment not found: ${VENV_DIR}/bin/activate" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
source "${VENV_DIR}/bin/activate"

CMD=(
    lerobot-train
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--policy.type=act"
    "--policy.device=${DEVICE}"
    "--policy.push_to_hub=${PUSH_TO_HUB}"
    "--policy.chunk_size=${CHUNK_SIZE}"
    "--policy.n_action_steps=${N_ACTION_STEPS}"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=${JOB_NAME}"
    "--batch_size=${BATCH_SIZE}"
    "--steps=${STEPS}"
    "--num_workers=${NUM_WORKERS}"
    "--log_freq=${LOG_FREQ}"
    "--save_freq=${SAVE_FREQ}"
    "--eval_freq=-1"
    "--wandb.enable=${WANDB_ENABLE}"
)

if [[ "${PUSH_TO_HUB}" == "true" ]]; then
    CMD+=("--policy.repo_id=${POLICY_REPO_ID}")
fi

CMD+=("$@")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
