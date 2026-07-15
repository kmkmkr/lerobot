#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${LEROBOT_OPENARM_IMAGE:-${SCRIPT_DIR}/lerobot_openarm.sif}"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not installed or not on PATH" >&2
    exit 127
fi

if [ ! -f "${IMAGE}" ]; then
    cat >&2 <<EOF
Missing image: ${IMAGE}

Build it from the repository root:
  apptainer build --fakeroot --force ${IMAGE} ${SCRIPT_DIR}/lerobot_openarm.def

If --fakeroot is unavailable:
  sudo apptainer build --force ${IMAGE} ${SCRIPT_DIR}/lerobot_openarm.def
EOF
    exit 1
fi

APPTAINER_ARGS=(
    --bind "${REPO_ROOT}:/workspace/lerobot"
    --pwd /workspace/lerobot
    --env "PYTHONDONTWRITEBYTECODE=1"
    --env "UV_LINK_MODE=copy"
    --env "UV_PROJECT_ENVIRONMENT=/workspace/lerobot/.venv"
)

if [ "${LEROBOT_OPENARM_NV:-1}" != "0" ]; then
    APPTAINER_ARGS+=(--nv)
fi

if [ "${LEROBOT_OPENARM_BIND_DEV:-0}" = "1" ] && [ -d /dev ]; then
    APPTAINER_ARGS+=(--bind /dev:/dev)
fi

if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
    APPTAINER_ARGS+=(--env "DISPLAY=${DISPLAY}" --bind /tmp/.X11-unix:/tmp/.X11-unix)
fi

exec apptainer exec "${APPTAINER_ARGS[@]}" "${IMAGE}" "$@"
