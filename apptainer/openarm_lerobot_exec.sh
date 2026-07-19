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

OPENARM_COMPENSATION_ENABLED=${LEROBOT_OPENARM_ENABLE_COMPENSATION:-1}
if [[ "${OPENARM_COMPENSATION_ENABLED}" != "0" && "${OPENARM_COMPENSATION_ENABLED}" != "1" ]]; then
    echo "LEROBOT_OPENARM_ENABLE_COMPENSATION must be 0 or 1" >&2
    exit 2
fi
APPTAINER_ARGS+=(--env "LEROBOT_OPENARM_ENABLE_COMPENSATION=${OPENARM_COMPENSATION_ENABLED}")

OPENARM_DYNAMICS_TMPDIR=""
cleanup_openarm_dynamics() {
    if [ -n "${OPENARM_DYNAMICS_TMPDIR}" ] && [ -d "${OPENARM_DYNAMICS_TMPDIR}" ]; then
        rm -rf -- "${OPENARM_DYNAMICS_TMPDIR}"
    fi
}
trap cleanup_openarm_dynamics EXIT

if [ "${OPENARM_COMPENSATION_ENABLED}" = "1" ]; then
    OPENARM_DYNAMICS_HOST_URDF=${LEROBOT_OPENARM_DYNAMICS_URDF:-}
    if [ -z "${OPENARM_DYNAMICS_HOST_URDF}" ]; then
        OPENARM_TELEOP_IMAGE=${LEROBOT_OPENARM_TELEOP_IMAGE:-${REPO_ROOT}/../openarm_teleop/openarm_teleop.sif}
        if [ ! -f "${OPENARM_TELEOP_IMAGE}" ]; then
            cat >&2 <<EOF
Missing native bilateral image required to generate the gravity model URDF:
  ${OPENARM_TELEOP_IMAGE}

Set LEROBOT_OPENARM_TELEOP_IMAGE to the openarm_teleop Apptainer image, or
set LEROBOT_OPENARM_DYNAMICS_URDF to a v10 bimanual URDF generated from the
same OpenArm Description version used by bilateral teleoperation.
EOF
            exit 1
        fi
        OPENARM_DYNAMICS_TMPDIR=$(mktemp -d)
        OPENARM_DYNAMICS_HOST_URDF=${OPENARM_DYNAMICS_TMPDIR}/openarm_v10_bimanual.urdf
        apptainer exec \
            --bind "${OPENARM_DYNAMICS_TMPDIR}:/output" \
            "${OPENARM_TELEOP_IMAGE}" \
            bash -lc \
            'set -eo pipefail; source /opt/ros/jazzy/setup.bash; source /opt/openarm_ros2_ws/install/setup.bash; xacro /opt/openarm_ros2_ws/src/openarm_description/urdf/robot/v10.urdf.xacro bimanual:=true -o /output/openarm_v10_bimanual.urdf'
    fi
    if [ ! -s "${OPENARM_DYNAMICS_HOST_URDF}" ]; then
        echo "OpenArm dynamics URDF is missing or empty: ${OPENARM_DYNAMICS_HOST_URDF}" >&2
        exit 1
    fi
    APPTAINER_ARGS+=(
        --bind "${OPENARM_DYNAMICS_HOST_URDF}:/workspace/openarm_v10_bimanual.urdf:ro"
        --env "LEROBOT_OPENARM_DYNAMICS_URDF=/workspace/openarm_v10_bimanual.urdf"
    )
fi

# Reuse the exact startup-trajectory CSV files exported for native bilateral
# teleoperation. The host path can be overridden when this repository is not
# checked out next to openarm_teleop.
OPENARM_TRAJECTORY_ROOT="${LEROBOT_OPENARM_TRAJECTORY_ROOT:-${REPO_ROOT}/../openarm_teleop/config/startup_trajectories}"
if [ -d "${OPENARM_TRAJECTORY_ROOT}" ]; then
    APPTAINER_ARGS+=(
        --bind "${OPENARM_TRAJECTORY_ROOT}:/workspace/openarm_startup_trajectories:ro"
    )
fi

if [ "${LEROBOT_OPENARM_NV:-1}" != "0" ]; then
    APPTAINER_ARGS+=(--nv)
fi

if [ "${LEROBOT_OPENARM_BIND_DEV:-0}" = "1" ] && [ -d /dev ]; then
    APPTAINER_ARGS+=(--bind /dev:/dev)
fi

if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
    APPTAINER_ARGS+=(--env "DISPLAY=${DISPLAY}" --bind /tmp/.X11-unix:/tmp/.X11-unix)
fi

apptainer exec "${APPTAINER_ARGS[@]}" "${IMAGE}" "$@"
