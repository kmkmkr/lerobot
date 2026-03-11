#!/usr/bin/env bash
set -euo pipefail

# Prefer local source tree over site-packages so edits under this repo are used.
# export PYTHONPATH="${HOME}/gitrepo/lerobot/src:${PYTHONPATH:-}"

host_mode_args=(--run-forever)
if [ "${1:-}" = "--connection-time-s" ]; then
    if [ -z "${2:-}" ]; then
        echo "Usage: $0 [--run-forever | --connection-time-s <seconds>] [extra host args...]"
        exit 1
    fi
    host_mode_args=(--connection-time-s "$2")
    shift 2
elif [ "${1:-}" = "--run-forever" ]; then
    host_mode_args=(--run-forever)
    shift
elif [ -n "${1:-}" ] && [[ "${1}" != -* ]]; then
    # Shorthand: first positional numeric value is treated as connection time in seconds.
    host_mode_args=(--connection-time-s "$1")
    shift
fi
../../.venv/bin/python -m lerobot.robots.lekiwi.lekiwi_host \
    --base-only \
    --robot.port /dev/ttyACM0 \
    "${host_mode_args[@]}" \
    "$@"
