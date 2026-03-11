#!/usr/bin/env bash
set -euo pipefail

# Prefer local source tree over site-packages so edits under this repo are used.
# export PYTHONPATH="${HOME}/gitrepo/lerobot/src:${PYTHONPATH:-}"

../../.venv/bin/python examples/lekiwi/teleoperate.py "$@"
