# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive confirmation shared by OpenArm deployment motions."""

import logging
import sys
from typing import TextIO

logger = logging.getLogger(__name__)


def _read_yes_no(input_stream: TextIO, output_stream: TextIO, prompt: str) -> bool:
    while True:
        output_stream.write(prompt)
        output_stream.flush()
        answer = input_stream.readline()
        if not answer:
            return False
        normalized = answer.strip().lower()
        if normalized in {"yes", "y"}:
            return True
        if normalized in {"no", "n"}:
            return False
        output_stream.write("Please answer yes or no.\n")
        output_stream.flush()


def confirm_openarm_motion(prompt: str, unavailable_message: str) -> bool:
    """Ask on the controlling terminal, falling back to interactive stdin."""
    try:
        with open("/dev/tty", "r+", encoding="utf-8") as terminal:
            confirmed = _read_yes_no(terminal, terminal, prompt)
    except OSError as error:
        logger.info("Cannot open /dev/tty (%s); falling back to standard input.", error)
        if not sys.stdin.isatty():
            logger.warning("%s", unavailable_message)
            return False
        confirmed = _read_yes_no(sys.stdin, sys.stderr, prompt)

    if not confirmed:
        logger.warning("%s", unavailable_message)
    return confirmed
