#!/usr/bin/env python

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

import logging
import subprocess
from unittest.mock import patch

from lerobot.utils.utils import log_say


def test_log_say_does_not_raise_when_text_to_speech_fails(caplog):
    error = subprocess.CalledProcessError(1, ["spd-say", "Stop recording", "--wait"])

    with (
        patch("lerobot.utils.utils.say", side_effect=error),
        caplog.at_level(logging.WARNING),
    ):
        log_say("Stop recording", play_sounds=True, blocking=True)

    assert "Text-to-speech failed" in caplog.text


def test_log_say_skips_text_to_speech_when_disabled():
    with patch("lerobot.utils.utils.say") as mock_say:
        log_say("Stop recording", play_sounds=False, blocking=True)

    mock_say.assert_not_called()
