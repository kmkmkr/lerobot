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

from io import StringIO
from unittest.mock import MagicMock

from lerobot.robots.bi_openarm_follower import confirmation


class _InteractiveInput(StringIO):
    def isatty(self) -> bool:
        return True


def test_confirmation_falls_back_to_interactive_standard_input(monkeypatch):
    input_stream = _InteractiveInput("invalid\nyes\n")
    output_stream = StringIO()
    open_tty = MagicMock(side_effect=OSError("no controlling terminal"))
    monkeypatch.setattr(confirmation, "open", open_tty, raising=False)
    monkeypatch.setattr(confirmation.sys, "stdin", input_stream)
    monkeypatch.setattr(confirmation.sys, "stderr", output_stream)

    assert confirmation.confirm_openarm_motion("Proceed? [yes/no]: ", "Choosing no.") is True
    open_tty.assert_called_once_with("/dev/tty", "r+", encoding="utf-8")
    assert output_stream.getvalue().count("Proceed? [yes/no]: ") == 2
    assert "Please answer yes or no." in output_stream.getvalue()


def test_confirmation_chooses_no_for_noninteractive_standard_input(monkeypatch):
    input_stream = MagicMock()
    input_stream.isatty.return_value = False
    open_tty = MagicMock(side_effect=OSError("no controlling terminal"))
    monkeypatch.setattr(confirmation, "open", open_tty, raising=False)
    monkeypatch.setattr(confirmation.sys, "stdin", input_stream)

    assert confirmation.confirm_openarm_motion("Proceed? [yes/no]: ", "Choosing no.") is False
    input_stream.readline.assert_not_called()
