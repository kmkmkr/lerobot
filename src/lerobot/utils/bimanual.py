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

from typing import Any

from lerobot.utils.decorators import check_if_already_connected


class BimanualMixin:
    """Lifecycle delegation for bimanual robots and teleoperators.

    Concrete subclasses must populate ``self.left_arm`` and ``self.right_arm`` in
    their own ``__init__``. They retain ownership of feature dicts and the
    data-routing methods (``get_action`` / ``send_action`` / ``get_observation`` /
    ``send_feedback``), which vary per-embodiment.

    Inherit before the ``Robot`` / ``Teleoperator`` base so the mixin's methods
    take precedence in the MRO::

        class BiFooFollower(BimanualMixin, Robot): ...
    """

    left_arm: Any
    right_arm: Any

    @staticmethod
    def _arm_has_live_connection(arm: Any) -> bool:
        """Detect partially connected arms, including a live CAN bus with a failed camera."""
        try:
            if bool(arm.is_connected):
                return True
        except Exception:
            pass
        bus = getattr(arm, "bus", None)
        try:
            if bus is not None and bool(bus.is_connected):
                return True
        except Exception:
            pass
        cameras = getattr(arm, "cameras", {})
        try:
            return any(bool(camera.is_connected) for camera in cameras.values())
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def any_arm_connected(self) -> bool:
        return self._arm_has_live_connection(self.left_arm) or self._arm_has_live_connection(self.right_arm)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def disconnect(self) -> None:
        errors: list[BaseException] = []
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            if not self._arm_has_live_connection(arm):
                continue
            try:
                arm.disconnect()
            except BaseException as error:
                error.add_note(f"Failed while disconnecting {side} arm")
                errors.append(error)

        if errors:
            first_error = errors[0]
            for additional_error in errors[1:]:
                first_error.add_note(f"Additional bimanual disconnect error: {additional_error!r}")
            raise first_error
