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

"""CSV parsing and interpolation for OpenArm policy-deployment home motions."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

MOTOR_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
)


@dataclass(frozen=True)
class DeploymentTrajectorySample:
    time_s: float
    positions_deg: tuple[float, ...]


def load_deployment_trajectory(path: str | Path, expected_side: str) -> list[DeploymentTrajectorySample]:
    """Load one exported bilateral trajectory and convert radians to motor-zero degrees."""
    source = Path(path).expanduser()
    metadata: dict[str, str] = {}
    data_rows: list[tuple[int, str]] = []
    with source.open(encoding="utf-8", newline="") as csv_file:
        for line_number, line in enumerate(csv_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                key_value = stripped[1:].strip().split("=", maxsplit=1)
                if len(key_value) == 2:
                    metadata[key_value[0].strip()] = key_value[1].strip()
                continue
            data_rows.append((line_number, stripped))

    if metadata.get("coordinate_frame") != "openarm_v1_motor_zero":
        raise ValueError(f"{source}: coordinate_frame must be openarm_v1_motor_zero")
    if metadata.get("position_unit") != "radian":
        raise ValueError(f"{source}: position_unit must be radian")
    if metadata.get("side") != expected_side:
        raise ValueError(f"{source}: expected side={expected_side}, got {metadata.get('side')!r}")
    if not data_rows:
        raise ValueError(f"{source}: trajectory CSV is empty")

    expected_header = ("time_s", *MOTOR_NAMES)
    header = tuple(next(csv.reader([data_rows[0][1]])))
    if header != expected_header:
        raise ValueError(f"{source}:{data_rows[0][0]}: expected header {expected_header}, got {header}")

    samples: list[DeploymentTrajectorySample] = []
    for line_number, row in data_rows[1:]:
        fields = next(csv.reader([row]))
        if len(fields) != len(expected_header):
            raise ValueError(
                f"{source}:{line_number}: expected {len(expected_header)} columns, got {len(fields)}"
            )
        try:
            values = tuple(float(field) for field in fields)
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: trajectory values must be numeric") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{source}:{line_number}: trajectory values must be finite")
        samples.append(
            DeploymentTrajectorySample(
                time_s=values[0],
                positions_deg=tuple(math.degrees(position) for position in values[1:]),
            )
        )

    if len(samples) < 2:
        raise ValueError(f"{source}: trajectory must contain at least two samples")
    return samples


def validate_deployment_trajectory(
    samples: list[DeploymentTrajectorySample],
    joint_limits: dict[str, tuple[float, float]],
    speed_scale: float,
    *,
    initial_position_tolerance_deg: float = math.degrees(0.15),
    max_sample_gap_s: float = 0.1,
    max_duration_s: float = 300.0,
    max_velocity_deg_s: float = math.degrees(8.0),
) -> None:
    """Validate the same coordinate, timing, position, and speed contract as native teleop."""
    if len(samples) < 2:
        raise ValueError("trajectory must contain at least two samples")
    if set(joint_limits) != set(MOTOR_NAMES):
        raise ValueError("joint_limits must contain exactly the OpenArm trajectory motors")
    if not math.isfinite(speed_scale) or speed_scale <= 0.0:
        raise ValueError("trajectory speed_scale must be positive and finite")
    if not math.isclose(samples[0].time_s, 0.0, abs_tol=1e-9):
        raise ValueError("trajectory must begin at time_s=0")
    if samples[-1].time_s > max_duration_s:
        raise ValueError("trajectory exceeds the maximum duration")

    for sample_index, sample in enumerate(samples):
        if len(sample.positions_deg) != len(MOTOR_NAMES):
            raise ValueError(f"trajectory sample {sample_index} has an invalid motor count")
        for motor_name, position in zip(MOTOR_NAMES, sample.positions_deg, strict=True):
            minimum, maximum = joint_limits[motor_name]
            if not math.isfinite(position) or not minimum <= position <= maximum:
                raise ValueError(
                    f"trajectory position limit exceeded: sample={sample_index} motor={motor_name} "
                    f"value={position} range=[{minimum}, {maximum}] deg"
                )

        if sample_index == 0:
            continue
        previous = samples[sample_index - 1]
        interval_s = sample.time_s - previous.time_s
        if not math.isfinite(interval_s) or not 0.0 < interval_s <= max_sample_gap_s:
            raise ValueError(f"invalid trajectory sample interval at sample {sample_index}: {interval_s} s")
        for motor_name, before, after in zip(
            MOTOR_NAMES, previous.positions_deg, sample.positions_deg, strict=True
        ):
            velocity = abs(after - before) / interval_s * speed_scale
            if velocity > max_velocity_deg_s:
                raise ValueError(
                    f"trajectory velocity limit exceeded: sample={sample_index} motor={motor_name} "
                    f"value={velocity} limit={max_velocity_deg_s} deg/s"
                )

    for motor_name, position in zip(MOTOR_NAMES, samples[0].positions_deg, strict=True):
        if abs(position) > initial_position_tolerance_deg:
            raise ValueError(
                f"trajectory does not begin near motor zero: motor={motor_name} offset={abs(position)} "
                f"tolerance={initial_position_tolerance_deg} deg"
            )


def interpolate_deployment_trajectory(
    samples: list[DeploymentTrajectorySample], time_s: float
) -> tuple[float, ...]:
    if not samples:
        raise ValueError("cannot interpolate an empty trajectory")
    if time_s <= samples[0].time_s:
        return samples[0].positions_deg
    if time_s >= samples[-1].time_s:
        return samples[-1].positions_deg

    low = 0
    high = len(samples) - 1
    while low + 1 < high:
        midpoint = (low + high) // 2
        if samples[midpoint].time_s <= time_s:
            low = midpoint
        else:
            high = midpoint

    before = samples[low]
    after = samples[high]
    alpha = (time_s - before.time_s) / (after.time_s - before.time_s)
    return tuple(
        before_position * (1.0 - alpha) + after_position * alpha
        for before_position, after_position in zip(before.positions_deg, after.positions_deg, strict=True)
    )


def build_return_to_zero_trajectory(
    forward_samples: list[DeploymentTrajectorySample], zero_transition_s: float
) -> list[DeploymentTrajectorySample]:
    """Reverse the recorded path and append the exact motor-zero endpoint."""
    if len(forward_samples) < 2:
        raise ValueError("return trajectory requires at least two samples")
    if not math.isfinite(zero_transition_s) or zero_transition_s <= 0.0:
        raise ValueError("zero_transition_s must be positive and finite")

    forward_end_s = forward_samples[-1].time_s
    return_samples = [
        DeploymentTrajectorySample(forward_end_s - sample.time_s, sample.positions_deg)
        for sample in reversed(forward_samples)
    ]
    return_samples.append(
        DeploymentTrajectorySample(
            return_samples[-1].time_s + zero_transition_s,
            (0.0,) * len(MOTOR_NAMES),
        )
    )
    return return_samples
