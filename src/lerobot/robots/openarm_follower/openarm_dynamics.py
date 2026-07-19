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

"""OpenArm gravity and friction terms matching the native bilateral follower."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _vector(value: str | None) -> NDArray[np.float64]:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    result = np.asarray([float(component) for component in value.split()], dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected a finite three-component URDF vector, got {value!r}")
    return result


def _rpy_rotation(rpy: NDArray[np.float64]) -> NDArray[np.float64]:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _transform(xyz: NDArray[np.float64], rpy: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rpy_rotation(rpy)
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("A revolute URDF joint must have a finite non-zero axis")
    x, y, z = axis / norm
    sine = math.sin(angle)
    cosine = math.cos(angle)
    one_minus_cosine = 1.0 - cosine
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )
    return result


@dataclass(frozen=True)
class _ChainJoint:
    name: str
    joint_type: str
    child_link: str
    origin: NDArray[np.float64]
    axis: NDArray[np.float64]


@dataclass(frozen=True)
class _LinkMass:
    mass: float
    center_of_mass: NDArray[np.float64]


class OpenArmGravityModel:
    """Compute KDL-equivalent static gravity compensation from an OpenArm URDF chain."""

    def __init__(self, urdf_path: str | Path, side: str, gravity_m_s2: float = 9.81):
        if side not in {"left", "right"}:
            raise ValueError("OpenArm gravity model side must be 'left' or 'right'")
        if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be positive and finite")

        source = Path(urdf_path).expanduser()
        try:
            root = ET.parse(source).getroot()
        except (OSError, ET.ParseError) as error:
            raise ValueError(f"Failed to load OpenArm dynamics URDF {source}: {error}") from error

        self.source = source
        self.side = side
        self.gravity = np.asarray([0.0, 0.0, -gravity_m_s2], dtype=np.float64)
        self._links = self._parse_link_masses(root)
        self._chain = self._parse_chain(root, "openarm_body_link0", f"openarm_{side}_hand")
        active_count = sum(joint.joint_type in {"continuous", "revolute"} for joint in self._chain)
        if active_count != 7:
            raise ValueError(
                f"OpenArm {side} dynamics chain must contain 7 revolute joints, got {active_count}"
            )

    @staticmethod
    def _parse_link_masses(root: ET.Element) -> dict[str, _LinkMass]:
        links: dict[str, _LinkMass] = {}
        for link in root.findall("link"):
            name = link.get("name")
            if name is None:
                continue
            inertial = link.find("inertial")
            if inertial is None:
                links[name] = _LinkMass(0.0, np.zeros(3, dtype=np.float64))
                continue
            mass_element = inertial.find("mass")
            mass = float(mass_element.get("value", "0")) if mass_element is not None else 0.0
            origin = inertial.find("origin")
            center_of_mass = _vector(origin.get("xyz") if origin is not None else None)
            if not math.isfinite(mass) or mass < 0.0:
                raise ValueError(f"URDF link {name!r} has an invalid mass {mass}")
            links[name] = _LinkMass(mass, center_of_mass)
        return links

    @staticmethod
    def _parse_chain(root: ET.Element, root_link: str, leaf_link: str) -> tuple[_ChainJoint, ...]:
        joints_by_child: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            child_link = child.get("link") if child is not None else None
            if child_link is not None:
                joints_by_child[child_link] = joint

        reversed_chain: list[ET.Element] = []
        current_link = leaf_link
        while current_link != root_link:
            joint = joints_by_child.get(current_link)
            if joint is None:
                raise ValueError(f"No URDF chain from {root_link!r} to {leaf_link!r}")
            reversed_chain.append(joint)
            parent = joint.find("parent")
            current_link = parent.get("link", "") if parent is not None else ""

        chain: list[_ChainJoint] = []
        for joint in reversed(reversed_chain):
            name = joint.get("name", "unnamed")
            joint_type = joint.get("type", "")
            if joint_type not in {"continuous", "fixed", "revolute"}:
                raise ValueError(f"Unsupported joint type {joint_type!r} in OpenArm dynamics chain")
            child = joint.find("child")
            if child is None or child.get("link") is None:
                raise ValueError(f"URDF joint {name!r} has no child link")
            origin = joint.find("origin")
            xyz = _vector(origin.get("xyz") if origin is not None else None)
            rpy = _vector(origin.get("rpy") if origin is not None else None)
            axis = joint.find("axis")
            axis_xyz = _vector(axis.get("xyz") if axis is not None else "1 0 0")
            chain.append(
                _ChainJoint(
                    name=name,
                    joint_type=joint_type,
                    child_link=child.get("link", ""),
                    origin=_transform(xyz, rpy),
                    axis=axis_xyz,
                )
            )
        return tuple(chain)

    def gravity_torques(self, positions_rad: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        """Return the seven actuator torques that statically compensate gravity."""
        if len(positions_rad) != 7:
            raise ValueError(f"OpenArm gravity model requires 7 positions, got {len(positions_rad)}")
        if not all(math.isfinite(position) for position in positions_rad):
            raise ValueError("OpenArm gravity positions must be finite")

        link_transform = np.eye(4, dtype=np.float64)
        active_axes: list[NDArray[np.float64]] = []
        active_origins: list[NDArray[np.float64]] = []
        torques = np.zeros(7, dtype=np.float64)
        position_index = 0

        for joint in self._chain:
            joint_transform = link_transform @ joint.origin
            if joint.joint_type in {"continuous", "revolute"}:
                active_origins.append(joint_transform[:3, 3].copy())
                active_axes.append(joint_transform[:3, :3] @ joint.axis)
                joint_transform = joint_transform @ _axis_rotation(joint.axis, positions_rad[position_index])
                position_index += 1
            link_transform = joint_transform

            link_mass = self._links.get(joint.child_link)
            if link_mass is None or link_mass.mass == 0.0:
                continue
            center_of_mass = link_transform[:3, :3] @ link_mass.center_of_mass + link_transform[:3, 3]
            gravity_force = link_mass.mass * self.gravity
            for index, (axis, origin) in enumerate(zip(active_axes, active_origins, strict=True)):
                # KDL's JntToGravity returns the actuator effort required to
                # balance physical gravity, hence the negative generalized force.
                torques[index] -= float(np.dot(axis, np.cross(center_of_mass - origin, gravity_force)))

        if not np.all(np.isfinite(torques)):
            raise RuntimeError("OpenArm gravity model produced a non-finite torque")
        return tuple(float(torque) for torque in torques)


def bilateral_friction_torque(
    velocity_rad_s: float,
    fc: float,
    friction_k: float,
    fv: float,
    fo: float,
    tanh_coefficient: float,
) -> float:
    """Apply the exact tanh friction equation used by native ``Control``."""
    values = (velocity_rad_s, fc, friction_k, fv, fo, tanh_coefficient)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("OpenArm friction inputs must be finite")
    return fc * math.tanh(tanh_coefficient * friction_k * velocity_rad_s) + fv * velocity_rad_s + fo
