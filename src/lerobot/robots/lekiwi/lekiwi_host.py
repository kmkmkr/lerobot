#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import base64
import json
import logging
import time

import cv2
import zmq

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--connection-time-s must be greater than 0.")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--robot.port",
        dest="robot_port",
        default="/dev/ttyACM0",
        help="Serial port for the LeKiwi motor bus (e.g. /dev/ttyAMA10).",
    )
    parser.add_argument(
        "--robot.id",
        dest="robot_id",
        default=None,
        help="Optional robot id used for calibration file naming.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Run LeKiwi without follower arm motors (only base motors ids 7,8,9).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--connection-time-s",
        type=_positive_float,
        default=None,
        help="Run host for a fixed number of seconds.",
    )
    group.add_argument(
        "--run-forever",
        action="store_true",
        help="Run host indefinitely until interrupted (Ctrl+C).",
    )
    # Keep backward compatibility with existing launch commands that may pass extra args.
    args, _ = parser.parse_known_args()
    return args


def main():
    args = _parse_args()

    logging.info("Configuring LeKiwi")
    robot_config = LeKiwiConfig(port=args.robot_port, id=args.robot_id, has_arm=not args.base_only)
    robot = LeKiwi(robot_config)

    logging.info("Connecting LeKiwi")
    robot.connect()

    logging.info("Starting HostAgent")
    host_config = LeKiwiHostConfig()
    if args.run_forever:
        host_config.connection_time_s = None
    elif args.connection_time_s is not None:
        host_config.connection_time_s = args.connection_time_s
    host = LeKiwiHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False
    if host.connection_time_s is None:
        logging.info("Waiting for commands (run forever)...")
    else:
        logging.info("Waiting for commands for %.1f seconds...", host.connection_time_s)
    try:
        # Business logic
        start = time.perf_counter()
        while True:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            last_observation = robot.get_observation()

            # Encode ndarrays to base64 strings
            for cam_key, _ in robot.cameras.items():
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""

            # Send the observation to the remote agent
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
            if host.connection_time_s is not None and duration >= host.connection_time_s:
                print("Cycle time reached.")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
