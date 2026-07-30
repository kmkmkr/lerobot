"""Minimal test script for Damiao motor with ID 3."""

from unittest.mock import MagicMock, call, patch

import pytest

from lerobot.utils.import_utils import _can_available

if not _can_available:
    pytest.skip("python-can not available", allow_module_level=True)

import can

from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import CAN_CMD_DISABLE, CAN_CMD_ENABLE


def _make_test_bus(*, two_motors: bool = False) -> DamiaoMotorsBus:
    motors = {
        "joint_1": Motor(
            id=0x01,
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=0x11,
        )
    }
    if two_motors:
        motors["joint_2"] = Motor(
            id=0x02,
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=0x12,
        )
    bus = DamiaoMotorsBus(port="can0", motors=motors)
    bus.canbus = MagicMock()
    bus._is_connected = True
    return bus


def _state_message(recv_id: int) -> can.Message:
    return can.Message(
        arbitration_id=recv_id,
        data=[0, 0, 0, 0, 0, 0, 0, 0],
        is_extended_id=False,
        is_fd=True,
    )


def test_simple_command_requires_a_decodable_response_only_in_strict_mode():
    bus = _make_test_bus()
    bus._drain_receive_queue = MagicMock()
    bus._recv_motor_response = MagicMock(return_value=None)

    bus._send_simple_command("joint_1", CAN_CMD_ENABLE)
    with pytest.raises(RuntimeError, match="No response"):
        bus._send_simple_command("joint_1", CAN_CMD_ENABLE, require_response=True)

    bus._drain_receive_queue.assert_called_once_with()


def test_strict_enable_retries_and_propagates_exhaustion():
    bus = _make_test_bus()
    bus._send_simple_command = MagicMock(side_effect=[RuntimeError("drop"), None])

    with patch("lerobot.motors.damiao.damiao.time.sleep"):
        bus.enable_torque("joint_1", num_retry=1, require_response=True)

    assert bus._send_simple_command.call_args_list == [
        call("joint_1", CAN_CMD_ENABLE, require_response=True),
        call("joint_1", CAN_CMD_ENABLE, require_response=True),
    ]

    bus._send_simple_command.side_effect = RuntimeError("still missing")
    with patch("lerobot.motors.damiao.damiao.time.sleep"), pytest.raises(RuntimeError, match="still missing"):
        bus.enable_torque("joint_1", num_retry=1, require_response=True)


def test_strict_disable_sends_even_when_receive_queue_cannot_be_drained():
    bus = _make_test_bus()
    bus._drain_receive_queue = MagicMock(side_effect=RuntimeError("CAN queue is flooded"))

    with patch("lerobot.motors.damiao.damiao.time.sleep"), pytest.raises(RuntimeError, match="joint_1"):
        bus.disable_torque("joint_1", num_retry=1, require_response=True)

    assert bus.canbus.send.call_count == 2
    for sent_call in bus.canbus.send.call_args_list:
        assert sent_call.args[0].data[-1] == CAN_CMD_DISABLE


def test_strict_disable_attempts_every_motor_before_raising():
    bus = _make_test_bus(two_motors=True)

    def send(motor, command, *, require_response=False):
        assert command == CAN_CMD_DISABLE
        assert require_response
        if motor == "joint_1":
            raise RuntimeError("joint_1 ACK missing")

    bus._send_simple_command = MagicMock(side_effect=send)

    with pytest.raises(RuntimeError, match="joint_1"):
        bus.disable_torque(num_retry=0, require_response=True)

    assert bus._send_simple_command.call_args_list == [
        call("joint_1", CAN_CMD_DISABLE, require_response=True),
        call("joint_2", CAN_CMD_DISABLE, require_response=True),
    ]


def test_strict_disable_attempts_every_motor_before_reraising_keyboard_interrupt():
    bus = _make_test_bus(two_motors=True)
    interrupt = KeyboardInterrupt("operator interrupted strict disable")

    def send(motor, command, *, require_response=False):
        assert command == CAN_CMD_DISABLE
        assert require_response
        if motor == "joint_1":
            raise interrupt

    bus._send_simple_command = MagicMock(side_effect=send)

    with pytest.raises(KeyboardInterrupt, match="operator interrupted") as exc_info:
        bus.disable_torque(num_retry=0, require_response=True)

    assert exc_info.value is interrupt
    assert bus._send_simple_command.call_args_list == [
        call("joint_1", CAN_CMD_DISABLE, require_response=True),
        call("joint_2", CAN_CMD_DISABLE, require_response=True),
    ]


def test_strict_disable_continues_when_keyboard_interrupt_arrives_during_retry_delay():
    bus = _make_test_bus(two_motors=True)
    interrupt = KeyboardInterrupt("operator interrupted retry delay")
    joint_1_attempts = 0

    def send(motor, command, *, require_response=False):
        nonlocal joint_1_attempts
        assert command == CAN_CMD_DISABLE
        assert require_response
        if motor == "joint_1":
            joint_1_attempts += 1
            if joint_1_attempts == 1:
                raise RuntimeError("joint_1 ACK missing")

    bus._send_simple_command = MagicMock(side_effect=send)

    with (
        patch("lerobot.motors.damiao.damiao.time.sleep", side_effect=interrupt),
        pytest.raises(KeyboardInterrupt, match="retry delay") as exc_info,
    ):
        bus.disable_torque(num_retry=1, require_response=True)

    assert exc_info.value is interrupt
    assert bus._send_simple_command.call_args_list == [
        call("joint_1", CAN_CMD_DISABLE, require_response=True),
        call("joint_1", CAN_CMD_DISABLE, require_response=True),
        call("joint_2", CAN_CMD_DISABLE, require_response=True),
    ]


def test_connect_handshake_failure_disables_all_motors_and_closes_socket():
    bus = _make_test_bus(two_motors=True)
    socket = MagicMock()
    bus.canbus = None
    bus._is_connected = False
    bus._handshake = MagicMock(side_effect=ConnectionError("joint missing"))
    bus._send_simple_command = MagicMock()

    with (
        patch("lerobot.motors.damiao.damiao.can.interface.Bus", return_value=socket),
        pytest.raises(ConnectionError, match="joint missing"),
    ):
        bus.connect()

    assert bus._send_simple_command.call_args_list == [
        call("joint_1", CAN_CMD_DISABLE, require_response=True),
        call("joint_2", CAN_CMD_DISABLE, require_response=True),
    ]
    socket.shutdown.assert_called_once_with()
    assert bus.canbus is None
    assert not bus.is_connected


def test_strict_mit_batch_rejects_partial_or_invalid_responses():
    bus = _make_test_bus(two_motors=True)
    bus._drain_receive_queue = MagicMock()
    bus._recv_all_responses = MagicMock(return_value={0x11: _state_message(0x11)})
    commands = {
        "joint_1": (10.0, 0.5, 0.0, 0.0, 0.0),
        "joint_2": (10.0, 0.5, 0.0, 0.0, 0.0),
    }

    bus._mit_control_batch(commands)
    with pytest.raises(RuntimeError, match="joint_2"):
        bus._mit_control_batch(commands, require_response=True)


def test_strict_state_read_retries_only_missing_motors():
    bus = _make_test_bus(two_motors=True)
    bus._batch_refresh = MagicMock(side_effect=[["joint_2"], []])

    with patch("lerobot.motors.damiao.damiao.time.sleep"):
        states = bus.sync_read_all_states(num_retry=1, require_response=True)

    assert set(states) == {"joint_1", "joint_2"}
    assert bus._batch_refresh.call_args_list == [
        call(["joint_1", "joint_2"], drain_before_send=True),
        call(["joint_2"], drain_before_send=True),
    ]


def test_strict_state_read_never_falls_back_to_stale_cache():
    bus = _make_test_bus()
    bus._batch_refresh = MagicMock(return_value=["joint_1"])

    with patch("lerobot.motors.damiao.damiao.time.sleep"), pytest.raises(RuntimeError, match="joint_1"):
        bus.sync_read_all_states(num_retry=2, require_response=True)

    assert bus._batch_refresh.call_count == 3


def test_process_response_reports_decode_failure():
    bus = _make_test_bus()

    assert not bus._process_response(
        "joint_1",
        can.Message(arbitration_id=0x11, data=[0], is_extended_id=False, is_fd=True),
    )


def test_handshake_processes_the_received_frame_not_the_enable_command():
    bus = _make_test_bus()
    response = _state_message(0x11)
    bus.canbus.recv.side_effect = [None, response]
    bus._process_response = MagicMock(return_value=True)

    with patch("lerobot.motors.damiao.damiao.time.sleep"):
        bus._handshake()

    bus._process_response.assert_called_once_with("joint_1", response)


@pytest.mark.skip(reason="Requires physical Damiao motor and CAN interface")
def test_damiao_motor():
    motors = {
        "joint_3": Motor(
            id=0x03,
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=0x13,
        ),
    }

    bus = DamiaoMotorsBus(port="can0", motors=motors)

    try:
        print("Connecting...")
        bus.connect()
        print("✓ Connected")

        print("Enabling torque...")
        bus.enable_torque()
        print("✓ Torque enabled")

        print("Reading all states...")
        states = bus.sync_read_all_states()
        print(f"✓ States: {states}")

        print("Reading position...")
        positions = bus.sync_read("Present_Position")
        print(f"✓ Position: {positions}")

        print("Testing MIT control batch...")
        current_pos = states["joint_3"]["position"]
        commands = {"joint_3": (10.0, 0.5, current_pos, 0.0, 0.0)}
        bus._mit_control_batch(commands)
        print("✓ MIT control batch sent")

        print("Disabling torque...")
        bus.disable_torque()
        print("✓ Torque disabled")

        print("Setting zero position...")
        bus.set_zero_position()
        print("✓ Zero position set")

    finally:
        print("Disconnecting...")
        bus.disconnect(disable_torque=True)
        print("✓ Disconnected")


if __name__ == "__main__":
    test_damiao_motor()
