import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="127.0.0.1", id="my_lekiwi", has_arm=False)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
keyboard = KeyboardTeleop(keyboard_config)

# To connect, run this on LeKiwi first (base-only): `python -m lerobot.robots.lekiwi.lekiwi_host --base-only --run-forever`
robot.connect()
print("Robot connected!") if robot.is_connected else print("Failed to connect to robot.")
keyboard.connect()
print("Keyboard connected!") if keyboard.is_connected else print("Failed to connect to keyboard.")

# _init_rerun(session_name="lekiwi_teleop")



if not robot.is_connected or not keyboard.is_connected:
    # raise ValueError("Robot or keyboard is not connected!")
    raise ValueError("Robot or keyboard is not connected!")

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()

    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    # log_rerun_data(observation, base_action)
    robot.send_action(base_action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
