
import asyncio
import logging
import time
from datetime import datetime
import os
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)
from unitree_webrtc_connect.constants import RTC_TOPIC

ROBOT_IP = os.environ.get("ROBOT_IP")


logging.basicConfig(level=logging.FATAL)


def odometry_callback(message):
    data = message.get("data", {})

    position = data.get("position", [])
    velocity = data.get("velocity", [])
    yaw_speed = data.get("yaw_speed", None)
    body_height = data.get("body_height", None)
    imu = data.get("imu_state", {})

    rpy = imu.get("rpy", []) if isinstance(imu, dict) else []

    print("\n========== GO2 ODOMETRY ==========")
    print("Time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])

    if len(position) >= 3:
        print(
            f"Position: X={position[0]:.4f}, "
            f"Y={position[1]:.4f}, Z={position[2]:.4f} m"
        )
    else:
        print("Position: unavailable", position)

    if len(velocity) >= 3:
        print(
            f"Velocity: X={velocity[0]:.4f}, "
            f"Y={velocity[1]:.4f}, Z={velocity[2]:.4f} m/s"
        )
    else:
        print("Velocity: unavailable", velocity)

    print("Yaw speed:", yaw_speed, "rad/s")
    print("Body height:", body_height, "m")

    if len(rpy) >= 3:
        print(
            f"RPY: Roll={rpy[0]:.4f}, "
            f"Pitch={rpy[1]:.4f}, Yaw={rpy[2]:.4f}"
        )
    else:
        print("RPY: unavailable")

    print("==================================", flush=True)


async def main():
    print(f"Connecting to Go2 at {ROBOT_IP}...")

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP
    )

    await conn.connect()

    print("Connected!")
    print("Subscribing to SportModeState...")

    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["LF_SPORT_MOD_STATE"],
        odometry_callback
    )

    print("Receiving odometry data. Press CTRL+C to stop.")

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOdometry monitoring stopped.")
