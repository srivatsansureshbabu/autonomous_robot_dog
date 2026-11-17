
import asyncio
import logging
import os 
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)
from unitree_webrtc_connect.constants import RTC_TOPIC

ROBOT_IP = os.environ.get("ROBOT_IP")
logging.basicConfig(level=logging.FATAL)

count = 0

def callback(message):
    global count
    count += 1

    data = message.get("data", {})

    print("\n========== LOWSTATE ==========")
    print("Message topic:", message.get("topic"))
    print("Message type:", message.get("type"))
    print("Available fields:", list(data.keys()))

    print("Foot force:", data.get("foot_force", "MISSING"))
    print("Estimated foot force:", data.get("foot_force_est", "MISSING"))

    print("Power voltage:", data.get("power_v", "MISSING"))
    print("IMU:", data.get("imu_state", "MISSING"))

    print("Message number:", count)
    print("=============================", flush=True)


async def main():
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP
    )

    print("Connecting...")
    await conn.connect()

    print("Subscribing to LOW_STATE...")

    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["LOW_STATE"],
        callback
    )

    print("Reading LowState. Press CTRL+C to stop.")

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
