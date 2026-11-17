import asyncio
import logging
import os
import sys

from aiortc.contrib.media import MediaPlayer

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD


logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("ROBOT_IP")


async def spin(conn):
    print("Spinning...")

    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {
                "x": 0,
                "y": 0,
                "z": 1.0
            }
        }
    )

    await asyncio.sleep(3)

    print("Stopping...")

    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["StopMove"]
        }
    )


async def play_audio(conn):
    # MP3 file in the same directory as this Python script
    audio_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "robot_noises.mp3"
    )

    print("Playing borrow-a-fry.mp3...")

    player = MediaPlayer(audio_file)
    audio_track = player.audio

    if audio_track is None:
        print("Could not load audio file.")
        return

    conn.pc.addTrack(audio_track)


async def main():
    try:
        # Connect to Go2
        conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA,
            ip=ROBOT_IP
        )

        print("Connecting to Go2...")
        await conn.connect()
        print("Connected!")

        # Play audio
        await play_audio(conn)

        # Spin
        await spin(conn)

        # Keep WebRTC connection alive so audio can continue playing
        await asyncio.sleep(30)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted")
        sys.exit(0)