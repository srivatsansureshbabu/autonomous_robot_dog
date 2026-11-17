
import asyncio
import logging
import os
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("ROBOT_IP")
HEX_PREVIEW_BYTES = 256

async def main():

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP
    )

    try:
        print("Connecting to Go2...")
        await conn.connect()
        print("Connected!")

        await conn.datachannel.disableTrafficSaving(True)

        print("Selecting LibVoxel decoder...")
        conn.datachannel.set_decoder(decoder_type="libvoxel")

        # Intercept raw bytes before the decoder processes them
        original_decode = conn.datachannel.decoder.decode

        packet_count = 0

        def debug_decode(binary_data, metadata):

            nonlocal packet_count
            packet_count += 1

            raw_bytes = bytes(binary_data)

            # Save the complete raw packet
            with open("lidar_payload.bin", "wb") as f:
                f.write(raw_bytes)

            # Print the first three packets for inspection
            if packet_count >= 3:

                print("\n" + "=" * 55)
                print(f"RAW LIDAR PACKET #{packet_count}")
                print("=" * 55)

                print("Payload size:", len(raw_bytes), "bytes")
                print("Metadata:", metadata)

                preview = raw_bytes[:HEX_PREVIEW_BYTES]

                print("\nHEX VALUES:")
                print(preview.hex(" "))

                print("\nBYTE VALUES:")
                print(list(preview))

                print("=" * 55)

            # Pass original bytes to the normal decoder
            return original_decode(binary_data, metadata)

        # Replace decoder function with diagnostic wrapper
        conn.datachannel.decoder.decode = debug_decode

        # Display decoded voxel map information
        def lidar_callback(message):

            data = message.get("data", {})

            print("\nDecoded LiDAR message received.")

            if isinstance(data, dict):
                print("Keys:", list(data.keys()))
                print("Point count:", data.get("point_count"))
                print("Face count:", data.get("face_count"))

        conn.datachannel.pub_sub.subscribe(
            "rt/utlidar/voxel_map_compressed",
            lidar_callback
        )

        conn.datachannel.pub_sub.subscribe(
            "rt/utlidar/lidar_state",
            lambda message: print("LiDAR state received")
        )

        await asyncio.sleep(1)

        print("\nEnabling LiDAR...")

        for _ in range(5):
            conn.datachannel.pub_sub.publish_without_callback(
                "rt/utlidar/switch",
                "ON"
            )
            await asyncio.sleep(0.1)

        print("\nWaiting for raw LiDAR packets...")
        print("Press Ctrl+C to stop.")

        while True:
            await asyncio.sleep(1)

    finally:
        print("\nClosing connection...")
        await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
