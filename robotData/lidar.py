
import asyncio
import logging
import os
import time
import threading

import numpy as np

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD


logging.basicConfig(level=logging.FATAL)

# ==========================================
# CONFIGURATION
# ==========================================

ROBOT_IP = "10.70.195.90"

# SAFETY: Start with dry-run enabled.
DRY_RUN = True

# Movement settings
FORWARD_SPEED = 0.12
MAX_TURN_SPEED = 0.35
TURN_GAIN = 0.8

# Obstacle detection
MAX_RANGE = 2.5
SAFETY_DISTANCE = 0.90
ROBOT_RADIUS = 0.35

# LiDAR filtering
MIN_Z = -0.25
MAX_Z = 1.0

# Processing
NUM_SECTORS = 61
FOV_DEGREES = 150
CONTROL_RATE = 10
LIDAR_TIMEOUT = 0.75


# ==========================================
# SHARED DATA
# ==========================================

latest_points = None
last_lidar_time = 0.0

data_lock = threading.Lock()

frame_count = 0
last_debug_time = 0.0


# ==========================================
# LIDAR CALLBACK
# ==========================================

def lidar_callback(message):

    global latest_points
    global last_lidar_time
    global frame_count
    global last_debug_time

    try:

        data = message.get("data", {})

        if not isinstance(data, dict):
            return

        # LibVoxel returns positions, not points().
        positions = data.get("positions")

        if positions is None:
            return

        points = np.asarray(
            positions,
            dtype=np.float32
        )

        # Positions should contain XYZ triples.
        if points.size == 0 or points.size % 3 != 0:
            return

        points = points.reshape(-1, 3)

        # Reject invalid measurements.
        points = points[np.isfinite(points).all(axis=1)]

        if len(points) == 0:
            return

        with data_lock:

            latest_points = points.copy()
            last_lidar_time = time.monotonic()
            frame_count += 1

        # Print diagnostic information periodically.
        now = time.monotonic()

        if now - last_debug_time > 3:

            last_debug_time = now

            print("\n========== LIDAR ==========")
            print("Point count:", data.get("point_count"))
            print("Face count:", data.get("face_count"))
            print("Extracted XYZ points:", len(points))
            print("Sample points:")
            print(points[:3])

            print("X range:", points[:, 0].min(),
                  points[:, 0].max())

            print("Y range:", points[:, 1].min(),
                  points[:, 1].max())

            print("Z range:", points[:, 2].min(),
                  points[:, 2].max())

            print("===========================\n")

    except Exception as error:

        print("LiDAR callback error:", repr(error))


# ==========================================
# POINT CLOUD PROCESSING
# ==========================================

def process_lidar(points):

    # Assumed coordinate convention:
    # X = forward
    # Y = left
    # Z = up

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    distance = np.sqrt(x**2 + y**2)

    angle = np.arctan2(y, x)

    # Keep points in front of the robot.
    valid = (
        (x > 0.05) &
        (distance < MAX_RANGE) &
        (z > MIN_Z) &
        (z < MAX_Z)
    )

    x = x[valid]
    y = y[valid]
    distance = distance[valid]
    angle = angle[valid]

    # Divide the field of view into sectors.
    sector_angles = np.linspace(
        -np.radians(FOV_DEGREES / 2),
        np.radians(FOV_DEGREES / 2),
        NUM_SECTORS
    )

    angle_step = (
        sector_angles[1] - sector_angles[0]
    )

    nearest = np.full(
        NUM_SECTORS,
        MAX_RANGE,
        dtype=np.float32
    )

    indices = np.rint(
        (angle - sector_angles[0]) / angle_step
    ).astype(int)

    valid_indices = (
        (indices >= 0) &
        (indices < NUM_SECTORS)
    )

    np.minimum.at(
        nearest,
        indices[valid_indices],
        distance[valid_indices]
    )

    return nearest, sector_angles


# ==========================================
# CREATE SAFETY BUBBLES
# ==========================================

def create_safety_bubbles(nearest, sector_angles):

    blocked = np.zeros(
        NUM_SECTORS,
        dtype=bool
    )

    angle_step = (
        sector_angles[1] - sector_angles[0]
    )

    for i, obstacle_distance in enumerate(nearest):

        if obstacle_distance >= SAFETY_DISTANCE:
            continue

        # Expand the obstacle to account for
        # the robot's physical width.
        radius_ratio = min(
            0.99,
            ROBOT_RADIUS / max(obstacle_distance, 0.1)
        )

        bubble_angle = np.arcsin(radius_ratio)

        bubble_sectors = int(
            np.ceil(bubble_angle / angle_step)
        )

        start = max(
            0,
            i - bubble_sectors
        )

        end = min(
            NUM_SECTORS,
            i + bubble_sectors + 1
        )

        blocked[start:end] = True

    return blocked


# ==========================================
# FIND THE LARGEST GAP
# ==========================================

def find_best_gap(blocked, sector_angles):

    free = ~blocked

    gaps = []
    start = None

    for i, is_free in enumerate(free):

        if is_free and start is None:
            start = i

        elif not is_free and start is not None:

            gaps.append((start, i - 1))
            start = None

    if start is not None:
        gaps.append((start, len(free) - 1))

    # Require a gap at least three sectors wide.
    gaps = [
        gap for gap in gaps
        if gap[1] - gap[0] + 1 >= 3
    ]

    if not gaps:
        return None

    # Prefer wide gaps, but slightly favor
    # directions closer to straight ahead.
    def gap_score(gap):

        start, end = gap

        width = end - start + 1

        center = (
            sector_angles[start] +
            sector_angles[end]
        ) / 2

        straight_penalty = abs(center) / np.radians(75)

        return width - 0.2 * straight_penalty

    best_gap = max(gaps, key=gap_score)

    center_index = (
        best_gap[0] + best_gap[1]
    ) // 2

    return float(sector_angles[center_index])


# ==========================================
# MOVEMENT DECISION
# ==========================================

def calculate_movement(target_angle, nearest):

    if target_angle is None:
        return 0.0, 0.0

    center_distance = nearest[NUM_SECTORS // 2]

    # Rotate toward the selected gap.
    wz = np.clip(
        TURN_GAIN * target_angle,
        -MAX_TURN_SPEED,
        MAX_TURN_SPEED
    )

    # Large turns should happen without
    # simultaneously walking forward.
    if abs(target_angle) > np.radians(40):
        return 0.0, float(wz)

    vx = FORWARD_SPEED

    # Slow down when approaching obstacles.
    if center_distance < 1.4:
        vx *= 0.5

    return float(vx), float(wz)


# ==========================================
# GO2 SPORT MODE COMMANDS
# ==========================================

async def send_move(conn, vx, wz):

    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {
                "x": vx,
                "y": 0.0,
                "z": wz
            }
        }
    )


async def stop_robot(conn):

    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["StopMove"]
        }
    )


# ==========================================
# INITIALIZE MOTION MODE
# ==========================================

async def initialize_motion(conn):

    print("Checking Go2 motion mode...")

    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"],
        {"api_id": 1001}
    )

    if response["data"]["header"]["status"]["code"] != 0:
        raise RuntimeError("Could not check motion mode.")

    import json

    mode_data = json.loads(response["data"]["data"])

    if mode_data["name"] != "normal":

        print("Switching to normal mode...")

        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {
                "api_id": 1002,
                "parameter": {"name": "normal"}
            }
        )

        await asyncio.sleep(5)

    print("Enabling Balance Stand...")

    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["BalanceStand"]}
    )

    await asyncio.sleep(2)


# ==========================================
# MAIN
# ==========================================

async def main():

    global latest_points
    global last_lidar_time

    if not ROBOT_IP:
        raise RuntimeError(
            "ROBOT_IP environment variable is not set."
        )

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP
    )

    connected = False

    try:

        print("Connecting to Go2...")

        await conn.connect()
        connected = True

        print("Connected!")

        await conn.datachannel.disableTrafficSaving(True)

        # Keep your working LibVoxel decoder.
        print("Selecting LibVoxel decoder...")

        conn.datachannel.set_decoder(
            decoder_type="libvoxel"
        )

        # Subscribe to LiDAR.
        conn.datachannel.pub_sub.subscribe(
            "rt/utlidar/voxel_map_compressed",
            lidar_callback
        )

        conn.datachannel.pub_sub.subscribe(
            "rt/utlidar/lidar_state",
            lambda message: print("LiDAR state received")
        )

        await asyncio.sleep(1)

        print("Enabling LiDAR...")

        for _ in range(5):

            conn.datachannel.pub_sub.publish_without_callback(
                "rt/utlidar/switch",
                "ON"
            )

            await asyncio.sleep(0.1)

        print("Waiting for point cloud...")

        # Wait up to 15 seconds for valid XYZ data.
        for _ in range(150):

            await asyncio.sleep(0.1)

            with data_lock:
                ready = latest_points is not None

            if ready:
                break

        if not ready:
            raise RuntimeError(
                "No decoded XYZ positions received."
            )

        print("LiDAR point cloud received!")

        if not DRY_RUN:

            await initialize_motion(conn)

        else:

            print("\nDRY RUN ENABLED")
            print("No movement commands will be sent.\n")

        print("Starting Follow the Gap controller...")

        while True:

            await asyncio.sleep(1.0 / CONTROL_RATE)

            with data_lock:

                if latest_points is None:
                    continue

                points = latest_points.copy()

                age = (
                    time.monotonic() -
                    last_lidar_time
                )

            # Fail-safe if LiDAR becomes stale.
            if age > LIDAR_TIMEOUT:

                print("STALE LIDAR: STOPPING")

                if not DRY_RUN:
                    await stop_robot(conn)

                continue

            # Process point cloud.
            nearest, sector_angles = process_lidar(points)

            # Expand obstacles into safety bubbles.
            blocked = create_safety_bubbles(
                nearest,
                sector_angles
            )

            # Select the largest open gap.
            target_angle = find_best_gap(
                blocked,
                sector_angles
            )

            # Calculate movement.
            vx, wz = calculate_movement(
                target_angle,
                nearest
            )

            if target_angle is None:

                print("No safe gap detected.")

            else:

                print(
                    f"Target: {np.degrees(target_angle):.1f}° | "
                    f"Forward: {vx:.2f} | "
                    f"Turn: {wz:.2f} | "
                    f"Points: {len(points)}"
                )

            # Only send commands when explicitly enabled.
            if not DRY_RUN:

                if target_angle is None:
                    await stop_robot(conn)

                else:
                    await send_move(conn, vx, wz)

    except KeyboardInterrupt:

        print("\nStopping controller...")

    finally:

        if connected:

            try:
                await stop_robot(conn)
            except Exception:
                pass

            print("Closing WebRTC connection...")

            await conn.disconnect()

        print("Program ended.")


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nStopped.")
