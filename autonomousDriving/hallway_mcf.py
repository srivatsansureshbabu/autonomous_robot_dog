import argparse
import asyncio
import math
import json
import time
import random
import numpy as np

print("=== GO2 HALLWAY PROGRAM STARTING ===", flush=True)

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
    RTC_TOPIC,
    SPORT_CMD,
    SPORT_CMD_MCF,
)

ROBOT_IP_DEFAULT = "192.168.8.114"
LIDAR_TOPIC = "rt/utlidar/voxel_map_compressed"
LIDAR_SWITCH = "rt/utlidar/switch"

latest_points = None
latest_state = None
last_lidar_time = 0.0
last_state_time = 0.0

# Cache the most recent valid front distance so one sparse LiDAR frame
# does not immediately make the hallway controller think it is blind.
last_valid_front = None
last_valid_front_time = 0.0


def recursive_find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            result = recursive_find(v, key)
            if result is not None:
                return result
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            result = recursive_find(v, key)
            if result is not None:
                return result
    return None


def lidar_callback(message):
    global latest_points, last_lidar_time

    try:
        positions = recursive_find(message, "positions")
        resolution = recursive_find(message, "resolution")
        origin = recursive_find(message, "origin")

        if positions is None:
            return

        pts = np.asarray(positions, dtype=float)

        if pts.ndim == 1:
            usable = (len(pts) // 3) * 3
            pts = pts[:usable].reshape(-1, 3)
        else:
            pts = pts.reshape(-1, 3)

        # Convert voxel indexes -> meters when metadata is available.
        if resolution is not None and origin is not None:
            org = np.asarray(origin, dtype=float).reshape(3)
            pts = org + pts * float(resolution)

        # remove NaN/inf
        pts = pts[np.all(np.isfinite(pts), axis=1)]

        # remove duplicate mesh vertices
        if len(pts):
            rounded = np.round(pts, 3)
            _, idx = np.unique(rounded, axis=0, return_index=True)
            pts = pts[np.sort(idx)]

        latest_points = pts
        last_lidar_time = time.time()

    except Exception as e:
        print("LiDAR callback error:", e, flush=True)


def sport_callback(message):
    global latest_state, last_state_time
    try:
        if isinstance(message, dict):
            latest_state = message.get("data", message)
            last_state_time = time.time()
    except Exception as e:
        print("SportState callback error:", e, flush=True)


def get_yaw():
    if not isinstance(latest_state, dict):
        return None

    imu = latest_state.get("imu_state")
    if isinstance(imu, dict):
        rpy = imu.get("rpy")
        if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
            try:
                return float(rpy[2])
            except Exception:
                pass

    rpy = recursive_find(latest_state, "rpy")
    if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
        try:
            return float(rpy[2])
        except Exception:
            return None

    return None


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def get_position():
    if not isinstance(latest_state, dict):
        return None

    pos = latest_state.get("position")
    if pos is None:
        pos = recursive_find(latest_state, "position")

    try:
        arr = np.asarray(pos, dtype=float).reshape(-1)
        if len(arr) >= 2:
            return arr
    except Exception:
        pass

    return None


def cloud_to_robot_frame(points):
    """
    LiDAR metadata reports frame_id='odom' on this Go2 stream.

    Convert odom/world XYZ -> robot-local XYZ using SportState position + yaw.

    Robot-local convention after this transform:
      +X = robot forward (candidate to verify)
      +Y = robot left
      +Z = up

    We still run the axis probe because the LiDAR payload orientation can vary
    by decoder/firmware and we want to verify it empirically before movement.
    """
    if points is None or len(points) == 0:
        return None

    pos = get_position()
    yaw = get_yaw()

    if pos is None or yaw is None:
        return points.copy()

    pts = points.copy()

    # Translate odom/world cloud so robot is at origin.
    pts[:, 0] -= pos[0]
    pts[:, 1] -= pos[1]
    if len(pos) >= 3:
        pts[:, 2] -= pos[2]

    # Rotate world -> robot frame by -yaw.
    c = math.cos(yaw)
    s = math.sin(yaw)

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    pts[:, 0] = c * x + s * y
    pts[:, 1] = -s * x + c * y

    return pts


def remap(points, axis):
    """
    Return columns [forward, left, up].
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if axis == "+x":
        f, l = x, y
    elif axis == "-x":
        f, l = -x, -y
    elif axis == "+y":
        f, l = y, -x
    elif axis == "-y":
        f, l = -y, x
    else:
        raise ValueError("Bad axis")

    return np.column_stack((f, l, z))


def near_distance(values, minimum_points=8):
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < minimum_points:
        return None
    return float(np.percentile(values, 10))


def analyze(points, axis):
    """
    Robust hallway LiDAR analysis.

    Key idea:
      - Ignore the Go2's own near-field/body voxel clutter.
      - Use a normal central front corridor first.
      - Use a slightly wider fallback corridor only when necessary.
      - Briefly cache the last good front value across sparse voxel frames.
    """
    global last_valid_front, last_valid_front_time

    local = cloud_to_robot_frame(points)

    if local is None or len(local) == 0:
        now = time.time()
        cached = (
            last_valid_front
            if last_valid_front is not None
            and (now - last_valid_front_time) <= 1.0
            else None
        )
        return {
            "front": cached,
            "back": None,
            "left": None,
            "right": None,
            "front_source": "cache" if cached is not None else "none",
        }

    p = remap(local, axis)

    f = p[:, 0]
    l = p[:, 1]
    z = p[:, 2]

    # Return to the proven vertical window from the earlier hallway version.
    h = (z > -0.15) & (z < 0.75)

    radial = np.sqrt(f*f + l*l)

    # Stronger self/body exclusion than the previous robust version.
    # The false 0.43 m hit happened inside this near-field region.
    clear_of_robot = radial > 0.55

    # Primary front corridor.
    front_mask = (
        h &
        clear_of_robot &
        (f > 0.55) &
        (f < 3.5) &
        (np.abs(l) < 0.35)
    )

    # Wider fallback: still narrower than the previous ±0.70 m version.
    front_fallback_mask = (
        h &
        clear_of_robot &
        (f > 0.55) &
        (f < 3.5) &
        (np.abs(l) < 0.50)
    )

    front = near_distance(f[front_mask], minimum_points=3)
    front_source = "normal"

    if front is None:
        front = near_distance(f[front_fallback_mask], minimum_points=2)
        front_source = "wide"

    now = time.time()

    if front is not None:
        last_valid_front = front
        last_valid_front_time = now
    elif (
        last_valid_front is not None
        and (now - last_valid_front_time) <= 1.0
    ):
        front = last_valid_front
        front_source = "cache"
    else:
        front_source = "none"

    back_mask = (
        h &
        (radial > 0.50) &
        (f < -0.50) &
        (f > -3.5) &
        (np.abs(l) < 0.45)
    )

    side_window = (
        h &
        (radial > 0.50) &
        (f > -0.50) &
        (f < 1.20)
    )

    left_mask = (
        side_window &
        (l > 0.45) &
        (l < 2.5)
    )

    right_mask = (
        side_window &
        (l < -0.45) &
        (-l < 2.5)
    )

    return {
        "front": front,
        "back": near_distance(-f[back_mask], minimum_points=4),
        "left": near_distance(l[left_mask], minimum_points=5),
        "right": near_distance(-l[right_mask], minimum_points=5),
        "front_source": front_source,
    }


def fd(v):
    return "---" if v is None else f"{v:.2f}m"


async def get_motion_mode(conn):
    """Read current motion switcher mode, but do NOT try to change it."""
    try:
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1001},
        )
        status = (
            response.get("data", {})
            .get("header", {})
            .get("status", {})
            .get("code")
        )
        if status != 0:
            return None

        raw = response.get("data", {}).get("data", "{}")
        return json.loads(raw).get("name")
    except Exception:
        return None


async def sport_call(conn, api_id, parameter=None):
    payload = {"api_id": api_id}
    if parameter is not None:
        payload["parameter"] = parameter

    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        payload,
    )

    status = (
        response.get("data", {})
        .get("header", {})
        .get("status", {})
        .get("code")
    )
    raw = response.get("data", {}).get("data", "")
    return status, raw


async def stop_robot(conn):
    """
    MCF StopMove uses the regular request/reply call.
    """
    try:
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD_MCF["StopMove"]},
        )
        code = (
            response.get("data", {})
            .get("header", {})
            .get("status", {})
            .get("code")
        )
        return code
    except Exception as e:
        print("STOP warning:", e, flush=True)
        return None


def move_robot_mcf(conn, speed, turn):
    """
    IMPORTANT:
    MCF Move is a NO-REPLY command.

    This matches unitree_webrtc_connect's official sportmode_mcf example:
      - SPORT_CMD_MCF["Move"]
      - header.policy.noreply = True
      - parameter is JSON encoded
      - publish_without_callback()
    """
    generated_id = (
        int(time.time() * 1000) % 2147483648
        + random.randint(0, 1000)
    )

    request_payload = {
        "header": {
            "identity": {
                "id": generated_id,
                "api_id": SPORT_CMD_MCF["Move"],
            },
            "policy": {
                "priority": 0,
                "noreply": True,
            },
        },
        "parameter": json.dumps(
            {
                "x": float(speed),
                "y": 0.0,
                "z": float(turn),
            }
        ),
        "binary": [],
    }

    conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["SPORT_MOD"],
        request_payload,
    )


async def connect_robot(ip):
    print(f"Connecting to Go2 at {ip} ...", flush=True)

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ip,
    )

    await conn.connect()
    print("WebRTC CONNECTED", flush=True)

    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["LF_SPORT_MOD_STATE"],
        sport_callback,
    )

    await conn.datachannel.disableTrafficSaving(True)

    # Use the decoder we already know works on your robot.
    conn.datachannel.set_decoder(decoder_type="libvoxel")

    conn.datachannel.pub_sub.publish_without_callback(
        LIDAR_SWITCH,
        "on",
    )

    conn.datachannel.pub_sub.subscribe(
        LIDAR_TOPIC,
        lidar_callback,
    )

    print("LiDAR ON", flush=True)
    print("Topic:", LIDAR_TOPIC, flush=True)
    print("Decoder: libvoxel", flush=True)

    return conn


async def wait_for_data(seconds=10):
    print("Waiting for SportState + LiDAR...", flush=True)

    start = time.time()

    while time.time() - start < seconds:
        if latest_points is not None and latest_state is not None:
            print(
                f"Data ready. Clean LiDAR points: {len(latest_points)}",
                flush=True,
            )
            return True

        await asyncio.sleep(0.1)

    print("Timed out waiting for data.", flush=True)
    print(
        f"SportState received: {latest_state is not None} | "
        f"LiDAR received: {latest_points is not None}",
        flush=True,
    )
    return False


async def probe(conn):
    print("", flush=True)
    print("==============================================", flush=True)
    print("PROBE MODE - ROBOT WILL NOT MOVE", flush=True)
    print("==============================================", flush=True)

    if not await wait_for_data():
        return

    while True:
        pts = latest_points

        if pts is None:
            await asyncio.sleep(0.2)
            continue

        print("", flush=True)
        pos = get_position()
        if pos is not None:
            print(
                f"Robot SportState position: "
                f"({pos[0]:.3f}, {pos[1]:.3f}"
                + (f", {pos[2]:.3f})" if len(pos) >= 3 else ")"),
                flush=True,
            )
        print(f"Robot yaw: {get_yaw()}", flush=True)
        print("LiDAR cloud transformed from odom -> robot frame.", flush=True)
        print("Put a box/chair about 0.7-1.0 m DIRECTLY", flush=True)
        print("in front of the Go2 and compare FRONT values.", flush=True)
        print("", flush=True)
        print("AXIS    FRONT     LEFT      RIGHT     BACK", flush=True)

        for axis in ("+x", "-x", "+y", "-y"):
            r = analyze(pts, axis)
            print(
                f"{axis:>3}     "
                f"{fd(r['front']):>7}   "
                f"{fd(r['left']):>7}   "
                f"{fd(r['right']):>7}   "
                f"{fd(r['back']):>7}",
                flush=True,
            )

        print(f"Yaw: {get_yaw()}", flush=True)
        print("Ctrl+C when finished.", flush=True)

        await asyncio.sleep(1.5)


async def run_hallway(conn, axis, distance, speed):
    print("", flush=True)
    print("==============================================", flush=True)
    print("AUTONOMOUS HALLWAY TEST - CLASSIC WALK V2 + ROBUST LIDAR", flush=True)
    print("==============================================", flush=True)
    print(f"Forward axis: {axis}", flush=True)
    print(f"Target forward progress: {distance:.2f} m", flush=True)
    print(f"Speed command: {speed:.2f} m/s", flush=True)
    print("Obstacle stop threshold: 0.75 m", flush=True)
    print("Ctrl+C = STOP", flush=True)
    print("", flush=True)

    if not await wait_for_data():
        return

    # This Go2 reports MCF mode. Use MCF commands directly.
    current_mode = await get_motion_mode(conn)
    print(f"Current motion mode: {current_mode}", flush=True)

    if current_mode != "mcf":
        print(
            "WARNING: robot is not reporting MCF mode. "
            "This mission is configured for MCF commands.",
            flush=True,
        )

    # Use ClassicWalk instead of TrotRun so the Go2 uses a normal
    # walking gait while the same repeated Move commands provide velocity.
    stand_api = SPORT_CMD_MCF.get("RecoveryStand", 1006)
    classic_walk_api = SPORT_CMD_MCF.get("ClassicWalk", 2049)

    print("Preparing locomotion: RecoveryStand -> ClassicWalk...", flush=True)

    stand_code, stand_raw = await sport_call(conn, stand_api)
    print(
        f"RecoveryStand: code={stand_code} raw={stand_raw!r}",
        flush=True,
    )
    if stand_code != 0:
        print("STOP: RecoveryStand was rejected.", flush=True)
        return

    await asyncio.sleep(1.2)

    walk_code, walk_raw = await sport_call(
        conn,
        classic_walk_api,
        {"data": True},
    )
    print(
        f"ClassicWalk ON: code={walk_code} raw={walk_raw!r}",
        flush=True,
    )
    if walk_code != 0:
        print(
            "STOP: ClassicWalk was rejected. "
            "Do not fall back to TrotRun automatically.",
            flush=True,
        )
        return

    await asyncio.sleep(0.8)

    yaw_target = get_yaw()
    start_pos = get_position()

    if yaw_target is None:
        print("No IMU yaw available. Aborting.", flush=True)
        return

    if start_pos is None:
        print("No SportState position available. Aborting.", flush=True)
        return

    start_xy = np.asarray(start_pos[:2], dtype=float).copy()

    # World-frame unit vectors aligned with initial robot heading.
    forward_unit = np.array(
        [math.cos(yaw_target), math.sin(yaw_target)],
        dtype=float,
    )
    left_unit = np.array(
        [-math.sin(yaw_target), math.cos(yaw_target)],
        dtype=float,
    )

    print(
        f"Start odometry: X={start_xy[0]:.3f}, Y={start_xy[1]:.3f}, "
        f"yaw={yaw_target:.3f}",
        flush=True,
    )

    start_time = time.time()
    hard_timeout = max(12.0, (distance / max(speed, 0.01)) * 3.0)

    best_forward = 0.0
    last_progress_time = start_time
    last_move_code = None
    front_missing_cycles = 0

    try:
        while True:
            now = time.time()

            if now - last_lidar_time > 1.0:
                print("\nSTOP: LiDAR data became stale.", flush=True)
                break

            if now - last_state_time > 1.0:
                print("\nSTOP: SportState data became stale.", flush=True)
                break

            current_pos = get_position()
            current_yaw = get_yaw()

            if current_pos is None or current_yaw is None:
                print("\nSTOP: position/yaw missing.", flush=True)
                break

            current_xy = np.asarray(current_pos[:2], dtype=float)
            delta = current_xy - start_xy

            # Distance actually made in the original forward direction.
            forward_progress = float(np.dot(delta, forward_unit))

            # Lateral drift relative to original hallway heading.
            lateral_progress = float(np.dot(delta, left_unit))

            if forward_progress > best_forward + 0.01:
                best_forward = forward_progress
                last_progress_time = now

            if forward_progress >= distance:
                print(
                    f"\nMISSION COMPLETE: measured forward progress "
                    f"{forward_progress:.2f} m.",
                    flush=True,
                )
                break

            # Give the Go2 more time than V3 because position updates / gait
            # can be step-like at low speed.
            if (now - start_time) > 5.0 and (now - last_progress_time) > 5.0:
                print(
                    "\nSTOP: no forward odometry progress for 5 seconds. "
                    f"Move path={last_move_code}.",
                    flush=True,
                )
                break

            if now - start_time >= hard_timeout:
                print(
                    f"\nSTOP: hard timeout. Forward progress="
                    f"{forward_progress:.2f} m.",
                    flush=True,
                )
                break

            pts = latest_points
            if pts is None:
                print("\nSTOP: LiDAR cloud missing.", flush=True)
                break

            r = analyze(pts, axis)
            front = r["front"]
            left = r["left"]
            right = r["right"]

            # Robust front-data handling:
            # analyze() already tries normal, wide, and recent-cache front data.
            # Only stop if all of those fail continuously for several seconds.
            if front is None:
                front_missing_cycles += 1
                print(
                    f"\rWaiting for usable front LiDAR "
                    f"({front_missing_cycles}/60)...",
                    end="",
                    flush=True,
                )
                if front_missing_cycles >= 60:
                    print(
                        "\nSTOP: front LiDAR remained unavailable for ~3 seconds.",
                        flush=True,
                    )
                    break
                await asyncio.sleep(0.05)
                continue
            else:
                front_missing_cycles = 0

            if front <= 0.75:
                print(
                    f"\nSTOP: obstacle ahead at {front:.2f} m "
                    f"(front source={r.get('front_source', '?')})",
                    flush=True,
                )
                break

            if left is not None and left <= 0.30:
                print(
                    f"\nSTOP: left wall too close at {left:.2f} m",
                    flush=True,
                )
                break

            if right is not None and right <= 0.30:
                print(
                    f"\nSTOP: right wall too close at {right:.2f} m",
                    flush=True,
                )
                break

            yaw_error = normalize_angle(yaw_target - current_yaw)

            # Heading hold is primary.
            turn = 0.90 * yaw_error

            # Gentle hallway centering.
            if left is not None and right is not None:
                wall_error = left - right
                turn += 0.25 * wall_error

            turn = max(-0.20, min(0.20, turn))

            # MCF Move is fire-and-forget by design; no response is expected.
            move_robot_mcf(conn, speed, turn)
            last_move_code = "MCF-no-reply"

            print(
                "\r"
                f"FWD={forward_progress:.2f}/{distance:.2f}m  "
                f"LAT={lateral_progress:+.2f}m  "
                f"pos=({current_xy[0]:+.2f},{current_xy[1]:+.2f})  "
                f"front={fd(front):>7}/{r.get('front_source','?'):<6}  "
                f"L={fd(left):>7}  "
                f"R={fd(right):>7}  "
                f"yawErr={yaw_error:+.3f}  "
                f"turn={turn:+.3f}  "
                f"move={last_move_code}",
                end="",
                flush=True,
            )

            await asyncio.sleep(0.05)

    finally:
        await stop_robot(conn)

        # Return ClassicWalk mode to OFF when the hallway mission ends.
        try:
            classic_walk_api = SPORT_CMD_MCF.get("ClassicWalk", 2049)
            walk_off_code, walk_off_raw = await sport_call(
                conn,
                classic_walk_api,
                {"data": False},
            )
            print(
                f"ClassicWalk OFF: code={walk_off_code} "
                f"raw={walk_off_raw!r}",
                flush=True,
            )
        except Exception as e:
            print(
                f"ClassicWalk OFF warning: {e}",
                flush=True,
            )

        final_pos = get_position()
        if final_pos is not None:
            final_xy = np.asarray(final_pos[:2], dtype=float)
            delta = final_xy - start_xy
            final_forward = float(np.dot(delta, forward_unit))
            final_lateral = float(np.dot(delta, left_unit))

            print(
                f"\nFinal progress: forward={final_forward:.3f} m, "
                f"lateral={final_lateral:+.3f} m",
                flush=True,
            )

        print("Robot STOPPED.", flush=True)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ip",
        default=ROBOT_IP_DEFAULT,
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--probe",
        action="store_true",
        help="LiDAR direction test. Robot does not move.",
    )

    group.add_argument(
        "--run",
        action="store_true",
        help="Run autonomous hallway test.",
    )

    parser.add_argument(
        "--forward-axis",
        choices=["+x", "-x", "+y", "-y"],
        default="+x",
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    # Safe default.
    if not args.run:
        args.probe = True

    conn = None

    try:
        conn = await connect_robot(args.ip)

        if args.probe:
            await probe(conn)
        else:
            await run_hallway(
                conn,
                args.forward_axis,
                max(0.25, min(args.distance, 3.0)),
                max(0.10, min(args.speed, 0.20)),
            )

    except KeyboardInterrupt:
        print("\nCTRL+C received.", flush=True)
        if conn is not None:
            await stop_robot(conn)

    except Exception as e:
        print(
            f"\nERROR: {type(e).__name__}: {e}",
            flush=True,
        )

        if conn is not None:
            await stop_robot(conn)

    finally:
        if conn is not None:
            try:
                conn.datachannel.pub_sub.publish_without_callback(
                    LIDAR_SWITCH,
                    "off",
                )
            except Exception:
                pass

            try:
                await conn.disconnect()
            except BaseException:
                pass

        print("Program finished.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())