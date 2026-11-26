"""
Go2 hallway walk: wall following with an emergency stop.

Based on the RoboRacer course:
  - Lecture 3 (frames):   LiDAR points are converted from the "odom" frame
                          into the robot's own frame (x = forward, y = left).
  - Lab 3 (wall follow):  fit a line to one wall, get its distance and angle,
                          look ahead, and use a PD controller to set turning speed.
  - Lab 2 (braking):      stop when time to collision ahead is too short.
Plus our own safety checks: battery, tilt (fall), stale data, max distance.

Usage (in Ubuntu, inside ~/unitree_webrtc_connect with the venv active):
  python hallway_walk.py                 # DRY RUN: robot does NOT move, prints what it would do
  python hallway_walk.py --go            # really walks
  python hallway_walk.py --go --wall left --distance 0.8 --max-distance 8

Stop at any time with Ctrl+C. Keep someone with the remote next to the robot.
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time

import numpy as np

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.12.1")

# ---------------------------------------------------------------- settings
WALK_SPEED = 0.3          # m/s forward. Keep slow while testing.
MAX_TURN = 0.5            # rad/s, largest turning speed allowed
KP = 1.2                  # how hard to correct distance error
KD = 0.3                  # damping: resists fast changes (stops wobble)
LOOKAHEAD = 0.5           # m, Lab 3 lookahead: correct for where we WILL be
KP_HEADING = 1.0          # straight mode: how hard to turn back to the starting direction
KP_LINE = 1.0             # straight mode: how hard to steer back onto the starting line

STOP_DISTANCE = 0.7       # m, always stop if something is this close ahead
TTC_LIMIT = 1.5           # s, Lab 2: stop if collision is closer than this in time
ROBOT_HALF_WIDTH = 0.30   # m, half the width of the "lane" checked ahead (with margin)

MIN_BATTERY = 20          # %, refuse to start below this
MAX_TILT = 0.45           # rad (~26 deg) roll or pitch = probably falling
DATA_TIMEOUT = 1.0        # s, stop if LiDAR or state data is older than this
FRONT_MIN_POINTS = 4      # Lab 2 false positives: an obstacle must be at least this many LiDAR points
OBSTACLE_CONFIRM = 3      # ...and be seen this many loops in a row (0.3 s) before stopping
WALL_LOST_TIME = 2.0      # s, walk straight this long without a wall (doorway), then stop
LOOP_HZ = 10              # control loop rate

# Which LiDAR points count, in the robot frame (meters)
Z_MIN, Z_MAX = -0.15, 0.40    # height band relative to the body (removes floor and ceiling)
SELF_RADIUS = 0.35            # ignore points this close (the robot's own legs/body)
WALL_X_MIN, WALL_X_MAX = -0.5, 1.5   # stretch of wall used for the line fit
WALL_Y_MAX = 2.5                     # wall must be within this distance to the side
MIN_WALL_POINTS = 15


# ---------------------------------------------------------------- shared state
class State:
    points_odom = None      # Nx3 LiDAR points in odom frame
    lidar_time = 0.0
    position = None         # [x, y, z] in odom frame
    rpy = None              # [roll, pitch, yaw]
    velocity = None         # [vx, vy, vz]
    state_time = 0.0
    battery = None


S = State()


def on_lidar(message):
    pts = message["data"]["data"].get("points")
    if pts is not None and len(pts):
        S.points_odom = np.asarray(pts, dtype=np.float64)
        S.lidar_time = time.time()


def on_sport_state(message):
    d = message["data"]
    S.position = d["position"]
    S.rpy = d["imu_state"]["rpy"]
    S.velocity = d["velocity"]
    S.state_time = time.time()


def on_low_state(message):
    S.battery = message["data"].get("bms_state", {}).get("soc")


# ---------------------------------------------------------------- Lecture 3: frames
def to_body_frame(points_odom, position, yaw):
    """Odom-frame points -> robot frame (x forward, y left, z up, origin at the body)."""
    d = points_odom - np.asarray(position, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    xb = c * d[:, 0] + s * d[:, 1]
    yb = -s * d[:, 0] + c * d[:, 1]
    return np.column_stack((xb, yb, d[:, 2]))


def usable_points(pb):
    """Keep points at body height and not on the robot itself."""
    keep = (pb[:, 2] > Z_MIN) & (pb[:, 2] < Z_MAX)
    keep &= np.hypot(pb[:, 0], pb[:, 1]) > SELF_RADIUS
    return pb[keep]


# ---------------------------------------------------------------- Lab 3: the wall
def measure_wall(pb, side, front=math.inf):
    """
    Fit a straight line to the wall on one side.
    Returns (distance, angle) or None if no wall is seen.
      distance: perpendicular distance to the wall (m)
      angle:    how much the robot is turned TOWARD the wall (rad, + = toward)
    """
    y = pb[:, 1] if side == "left" else -pb[:, 1]   # mirror so the wall is always at +y
    x = pb[:, 0]
    x_max = min(WALL_X_MAX, front - 0.4)   # don't mistake the wall ahead for the side wall
    sel = (x > WALL_X_MIN) & (x < x_max) & (y > 0.1) & (y < WALL_Y_MAX)
    if sel.sum() < MIN_WALL_POINTS:
        return None
    xs, ys = x[sel], y[sel]
    m, c = np.polyfit(xs, ys, 1)            # wall: y = m*x + c
    resid = ys - (m * xs + c)
    inner = resid < 0.15                    # drop far outliers once, refit
    if inner.sum() >= MIN_WALL_POINTS:
        m, c = np.polyfit(xs[inner], ys[inner], 1)
    angle = -math.atan(m)                   # wall getting closer ahead => turned toward it
    distance = c * math.cos(math.atan(m))
    return distance, angle


# ---------------------------------------------------------------- Lab 2: what's ahead
def front_distance(pb):
    """Distance to what's straight ahead, ignoring single stray LiDAR points.
    Returns (distance, closest_point) where closest_point is (x, y, z) or None."""
    ahead = pb[(pb[:, 0] > 0) & (np.abs(pb[:, 1]) < ROBOT_HALF_WIDTH)]
    if len(ahead) < FRONT_MIN_POINTS:
        return math.inf, None
    order = np.argsort(ahead[:, 0])
    nth = ahead[order[FRONT_MIN_POINTS - 1]]      # the 4th-closest point, not a lone outlier
    return float(nth[0]), tuple(round(float(v), 2) for v in ahead[order[0]])


def time_to_collision(dist, speed):
    return dist / speed if speed > 0.01 else math.inf


# ---------------------------------------------------------------- robot commands
async def send_move(conn, vx, vyaw):
    try:
        await asyncio.wait_for(
            conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["Move"], "parameter": {"x": vx, "y": 0, "z": vyaw}},
            ),
            timeout=0.5,
        )
    except asyncio.TimeoutError:
        pass


async def stop_robot(conn):
    await send_move(conn, 0.0, 0.0)
    try:
        await asyncio.wait_for(
            conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StopMove"]}
            ),
            timeout=1.0,
        )
    except asyncio.TimeoutError:
        pass


async def ensure_normal_mode(conn):
    resp = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
    )
    mode = None
    if resp["data"]["header"]["status"]["code"] == 0:
        mode = json.loads(resp["data"]["data"])["name"]
    print(f"Motion mode: {mode}")
    if mode != "normal":
        print("Switching to 'normal' mode, wait 5 s...")
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1002, "parameter": {"name": "normal"}}
        )
        await asyncio.sleep(5)


# ---------------------------------------------------------------- main loop
async def main(args):
    live = args.go
    print("=" * 60)
    print("LIVE RUN: the robot WILL walk." if live else "DRY RUN: the robot will NOT move.")
    if args.mode == "straight":
        print(f"Walking straight in the starting direction, up to {args.max_distance:.1f} m.")
    else:
        keep = f"{args.distance:.2f} m" if args.distance else "its starting distance"
        print(f"Following the {args.wall} wall at {keep}, up to {args.max_distance:.1f} m.")
    print("=" * 60)

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    await conn.connect()

    # LiDAR in meters (native decoder), plus robot state and battery
    await conn.datachannel.disableTrafficSaving(True)
    conn.datachannel.set_decoder(decoder_type="native")
    conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "on")
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["ULIDAR_ARRAY"], on_lidar)
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], on_sport_state)
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LOW_STATE"], on_low_state)

    print("Waiting for LiDAR, position and battery data...")
    t0 = time.time()
    while S.points_odom is None or S.position is None or S.battery is None:
        if time.time() - t0 > 15:
            print("No data after 15 s. Check the connection and LiDAR. Exiting.")
            await conn.disconnect()
            return
        await asyncio.sleep(0.2)

    print(f"Battery: {S.battery}%")
    if live and S.battery < MIN_BATTERY:
        print(f"Battery below {MIN_BATTERY}%. Charge the robot first. Exiting.")
        await conn.disconnect()
        return

    if live:
        await ensure_normal_mode(conn)

    start = list(S.position)
    start_yaw = S.rpy[2]
    target = args.distance          # wall mode: None = keep the starting distance
    prev_error, prev_t = None, time.time()
    wall_lost_since = None
    obstacle_count = 0
    loop = 0
    reason = "finished"

    try:
        while True:
            now = time.time()

            # --- safety: fresh data, battery, tilt, distance travelled
            if now - S.lidar_time > DATA_TIMEOUT or now - S.state_time > DATA_TIMEOUT:
                reason = "data stopped arriving"
                break
            if S.battery is not None and S.battery < MIN_BATTERY - 5:
                reason = f"battery low ({S.battery}%)"
                break
            roll, pitch, yaw = S.rpy
            if abs(roll) > MAX_TILT or abs(pitch) > MAX_TILT:
                reason = "robot is tilting (fall?)"
                break
            travelled = math.hypot(S.position[0] - start[0], S.position[1] - start[1])
            if travelled >= args.max_distance:
                reason = f"walked {travelled:.1f} m"
                break

            # --- Lecture 3: LiDAR into the robot frame
            pb = usable_points(to_body_frame(S.points_odom, S.position, yaw))

            # --- Lab 2: emergency stop
            front, closest = front_distance(pb)
            ttc = time_to_collision(front, WALK_SPEED)
            if front < STOP_DISTANCE or ttc < TTC_LIMIT:
                obstacle_count += 1
                if obstacle_count >= OBSTACLE_CONFIRM:
                    reason = (f"obstacle/end of hallway {front:.2f} m ahead (TTC {ttc:.1f} s), "
                              f"closest point x,y,z = {closest}")
                    break
            else:
                obstacle_count = 0

            # --- straight mode: hold the starting direction using the IMU heading
            if args.mode == "straight":
                heading_err = math.atan2(math.sin(start_yaw - yaw), math.cos(start_yaw - yaw))
                dx, dy = S.position[0] - start[0], S.position[1] - start[1]
                side = -dx * math.sin(start_yaw) + dy * math.cos(start_yaw)   # + = left of the line
                turn = KP_HEADING * heading_err - KP_LINE * side
                turn = max(-MAX_TURN, min(MAX_TURN, turn))
                status = (f"heading off {math.degrees(heading_err):+5.1f} deg, "
                          f"{side:+.2f} m off the line")
                wall = "skip"
            else:
                # --- Lab 3: wall following
                wall = measure_wall(pb, args.wall, front)
            if wall == "skip":
                pass
            elif wall is None:
                # doorway or gap: keep straight for a short time, then give up
                wall_lost_since = wall_lost_since or now
                if now - wall_lost_since > WALL_LOST_TIME:
                    reason = "lost the wall"
                    break
                turn = 0.0
                status = "no wall (doorway?) -> straight"
            else:
                wall_lost_since = None
                dist, angle = wall
                if target is None:          # first good reading: keep this distance
                    target = dist
                    print(f"Keeping the starting distance: {target:.2f} m from the {args.wall} wall")
                future = dist - LOOKAHEAD * math.sin(angle)   # where we'll be after LOOKAHEAD m
                error = target - future                # + = too close to the wall
                dt = max(now - prev_t, 1e-3)
                d_err = 0.0 if prev_error is None else (error - prev_error) / dt
                prev_error, prev_t = error, now
                turn_away = KP * error + KD * d_err           # turn away from the wall if positive
                turn_away = max(-MAX_TURN, min(MAX_TURN, turn_away))
                # left turn is +yaw. Away from a right wall = left (+); away from a left wall = right (-)
                turn = turn_away if args.wall == "right" else -turn_away
                status = (f"wall {dist:.2f} m, angle {math.degrees(angle):+5.1f} deg, "
                          f"error {error:+.2f}")

            loop += 1
            if loop % 5 == 0:   # a log line twice per second, so you can scroll back
                print(f"{status:<48} front {front:5.2f} m | turn {turn:+.2f} rad/s | "
                      f"{travelled:4.1f} m", flush=True)

            if live:
                await send_move(conn, WALK_SPEED, turn)
            await asyncio.sleep(1.0 / LOOP_HZ)

    except (KeyboardInterrupt, asyncio.CancelledError):
        reason = "stopped by user"
    finally:
        if live:
            await stop_robot(conn)
        print(f"\nStopped: {reason}")
        await conn.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Go2 hallway wall following")
    p.add_argument("--go", action="store_true", help="actually move the robot (default: dry run)")
    p.add_argument("--wall", choices=["left", "right"], default="right", help="wall to follow")
    p.add_argument("--mode", choices=["straight", "wall"], default="straight",
                   help="straight = hold starting direction (default); wall = follow a wall")
    p.add_argument("--distance", type=float, default=None,
                   help="wall mode: distance to keep from the wall in m (default: the starting distance)")
    p.add_argument("--max-distance", type=float, default=8.0, help="stop after walking this far (m)")
    try:
        asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt:
        print("\nStopped by user")
        sys.exit(0)
