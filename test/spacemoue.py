"""
SpaceMouse -> target pose producer.
  * connects to the Mac SpaceMouse bridge over the socket (localhost:5005),
    same transport as your existing sender;
  * turns the 6 axes (each in [-1, 1]) into a Cartesian velocity twist;
  * integrates it into a target pose every loop:  p = p + v*dt  (and the
    orientation by the angular velocity);
  * reads the two buttons -> continuous 0..1 gripper closure fraction;
  * constanty overwrites a shared memory buffer with (pos, quat, gripper).
"""

from __future__ import annotations

import os
import math
import mmap
import time
import json
import socket
import argparse

import numpy as np
from evdev import InputDevice, ecodes, list_devices


# Tuning
MAC_HOST = "127.0.0.1"
MAC_PORT = 5005 # SpaceMouse bridge (SSH -R tunnel from the Mac)
SPACEMOUSE_DEVICE = None # None = automatically find 3Dconnexion SpaceMouse on Linux

LIN_SPEED = 0.5 # m/s of TCP motion at full stick deflection
ANG_SPEED = 1.20 # rad/s of TCP rotation at full twist
DEADZONE = 0.10 # ignore |axis| below this
MAX_DT = 0.025 # cap integration dt; prevents one delayed loop from creating a large pose jump
INPUT_STALE_TIMEOUT = 0.12 # s; no fresh SpaceMouse packet -> velocity = 0

LIN_SIGN = np.array([+1.0, -1.0, -1.0], dtype=np.float64) # (x, y, z)
ANG_SIGN = np.array([+1.0, -1.0, -1.0], dtype=np.float64) # (roll, pitch, yaw)

LINEAR_FRAME = "world" # "world" (fixed axes, top-down friendly) or "body"
ANGULAR_FRAME = "world"

TWO_BUTTON_MODE = True
BUTTON_CLOSE_INDEX = 0
BUTTON_OPEN_INDEX = 1
GRIP_SPEED = 0.55 # closure-fraction / second while a button is held

LOOP_HZ = 200.0 # producer update rate.
BOX_HALF = np.array([0.6, 0.6, 0.5], dtype=np.float64)

# Shared memory target buffer
SHARED_PATH_DEFAULT = "/tmp/sm_teleop_target.bin"


class SharedTarget:
    N = 10
    SIZE = N * 8

    def __init__(self, path=SHARED_PATH_DEFAULT):
        self.path = path
        if (not os.path.exists(path)) or os.path.getsize(path) != self.SIZE:
            with open(path, "wb") as f:
                f.write(b"\x00" * self.SIZE)
        self.f = open(path, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), self.SIZE)
        self.arr = np.ndarray((self.N,), dtype=np.float64, buffer=self.mm)  # writable
        self._last = None

    def write(self, pos, quat, grip, ready=1.0):
        seq = float(self.arr[0])
        self.arr[0] = seq + 1.0            
        self.arr[1:4] = pos
        self.arr[4:8] = quat
        self.arr[8] = float(grip)
        self.arr[9] = float(ready)
        self.arr[0] = seq + 2.0           

    def read(self):
        for _ in range(16):
            s1 = float(self.arr[0])
            if int(s1) & 1:
                continue
            pos = np.array(self.arr[1:4], dtype=np.float64)
            quat = np.array(self.arr[4:8], dtype=np.float64)
            grip = float(self.arr[8])
            ready = float(self.arr[9])
            if float(self.arr[0]) == s1:
                self._last = (pos, quat, grip, ready)
                return pos, quat, grip, ready
        if self._last is not None:
            return self._last
        return (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 0.0, 0.0)

    def is_ready(self):
        return float(self.arr[9]) >= 0.5

    def close(self):
        try:
            self.mm.close()
            self.f.close()
        except Exception:
            pass


# quaternion helpers (xyzw)
def _norm4(q):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    s = math.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)])


def _rotate_vec(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], dtype=np.float64)


def _shape_axis(v, dz):
    a = abs(v)
    if a < dz:
        return 0.0
    return math.copysign((a - dz) / (1.0 - dz), v)


# Direct Linux SpaceMouse reader
class SpaceMouse:
    def __init__(self, device_path=SPACEMOUSE_DEVICE):
        self.ok = False
        self.device = None
        self.device_path = device_path
        self._lin = np.zeros(3, dtype=np.float64)
        self._ang = np.zeros(3, dtype=np.float64)
        self._buttons = [0, 0]
        self._last_debug = 0.0
        self._last_packet_time = None

        # SpaceMouse Compact reports relative values whose practical full range
        # is usually around +/-350. Normalize them to roughly [-1, 1].
        self.axis_scale = 350.0

        try:
            if self.device_path is None:
                self.device_path = self._find_spacemouse()

            if self.device_path is None:
                raise RuntimeError("No 3Dconnexion SpaceMouse found under /dev/input/event*")

            self.device = InputDevice(self.device_path)
            self.ok = True
            print(f"[SpaceMouse] Connected directly to Linux device {self.device_path}.")
            print(f"[SpaceMouse] Device name: {self.device.name}")
            print(f"[SpaceMouse] button {BUTTON_CLOSE_INDEX}=close, "
                  f"button {BUTTON_OPEN_INDEX}=open.")
        except Exception as e:
            print(f"[SpaceMouse] Could not open Linux SpaceMouse: {e}")
            print("[SpaceMouse] Target will HOLD until the device is available.")
            self.ok = False

    @staticmethod
    def _find_spacemouse():
        for path in list_devices():
            try:
                dev = InputDevice(path)
                name = (dev.name or "").lower()
                if "3dconnexion" in name or "spacemouse" in name:
                    dev.close()
                    return path
                dev.close()
            except Exception:
                pass
        return None

    def _normalize(self, value):
        return float(np.clip(float(value) / self.axis_scale, -1.0, 1.0))

    def read(self):
        if not self.ok or self.device is None:
            return None

        # EV_REL samples are treated as this poll's motion command only..
        lin = np.zeros(3, dtype=np.float64)
        ang = np.zeros(3, dtype=np.float64)
        got_motion = False
        got_any = False

        try:
            while True:
                event = self.device.read_one()
                if event is None:
                    break

                if event.type == ecodes.EV_REL:
                    value = self._normalize(event.value)

                    if event.code == ecodes.REL_X:
                        lin[0] = value
                        got_motion = True
                        got_any = True
                    elif event.code == ecodes.REL_Y:
                        lin[1] = value
                        got_motion = True
                        got_any = True
                    elif event.code == ecodes.REL_Z:
                        lin[2] = value
                        got_motion = True
                        got_any = True
                    elif event.code == ecodes.REL_RX:
                        ang[0] = value
                        got_motion = True
                        got_any = True
                    elif event.code == ecodes.REL_RY:
                        ang[1] = value
                        got_motion = True
                        got_any = True
                    elif event.code == ecodes.REL_RZ:
                        ang[2] = value
                        got_motion = True
                        got_any = True

                elif event.type == ecodes.EV_KEY:
                    if event.code in (ecodes.BTN_0, ecodes.BTN_LEFT):
                        self._buttons[0] = 1 if event.value else 0
                        got_any = True
                    elif event.code in (ecodes.BTN_1, ecodes.BTN_RIGHT):
                        self._buttons[1] = 1 if event.value else 0
                        got_any = True

        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[SpaceMouse] Linux read failed: {e!r}")
            self.ok = False
            return None

        now = time.perf_counter()
        if got_any:
            self._last_packet_time = now

        # Save only for diagnostics; these values are not reused as future velocity.
        self._lin[:] = lin
        self._ang[:] = ang

        if got_any and now - self._last_debug >= 0.5:
            self._last_debug = now
            print(f"[RX] lin={np.round(lin,3).tolist()} "
                  f"ang={np.round(ang,3).tolist()} btn={self._buttons}")

        # No new motion event in this poll -> zero Cartesian velocity immediately.
        # Button state is still retained.
        if not got_motion:
            lin[:] = 0.0
            ang[:] = 0.0

        return lin, ang, list(self._buttons)

    def close(self):
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None
        self.ok = False


# main producer loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", type=str, default=SHARED_PATH_DEFAULT)
    ap.add_argument("--host", type=str, default=MAC_HOST)
    ap.add_argument("--port", type=int, default=MAC_PORT)
    ap.add_argument("--use-box", action="store_true",
                    help="clamp the target to a box around the seed pose")
    ap.add_argument("--wait-timeout", type=float, default=0.0,
                    help="seconds to wait for the sim to seed (0 = forever)")
    args = ap.parse_args()

    shared = SharedTarget(args.buffer)

    # Old Mac bridge method. Keep this line if you want to switch back later.
    # mouse = SpaceMouse(args.host, args.port)

    # Direct Linux SpaceMouse method.
    mouse = SpaceMouse(SPACEMOUSE_DEVICE)

    # Wait for the sim to seed the buffer, then adopt its pose so the
    # target starts exactly on the gripper tip.
    print(f"[target] waiting for the sim to seed the buffer ({args.buffer}) ...")
    t0 = time.perf_counter()
    while not shared.is_ready():
        time.sleep(0.05)
        if args.wait_timeout > 0 and (time.perf_counter() - t0) > args.wait_timeout:
            print("[target] no seed yet; starting from identity pose.")
            break

    seed_pos, seed_quat, seed_grip, ready = shared.read()
    if ready < 0.5:
        seed_pos = np.array([0.0, 0.0, 1.2], dtype=np.float64)
        seed_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        seed_grip = 0.0
    p = seed_pos.copy()
    q = _norm4(seed_quat)
    grip = float(np.clip(seed_grip, 0.0, 1.0))
    print(f"[target] seeded at p={np.round(p,3).tolist()} q={np.round(q,3).tolist()}")

    box_lo = p - BOX_HALF
    box_hi = p + BOX_HALF

    prev_toggle = 0
    last_wall = None
    last_dbg = 0.0
    period = 1.0 / LOOP_HZ

    try:
        while True:
            t = time.perf_counter()
            dt = period if last_wall is None else min(max(t - last_wall, 0.0), MAX_DT)
            last_wall = t

            reading = mouse.read()
            if reading is not None:
                lin_raw, ang_raw, buttons = reading
                lin = np.array([_shape_axis(v, DEADZONE) for v in lin_raw]) * LIN_SIGN
                ang = np.array([_shape_axis(v, DEADZONE) for v in ang_raw]) * ANG_SIGN
                v_lin = lin * LIN_SPEED
                w_ang = ang * ANG_SPEED

                # Integrate only the current valid velocity sample.
                # When input is neutral/stale, v_lin == 0, therefore p HOLDS.
                # translate: p = p_previous + v*dt
                world_v = _rotate_vec(q, v_lin) if LINEAR_FRAME == "body" else v_lin
                p = p + world_v * dt
                if args.use_box:
                    p = np.minimum(np.maximum(p, box_lo), box_hi)

                # rotate: integrate angular velocity into q
                ang_mag = float(np.linalg.norm(w_ang)) * dt
                if ang_mag > 1e-9:
                    axis = w_ang / np.linalg.norm(w_ang)
                    dq = _quat_from_axis_angle(axis, ang_mag)
                    if ANGULAR_FRAME == "body":
                        q = _norm4(_quat_mul(q, dq)) # current (x) delta
                    else:
                        q = _norm4(_quat_mul(dq, q)) # delta (x) current

                # Gripper command is continuous:
                # hold close -> grip moves toward 1.0
                # hold open  -> grip moves toward 0.0
                # release both -> hold the current amount of closure.
                def pressed(i):
                    return 1 if (0 <= i < len(buttons) and buttons[i]) else 0

                if TWO_BUTTON_MODE:
                    c = pressed(BUTTON_CLOSE_INDEX)
                    o = pressed(BUTTON_OPEN_INDEX)
                    if c and not o:
                        grip = min(1.0, grip + GRIP_SPEED * dt)
                    elif o and not c:
                        grip = max(0.0, grip - GRIP_SPEED * dt)
                else:
                    b = pressed(0)
                    if b and not prev_toggle:
                        # fallback toggle still changes the desired endpoint
                        grip = 0.0 if grip >= 0.5 else 1.0
                    prev_toggle = b

            # Constantly overwrite the current target position.
            shared.write(p, q, grip, ready=1.0)

            if t - last_dbg >= 0.5:
                last_dbg = t
                print(f"[target] p={np.round(p,3).tolist()} grip={grip:0.2f}")

            sleep_left = period - (time.perf_counter() - t)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\n[target] stopping.")
    finally:
        mouse.close()
        shared.close()


if __name__ == "__main__":
    main()