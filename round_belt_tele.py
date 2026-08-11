"""
SpaceMouse tele-operation of the UR10 + Robotiq gripper in the round-belt scene.

Every frame we read the SpaceMouse (6-DoF), treat the six axes as an end-effector
twist (linear + angular velocity), integrate it into the current TCP target pose,
and hand that pose to the unchanged `set_task_target(pos, quat, gripper_values)`.
The IK solver already tracks whatever target we give it, so the arm follows.

Gripper: the two SpaceMouse buttons are a binary open/close (no partial closing),
matching the two gripper presets the base class already computes
(`gripper_open_values` / `gripper_closed_values`).
"""

from __future__ import annotations

import math
import time
import json
import socket

import numpy as np

import newton
import newton.examples

from round_belt import (
    BELT_APPROACH_POS,
    GRIPPER_DOWN_QUAT,
    Example,
)

# Tuning knobs
LIN_SPEED = 0.12 # m/s of TCP motion at full stick deflection
ANG_SPEED = 1.20 # rad/s of TCP rotation at full twist
DEADZONE = 0.10 # ignore |axis| below this (device noise floor)
MAX_DT = 0.10 # cap integration dt so a slow physics frame can't jump

# Per-axis sign flips. Order is (x, y, z) for translation and
# (roll, pitch, yaw) for rotation.
LIN_SIGN = np.array([+1.0, +1.0, +1.0], dtype=np.float64)
ANG_SIGN = np.array([+1.0, +1.0, +1.0], dtype=np.float64)

# "world" : x/y/z move along world axes, twist about world axes (intuitive
#            relative to the fixed camera -- recommended for a top-down grasp).
# "body" : motion/rotation is expressed in the gripper's own frame.
LINEAR_FRAME = "world"
ANGULAR_FRAME = "world"

# Integrate against real wall-clock time (so the arm moves at LIN_SPEED m/s of
# actual seconds even though this coupled sim runs slower than real time).
# Set False to integrate against the fixed sim frame_dt instead.
USE_WALLCLOCK_DT = True

# Keep the target inside a sane box so a shove can't fling the IK target to
# infinity.  Set to None to disable.  (min_xyz, max_xyz) in world metres.
WORKSPACE_BOX = (
    np.array([-0.65, -0.45, 0.72], dtype=np.float64), # min x,y,z
    np.array([+0.65, +0.55, 1.30], dtype=np.float64), # max x,y,z
)

# Button mapping. Two-button mode: one button closes, one opens (edge-triggered).
# Single-button mode: button 0 toggles open<->close on each press.
TWO_BUTTON_MODE = True
BUTTON_CLOSE_INDEX = 0 # left button closes / grasps
BUTTON_OPEN_INDEX = 1 # right button opens / releases

# GRIPPER_DOWN_QUAT in round_belt.py is written scalar-first: (w, x, y, z).
# Warp/Newton quaternions are (x, y, z, w).  This flag tells the teleop how to
# read/write that 4-vector so incremental rotations compose correctly.
TARGET_QUAT_IS_WXYZ = True


# Quaternion helpers
def _norm4(q):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _quat_mul_xyzw(a, b):
    """Hamilton product a (x) b, both xyzw.  Rotating a vector by the result
    applies b first, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quat_from_axis_angle_xyzw(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    s = math.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)])


def _rotate_vec_xyzw(q, v):
    """Rotate vec3 v by quaternion q (xyzw)."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], dtype=np.float64)


def _target_to_xyzw(q4):
    """Convert a round_belt-layout quaternion to internal xyzw."""
    if TARGET_QUAT_IS_WXYZ:
        w, x, y, z = q4
        return _norm4([x, y, z, w])
    return _norm4(q4)


def _xyzw_to_target(q_xyzw):
    """Convert internal xyzw back to round_belt layout for set_task_target."""
    x, y, z, w = q_xyzw
    if TARGET_QUAT_IS_WXYZ:
        return (float(w), float(x), float(y), float(z))
    return (float(x), float(y), float(z), float(w))


def _shape_axis(v, dz):
    """Deadzone + rescale so motion ramps from 0 at the deadzone edge."""
    a = abs(v)
    if a < dz:
        return 0.0
    return math.copysign((a - dz) / (1.0 - dz), v)


# SpaceMouse wrapper
#
# IMPORTANT: Newton runs on the remote Linux machine, while the physical
# SpaceMouse is connected to the Mac.  Therefore this class does NOT try to
# open a local HID device.  It receives SpaceMouse samples from the Mac sender
# through the SSH reverse tunnel on localhost:5005.
class SpaceMouse:
    def __init__(self, host="127.0.0.1", port=5005):
        self.ok = False
        self.sock = None
        self._buffer = b""

        self._lin = np.zeros(3, dtype=np.float64)
        self._ang = np.zeros(3, dtype=np.float64)
        self._buttons = [0, 0]

        self._packet_count = 0
        self._last_debug = 0.0

        try:
            self.sock = socket.create_connection((host, port), timeout=3.0)
            self.sock.setblocking(False)
            self.ok = True
            print(f"[SpaceMouse] Connected to Mac bridge at {host}:{port}.")
            print("[SpaceMouse] Push/tilt to move TCP; "
                  f"button {BUTTON_CLOSE_INDEX}=close, "
                  f"button {BUTTON_OPEN_INDEX}=open.")
        except Exception as e:
            print(f"[SpaceMouse] Could not connect to Mac bridge at {host}:{port}: {e}")
            print("[SpaceMouse] Arm will hold. Check the Mac sender and SSH -R tunnel.")
            self.ok = False

    def read(self):
        """Return (lin[3], ang[3], buttons[list]) from the Mac bridge."""
        if not self.ok or self.sock is None:
            return None

        # Drain every byte currently available without blocking the simulation.
        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    print("[SpaceMouse] Mac bridge disconnected.")
                    self.ok = False
                    return None
                self._buffer += chunk
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[SpaceMouse] socket read failed: {e!r}")
            self.ok = False
            return None

        # Consume complete newline-delimited JSON packets.  If several arrived
        # during one slow simulation frame, the last packet becomes the current
        # SpaceMouse state.
        got_packet = False
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
                self._lin = np.array(
                    [msg.get("x", 0.0), msg.get("y", 0.0), msg.get("z", 0.0)],
                    dtype=np.float64,
                )
                self._ang = np.array(
                    [msg.get("roll", 0.0), msg.get("pitch", 0.0), msg.get("yaw", 0.0)],
                    dtype=np.float64,
                )
                self._buttons = list(msg.get("buttons", [0, 0]) or [0, 0])
                self._packet_count += 1
                got_packet = True
            except Exception as e:
                print(f"[SpaceMouse] bad packet ignored: {e!r}")

        # Temporary diagnostic: at most twice per second print the actual values
        # reaching Newton.  Once everything works, these lines can be removed.
        now = time.perf_counter()
        if got_packet and now - self._last_debug >= 0.5:
            self._last_debug = now
            print(
                "[SpaceMouse RX] "
                f"lin={np.round(self._lin, 3).tolist()} "
                f"ang={np.round(self._ang, 3).tolist()} "
                f"buttons={self._buttons}"
            )

        return self._lin.copy(), self._ang.copy(), list(self._buttons)

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.ok = False


# The tele-op example
class SpaceMouseTeleopExample(Example):
    """Drive the gripper's TCP target with a 6-DoF SpaceMouse; buttons for grip."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)

        # Current TCP target pose, seeded to the base class's start pose.
        self._tcp_pos = np.asarray(BELT_APPROACH_POS, dtype=np.float64).copy()
        self._tcp_xyzw = _target_to_xyzw(GRIPPER_DOWN_QUAT)

        # Binary gripper state + button edge memory.
        self._gripper_closed = False
        self._prev_close_btn = 0
        self._prev_open_btn = 0
        self._prev_toggle_btn = 0

        self._mouse = SpaceMouse()
        self._last_wall = None

        # Push the seed pose once so t=0 matches the base class exactly.
        self.set_task_target(
            tuple(self._tcp_pos),
            _xyzw_to_target(self._tcp_xyzw),
            self.gripper_open_values,
        )

    # integration dt
    def _dt(self) -> float:
        if not USE_WALLCLOCK_DT:
            return float(self.frame_dt)
        now = time.perf_counter()
        if self._last_wall is None:
            self._last_wall = now
            return float(self.frame_dt)
        dt = now - self._last_wall
        self._last_wall = now
        return float(min(max(dt, 0.0), MAX_DT))

    # gripper button
    def _update_gripper(self, buttons):
        def pressed(idx):
            return 1 if (0 <= idx < len(buttons) and buttons[idx]) else 0

        if TWO_BUTTON_MODE:
            c = pressed(BUTTON_CLOSE_INDEX)
            o = pressed(BUTTON_OPEN_INDEX)
            if c and not self._prev_close_btn:
                self._gripper_closed = True
            if o and not self._prev_open_btn:
                self._gripper_closed = False
            self._prev_close_btn, self._prev_open_btn = c, o
        else:
            b = pressed(0)
            if b and not self._prev_toggle_btn:
                self._gripper_closed = not self._gripper_closed
            self._prev_toggle_btn = b

    # per-frame control hook
    def solve_gripper_targets(self):
        reading = self._mouse.read()
        dt = self._dt()

        if reading is not None:
            lin_raw, ang_raw, buttons = reading

            lin = np.array([_shape_axis(v, DEADZONE) for v in lin_raw]) * LIN_SIGN
            ang = np.array([_shape_axis(v, DEADZONE) for v in ang_raw]) * ANG_SIGN

            v_lin = lin * LIN_SPEED 
            w_ang = ang * ANG_SPEED

            # translate
            if LINEAR_FRAME == "body":
                world_v = _rotate_vec_xyzw(self._tcp_xyzw, v_lin)
            else:
                world_v = v_lin
            self._tcp_pos = self._tcp_pos + world_v * dt

            if WORKSPACE_BOX is not None:
                lo, hi = WORKSPACE_BOX
                self._tcp_pos = np.minimum(np.maximum(self._tcp_pos, lo), hi)

            # rotate
            angle = float(np.linalg.norm(w_ang)) * dt
            if angle > 1e-9:
                axis = w_ang / np.linalg.norm(w_ang)
                dq = _quat_from_axis_angle_xyzw(axis, angle)
                if ANGULAR_FRAME == "body":
                    # rotate about the tool's own axes: current (x) delta
                    self._tcp_xyzw = _norm4(_quat_mul_xyzw(self._tcp_xyzw, dq))
                else:
                    # rotate about world axes: delta (x) current
                    self._tcp_xyzw = _norm4(_quat_mul_xyzw(dq, self._tcp_xyzw))

            # gripper
            self._update_gripper(buttons)

        grip_q = (self.gripper_closed_values if self._gripper_closed
                  else self.gripper_open_values)

        self.set_task_target(
            tuple(float(x) for x in self._tcp_pos),
            _xyzw_to_target(self._tcp_xyzw),
            grip_q,
        )

    def __del__(self):
        try:
            self._mouse.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = SpaceMouseTeleopExample.create_parser()
    viewer, args = newton.examples.init(parser)
    viewer._pause = False
    newton.examples.run(SpaceMouseTeleopExample(viewer, args), args)