# Test Simulation Notes

This folder contains isolated simulation tests used to validate individual parts of the full belt project.

---

## `test.py`

### Setup

This is the successful baseline cable example adapted from Newton's `example_cable_twist.py`.

* Creates an elliptical closed-loop cable.
* Uses **64 elements**.
* Segment length is **0.1 m**, so the total cable length is about **6.4 m**.
* Cable radius is **0.02 m**, so the cable diameter is **0.04 m**.
* Full ellipse dimensions: approximately **2.56 m × 1.60 m**
* The rod uses Newton’s default mass/inertia setup.

This test is mainly used as a stable large-scale baseline for comparison with the smaller real belt.

---

## `only_belt.py`

### Setup

This is the isolated belt-only test. Goal is to match the real belt size: **248 mm × 168 mm × 6.6 mm**.
* Uses **48 elements**.
* Cable radius is **0.0033 m**, so the cable diameter is **0.0066 m**.
* Full ellipse dimensions: **0.248 m × 0.168 m**
* Uses softer belt parameters than `test.py`.
* Also tries softer user force / picking settings.
* Mass is not manually set in the script; this still needs to be tuned if necessary.

---

## `only_ur10.py`

### Setup

The setup includes everything in `only_ur10_old.py`, along with additional features:

* UR10 and Robotiq 2F-85

* SpaceMouse control

  * enables Cartesian teleoperation of the UR10 end effector using the SpaceMouse
  * reads the target position, orientation, and gripper command from the shared-memory target buffer

---

## `only_ur10_old.py`

### Setup

This is the isolated UR10 robot test.

* Loads the UR10 robot.
* Loads the Robotiq 2F-85 gripper.
* Uses the project-level `../2f85.xml` MJCF file.
* Tests robot and gripper model construction.

This test is mainly used to validate the UR10 and gripper setup before coupling the robot with the VBD belt simulation.

---

## `spacemouse.py`

### Setup

This is the SpaceMouse input translator used for robot teleoperation.

---

## `spacemouse_via_socket_sender.py`

### Setup

This is the optional SpaceMouse socket sender for running the SpaceMouse on our own computer while the Newton simulation runs on another computer.