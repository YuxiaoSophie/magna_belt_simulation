# Round Belt Simulation Notes

This project develops a Newton/Warp round belt simulation.

## Expected folder structure

```text
my_projects/
├── assets/
│   └── project assets and additional simulation resources
│
├── external/
│   └── newton
│
├── task_board_urdf/
│   ├── common/
│   │   ├── table/
│   │   │   ├── table.obj
│   │   │   └── table.mtl
│   │   └── task_board_just_board.glb
│   │
│   ├── round_belt_task/
│   │   └── round_belt_task_board/
│   │       ├── small_round_pulley/
│   │       └── large_round_pulley/
│   │
│   └── timing_belt_task/
│       └── ...
│
├── test/
│   ├── README.md
│   ├── test.py
│   ├── only_belt.py
│   └── only_ur10.py
│
├── 2f85.xml
│   └── MuJoCo MJCF model for the Robotiq 2F-85 gripper
│
├── README.md
├── round_belt.py
├── round_belt_command.py
└── round_belt.urdf.xacro
```

## Run

### Terminal 1: Start the VirtualGL client

```bash
/opt/VirtualGL/bin/vglclient
```

### Terminal 2: Run the full simulation

```bash
vglrun -d :1 uv run python round_belt_command.py
```

---

## Files

### 1. Main files

#### `round_belt.py`

##### Current setup

This is the main full-scene simulation.

The setup includes, in this order:

* Round belt and task environment
  * table
  * task board
  * small pulley
  * large pulley
  * deformable closed-loop round belt
  * real belt dimensions: 248 mm × 168 mm × 6.6 mm
  * 48 rod elements
  * cable radius: 0.0033 m
  * target belt mass: 22 grams

* UR10 and Robotiq 2F-85
  * UR10 robot arm
  * Robotiq 2F-85 gripper
  * MuJoCo-controlled articulated robot model

##### Previous issues

* Some parts appeared on the floor because of coordinate / height mismatch.

* The belt dimension needed to match the real 248 mm × 168 mm × 6.6 mm size.

* High stiffness caused instability: when user force pulled the belt, the rod tried to maintain stiff constraints and could explode.

* `add_rod_graph` builds the cable from an explicit graph topology: nodes, edges, and connection data must be provided manually.

* `add_rod(..., closed=True)` builds the rod from an ordered list of points along one continuous path. For this belt, the geometry is just one closed ellipse, so `closed=True` automatically connects the last segment back to the first and is simpler and less error-prone.

* Use the latest Newton source to reduce cable explosion

  This project now uses:

  ```bash
  git clone https://github.com/newton-physics/newton
  ```

  The cloned Newton source is used because it may include newer solver fixes that are not yet available in the released `pip` package.

  The important improvement is in Newton’s VBD rigid contact behavior for finite-radius objects such as cables. Previously, when a small-radius cable contacted another object while rotating, the normal contact response could act at a rotating surface anchor point. This could inject artificial kinetic energy into the simulation, making the cable suddenly jump, spin, or explode. The newer Newton source improves this contact handling by applying the normal contact response more stably for cable-like objects, reducing non-physical energy gain during contact.

* Use center-of-mass body frames for rod segments

  The rod setup was also changed to use:

  ```python
  body_frame_origin="com"
  ```

  This places each rod segment’s body frame at its center of mass instead of at the segment start point.

##### Updates

* Currently, the interaction is two-way: the cable feels the robot through the VBD proxy bodies that carry the gripper’s motion and inertia, while the robot feels the cable because the cable’s contact reaction forces are fed back to the corresponding MuJoCo gripper bodies.
* Currently, I am using standard Newton contact (since the belt remains stable and does not slip away in the simulation, so I have not switched to hydroelastic contact).

---

#### `round_belt_command.py`

##### Current setup

This is the commanded-motion version of the full simulation.

---

### 2. Test files

The test scripts are stored in:

```text
test/
```

See:

```text
test/README.md
```

for a short description of each test setup.