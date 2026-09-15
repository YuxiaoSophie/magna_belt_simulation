# Scene directives

A reference for authoring or reading a scene YAML without reading the loader source. The
loader itself is `utils/directives/` (`schema.py` — parsing, no builder; `runtime.py` —
execution onto a `newton.ModelBuilder`); the round-belt extensions are
`round_belt_task/directives.py`.
Where this doc and the code disagree, the code wins — file an issue against this doc, not
the code.

> **Float literals.** PyYAML is YAML 1.1, so `8.95207485e01` and `2e4` parse as strings; the
> loader coerces them with `float()` and fails only on a non-number (`abc`). Still write the
> signed form (`8.95207485e+01`, `2.0e+4`): magna's `.dmd.yaml` files do, so the two repos diff.

## 1. What this is

A scene YAML is a Drake model-directives file executed onto a `newton.ModelBuilder` by
`utils.directives.load_directives`. The core directives — `add_model`, `add_weld`,
`add_frame`, `add_directives`, the `X_PC`/`X_PF` pose blocks and the `!Rpy { deg: [...] }`
tag — are Drake's syntax **verbatim**, so the numbers diff line-by-line against the Drake
`.dmd.yaml` they were ported from. Everything Drake cannot say is an explicit
Newton-native extension: `static: true`, `kind:`, colours, `importer_options`,
`newton_asset://` paths and custom directive kinds (`add_rod_ellipse` and friends).

Because of those extensions there is no `package://` map and no SDF support, so **these
files are not Drake-loadable, and Drake's directives are not loadable by this repo's
loader either** — hence the extension `*.yaml`, never `*.dmd.yaml`. What genuinely is
shared with Drake: the four core directive names, the pose syntax, and — where a model was
ported verbatim — the same numbers.

## 2. File layout

```text
assets/
├── <task>/
│   └── <task>_scene.yaml        # e.g. round_belt_task/round_belt_scene.yaml
└── common/
    └── directives/
        └── <shared>.yaml         # e.g. ur10_2f85.yaml — pulled in via add_directives
```

- A task's own scene lives at `assets/<task>/<task>_scene.yaml` (the current example:
  `assets/round_belt_task/round_belt_scene.yaml`).
- Anything shared across tasks (the UR10 + Robotiq 2F-85 rig, the two ZED cameras) lives under
  `assets/common/directives/` and is pulled in with `add_directives`, mirroring how Drake's
  task scenes include `ur.dmd.yaml`.
- `file:` on any directive resolves one of three ways (`utils/directives/schema.py::_resolve`):
  1. `newton_asset://<asset>/<relative/path>` — downloaded/cached via
     `newton.utils.download_asset` (used for the UR10 USD:
     `newton_asset://universal_robots_ur10/usd/ur10_instanceable.usda`).
  2. an absolute path.
  3. otherwise, relative to the directory of the YAML file that *authored* the entry — not
     the top-level scene file, not the process's cwd. E.g. `round_belt_scene.yaml`'s
     `file: ../common/scene.urdf` resolves from `assets/round_belt_task/`, i.e.
     `assets/common/scene.urdf`.
- **Timing matters.** `add_model` file paths resolve at *load* time (inside
  `_load_model`, via `DirectiveContext.resolve_path`), so `parse_directives` never
  downloads a model's asset — it reads the YAML files (including everything reachable
  through `add_directives`), and touches the asset cache only if an include is itself a
  `newton_asset://` path. That is what lets `round_belt_task/constants.py` call
  `parse_directives` at import time with no side effects (its includes are relative).
  `add_directives` file paths, by contrast, resolve **and get read** at *parse* time,
  because the included entries must be flattened into the same list before anything else
  can run.

## 3. Pose syntax

Every core-directive pose block (`X_PC`, `X_PF`) is:

```yaml
translation: [x, y, z]                       # metres
rotation: !Rpy { deg: [roll, pitch, yaw] }    # degrees
```

`R = Rz(yaw) · Ry(pitch) · Rx(roll)` (extrinsic X-Y-Z) — Drake's and URDF's `<origin rpy>`
convention. The only place this is turned into a quaternion is
`utils.transforms.drake_xform`, whose module docstring says as much
(`utils.directives`'s own docstring closes with "Poses go through
`utils.transforms.drake_xform` and nothing else"); that module asserts the matrix identity
above against the round-belt board's own yaw at *import* time and raises `AssertionError`
if anyone changes the convention.

Details:
- Omitting `rotation` → identity. Omitting `translation` → `[0, 0, 0]`. Omitting the whole
  block (e.g. `add_weld: {parent: world, child: table::scene}` with no `X_PC:`) → identity
  `Pose()`.
- `!Rpy { rad: [...] }` is rejected outright (`ValueError`) — only `deg:` is accepted.
- The float-literal note above applies to every number in these blocks
  (and every other numeric leaf the loader reads).

## 4. Core directives

### `add_model`

Required: `name`, `file`. `kind` is `static | urdf | mjcf | usd`, either explicit or
inferred from the file suffix (`.urdf` → `urdf`, `.xml` → `mjcf`, `.usd`/`.usda`/`.usdc` →
`usd`). `static: true` forces `kind: static` (and conflicts with any other explicit
`kind:`); a static model's file must be a `.urdf`.

Allowed keys by kind (`utils/directives/schema.py::_MODEL_KEYS`, beyond the shared
`name`/`file`/`kind`/`static`):

| kind     | extra keys |
|----------|------------|
| `static` | `color`, `link_colors`, `split_components`, `component_colors` |
| `urdf`   | `urdf_fixups`, `default_joint_positions`, `gravity_compensation`, `importer_options` |
| `mjcf`   | `default_joint_positions`, `gravity_compensation`, `importer_options` |
| `usd`    | `default_joint_positions`, `gravity_compensation`, `importer_options` |

Any other key raises `ValueError` naming the unknown key(s) and what's allowed.

- **`static: true`.** Imports a fixed-only URDF as world shapes (body `-1`) via
  `utils.urdf.add_urdf_as_static_shapes`, *not* `builder.add_urdf`. **Why:** the VBD half of
  the coupled solver must own the table/board/holder, so they have to be plain world shapes
  rather than a (zero-mass) articulation sitting inside the MuJoCo entry — exactly what
  `round_belt.py`'s `add_table`/`add_board` do by hand.
- **`importer_options`** is a verbatim kwargs pass-through to `builder.add_urdf` /
  `add_mjcf` / `add_usd` (e.g. `enable_self_collisions`, `collapse_fixed_joints`,
  `hide_collision_shapes` — see the UR10 and Franka entries in the example files).
- **`urdf_fixups`** (kind `urdf` only, default `true`): after `add_urdf`, three post-import
  passes run over the model's new shapes — `utils.labels.label_shapes_by_body`,
  `utils.meshes.fix_inverted_mesh_winding`, `utils.meshes.neutralize_textured_shape_colors`.
  These patch quirks of Newton's URDF import, not scene data; see §6.
- **`gravity_compensation`**: sets the `mujoco:gravcomp` custom attribute to `1.0` on every
  body the model added. Done on the builder (not later on the finalized `Model`) because
  `mujoco:gravcomp` is a builder-only custom attribute.
- **`link_colors`** (static only): `{link_name: [r, g, b]}`; recolours every visual shape
  whose label starts with `<model>/<link>/visual`.
- **`component_colors`** (static only, requires `split_components: true`): a list of
  `{color, max_span, near_local_xy, radius}` rules, first match wins. For each split mesh
  component: skip it if its XY bbox span is `>= max_span`; otherwise colour it `color` if
  its XY-bbox centre is within `radius` of `near_local_xy`. Components matching no rule
  take the normal order (`utils/urdf.py::_mesh_color`): white if textured, else the
  asset-authored colour, else the model's `color`. This is how the round-belt board's
  pulley mounting plate gets painted black without its own mesh file — **the pulley
  colours themselves are this built-in mechanism, not a separate directive.**

### `add_weld`

Keys: `parent`, `child`, `X_PC` (optional, default identity).

- **Look-ahead.** Newton needs a body's parent at import time, so for every `add_model`,
  the loader scans the *whole* flattened directive list for the `add_weld` whose `child` is
  `<name>` or `<name>::<link>`, and imports that model already parented/posed; the
  `add_weld` entry itself is a no-op at its own position in the file. From the loader
  docstring: *"At most one weld per model, every weld must be consumed by exactly one
  model, and the weld child must be the model's root link."*
- **Root-link check.** After import, the loader verifies the weld child is the model's
  articulation root by checking it against the base joint the importer just created (the
  one joint in the model's range whose parent is the weld parent) — not a pose comparison,
  because `add_urdf` (unlike MJCF/USD) leaves every `builder.body_q` at identity, i.e. it
  does no build-time forward kinematics. Naming any other link raises `ValueError`.
- **Bare `<model>` child (MJCF).** `add_weld: {parent: ..., child: robotiq_2f85, ...}` in
  `ur10_2f85.yaml` means "the model's own import frame", not a body — needed because
  `2f85.xml`'s root body (`base_mount`) sits `pos="0 0 0.007"` off the import frame. Naming
  that body instead (`robotiq_2f85::base_mount`) fails the post-import pose check with the
  7 mm miss and a pointer back to the bare name.
- **No weld = floating.** A model with no `add_weld` naming it imports with
  `floating=True` (urdf/mjcf/usd) or, for `static`, is simply not weldable to anything but
  `world`.
- Orphan welds (consumed by nothing) raise `ValueError` at the end of `load_directives`,
  naming the unconsumed child(ren).

### `add_frame`

Keys: `name`, `X_PF: {base_frame, translation, rotation}`.

`base_frame` (and any other frame reference in the file — a weld's `parent`, a frame's own
`base_frame`) is one of: `world`, `<model>::<link>`, or a name already declared by an
earlier `add_frame`. `<model>::<link>` matches whichever of that model's own bodies has a
label ending in `/<link>` — the match must be unique within the model's body range. Label
formats differ per importer:

| importer | example label |
|---|---|
| URDF | `panda_arm/panda_link8` |
| USD  | `/ur10/wrist_3_link` |
| MJCF | `robotiq_2f85/worldbody/base_mount/base/left_spring_link/left_follower/left_pad` |

Worked example (`assets/common/directives/ur10_2f85.yaml`) — the frame that absorbs the
NVIDIA USD UR10's `wrist_3_link` frame difference from the Drake/URDF one:

```yaml
- add_frame:
    name: ur10_wrist_3_link_drake
    X_PF:
      base_frame: ur10::wrist_3_link
      translation: [0.0, 0.0922, 0.0]
      rotation: !Rpy { deg: [-90.0, 0.0, 0.0] }
```

### `add_directives`

One key: `file`. Recursively parses that file and splices its directives into the current
list at this position (cycle-checked). This is how `round_belt_scene.yaml` pulls in the
shared UR10 + Robotiq rig:

```yaml
- add_directives: {file: ../common/directives/ur10_2f85.yaml}
```

## 5. Extension directives

Registered as `EXTENSION_DIRECTIVES` in `round_belt_task/directives.py`, passed to
`load_directives(..., directives=EXTENSION_DIRECTIVES)`. Each validates its own keys
strictly (the shared `task_common.directives.params` helper: unknown key → `ValueError`, missing required key →
`ValueError`) and, by convention (not enforced by the loader), stores its output in
`ctx.scene.extras[params["name"]]`.

The Robotiq's ALOHA fingers are not a directive: they are geoms in `2f85.xml`'s pad bodies
(see `assets/README.md`).

| directive | required params | optional params (default) | `extras[name]` | ordering |
|---|---|---|---|---|
| `add_tabletop_collision` | `table_visual`, `top_z`, `thickness`, `color` | `name` (`"tabletop_collision"`) | `{"shape": int, "aabb": (lo, hi)}` | must sit inside the static shape range, between the model owning `table_visual` and whatever comes after it (order is load-bearing for `task_common/scene.py`'s contiguous static-shape check) |
| `add_ground_plane` | exactly one of `height` or `height_from_aabb_min_z_of` | `name` (`"ground"`) | `{"shape": int, "height": float}` | by convention last — not enforced by the loader, just kept out of any model's contiguous shape range |
| `add_rgbd_camera` | `name`, `base_frame` (`world`, an `add_frame` name, or a static model's weld child) | `width`/`height` (`640`/`480`), `fps` (`20`), `fov_y_deg` (`45`) or `focal_x`+`focal_y` [px], `center_x`/`center_y` (image centre, `(w - 1) / 2`), `z_near`/`z_far` (`0.1`/`5.0`) | `task_common.cameras.CameraSpec` | anywhere after its `base_frame`; adds nothing to the builder. `base_frame` is the OpenCV optical frame (+Z forward, +Y down) and must be world-fixed; defaults are Drake's `CameraConfig` |
| `add_cropped_point_cloud` | `name`, `cameras` (list of earlier `add_rgbd_camera` names), `crop_lower_xyz`, `crop_upper_xyz` [m, world] | `voxel_size` [m] (`0.0`, no downsample) | `task_common.point_cloud.PointCloudSpec` | after the cameras it names; adds nothing to the builder. Evaluated by `task_common.point_cloud.CroppedPointCloud`: Drake's `DepthImageToPointCloud` + `Concatenate` + `Crop` + `VoxelizedDownSample` |
| `add_rod_ellipse` | `name`, `center`, `semi_axes`, `radius`, `num_elements`, `color`, `stretch_stiffness`, `stretch_damping`, `bend_stiffness`, `bend_damping` | `twist_total` (`0.0`), `closed` (`true`), `body_frame_origin` (`"com"`), `margin` (`0.0`), `gap` (`0.001`), `density`/`ke`/`kd`/`mu` (fall back to `round_belt`'s belt-density estimate and cable-contact constants) | `{"bodies": [...], "joints": [...], "shapes": [...]}` | by convention after all robot models — required in practice because `task_common/scene.py`'s `span` helper needs each robot model's body/joint/shape ranges to be mutually contiguous, which a rod inserted in between would break; not checked by the loader itself |

## 6. What is NOT data and why

- **The belt is procedural.** `add_rod_ellipse` takes an ellipse's parameters (`center`,
  `semi_axes`, `radius`, `num_elements`, stiffnesses, ...), but the point sampling around
  the ellipse and the rod construction itself (`builder.add_rod` with parallel-transported
  edge quaternions) are plain Python in `round_belt_task/directives.py`; no vertex data lives in
  the YAML.
- **The tabletop collider is sized from the table's AABB.** `add_tabletop_collision` only
  takes `top_z`/`thickness`/`color`; its XY footprint comes from the table visual shape's
  collected world-space AABB (`ctx.scene.aabbs`), not from authored dimensions.
- **The pulley-mount colour rule is a per-component geometric predicate.**
  `component_colors` rules are evaluated per split mesh component (its bbox span and centre)
  at import time; the YAML supplies thresholds, not which triangles are which colour.
- **`urdf_fixups` is a boolean switch, not data.** What it actually does — relabel shapes by
  body, flip inward-wound mesh normals, neutralize textured shape colours — is Python
  patching applied to whatever `add_urdf` produced; none of that is expressible as directive
  parameters.

## 7. Adding a new task

Sketch for the bike-chain task (`magna/models/bike_chain_task/bike-chain-scene.dmd.yaml`,
read-only reference outside this repo):

1. Copy `assets/round_belt_task/round_belt_scene.yaml` to something like
   `assets/bike_chain_task/bike_chain_scene.yaml`.
2. Swap the `board` model and its weld for the bike-chain board and its Drake pose:
   ```yaml
   - add_weld:
       parent: world
       child: board::board
       X_PC:
         translation: [0.67416894, -0.19778154, 0.00619352]
         rotation: !Rpy { deg: [-0.960679, -0.4460114, 89.5166137] }
   ```
   (from `bike-chain-scene.dmd.yaml`'s `nist_board` weld — same structure, different
   numbers than the round-belt board's).
3. Keep the `belt_chain_holder` model/weld as-is: the Drake bike-chain scene welds it to
   the *same* pose as the round-belt scene (`translation: [0.4736603358808432,
   0.3520562100563749, -0.02858]`, `rotation: !Rpy { deg: [0, 0, 90] }`), so nothing here
   changes.
4. Keep the Franka block and `add_directives: {file: ../common/directives/ur10_2f85.yaml}`
   unchanged — same arm, same shared gripper rig. Only the task's *own*
   `ur10::base_link` weld numbers need updating (the bike-chain Drake scene welds the UR10
   to a slightly different pose than the round-belt one:
   `translation: [1.32978889, -0.17550038, 0.04153631]`,
   `rotation: !Rpy { deg: [-1.19465397, 1.55078702, 178.47057524] }`).
5. Replace `add_rod_ellipse` with a new custom directive (e.g. `add_chain`) that builds
   whatever a chain needs — it will not be a closed elliptical rod, so it does not belong in
   `add_rod_ellipse`'s parameter set. Register it in a `directives={...}` mapping the way
   `EXTENSION_DIRECTIVES` does today, following §8.
6. Add a `bike_chain_task/` package mirroring `round_belt_task/`: a `constants.py` that
   calls `parse_directives` on the new scene file, and a `scene.py` whose `build_scene`
   calls `load_directives(builder, directives_path, directives=...)` and maps the resulting
   `LoadedScene` into a task-specific `SceneInfo`-shaped dataclass (in this repo,
   `round_belt_task/scene.py`'s `build_scene` is the pattern to follow, though it is closer
   to 60 lines than 30 once the static/robot bookkeeping is included).

**Honest caveat:** none of this can run today. The bike-chain board asset in `magna` is
still `bike_chain_task_board.sdf`/`.obj` (Newton has no SDF importer, and it is not
vendored into `assets/` here); porting it to a URDF the way
`round_belt_task_board.urdf` was ported from `round_belt_task_board.sdf` is a separate job.
The timing-belt task (`magna/models/timing_belt_task/`) is in the same state — its board is
also still an `.sdf`, and even the Drake scene (`bike-chain-scene.dmd.yaml`, in a commented-out
`add_model` block) never actually loaded a timing belt model. Nothing under
`assets/bike_chain_task/` or `assets/timing_belt_task/` exists in this repo yet.

## 8. Adding a custom directive

A custom directive is any `DirectiveFn`:

```python
DirectiveFn = Callable[[DirectiveContext, Mapping[str, Any]], None]
```

registered by name in the `directives=` mapping passed to `load_directives` (any name not
in `CORE_DIRECTIVES = ("add_model", "add_weld", "add_frame", "add_directives")`; reusing a
core name raises `ValueError` at `load_directives` call time). It appears in the YAML in
the position it should run, exactly like a core directive — the file is the complete
top-to-bottom recipe.

`DirectiveContext` (what the function is handed):

| field | what it is |
|---|---|
| `builder` | the `newton.ModelBuilder` being assembled |
| `scene` | the `LoadedScene` so far — `scene.models`, `scene.frames`, `scene.aabbs`, `scene.extras`, plus `scene.body(ref)` / `scene.frame_transform(ref)` for resolving `world` / `<model>::<link>` / frame-name references |
| `source` | the directory of the YAML that authored this entry |
| `visual_cfg` / `collision_cfg` | the shared `ModelBuilder.ShapeConfig`s used everywhere else in the scene |
| `resolve_path(file)` | resolves a `file:`-style string exactly like `add_model` does |

Rules, from the `round_belt_task/directives.py` module docstring and the three existing
directives:
- **Validate your own keys strictly.** The loader does not check a custom directive's
  `params` at all — unknown keys, missing required keys, and type coercion are entirely
  the directive's job. `task_common/directives.py::params` is the pattern the three
  existing directives use (merge over a spec dict where `REQUIRED` marks no default,
  raise on anything unknown or still-`REQUIRED`).
- **Coerce numbers with the loader's own helpers** (`utils.directives.as_float` /
  `as_floats` / `as_vec3`, exported for this purpose), so a bad literal fails the same way a
  core directive's would.
- **By convention, store your output** in `ctx.scene.extras[params["name"]]` so later
  directives and the task's `scene.py` can find it — not enforced by the loader, but every
  existing directive does it.
- **Data only.** A directive's `params` come straight from parsed YAML (strings, numbers,
  lists, nested mappings) — no callables, no Python objects, ever cross that boundary.

## 9. Divergences from Drake

| Drake | here | why |
|---|---|---|
| `nist_board` | `board` | renamed on port (see the `# Drake:` comments in `round_belt_scene.yaml`) |
| `panda` | `panda_arm` | renamed on port |
| `robotiq_85` | `robotiq_2f85` | renamed on port |
| `belt_holder` | `belt_chain_holder` | renamed on port |
| `package://...` URIs | relative paths / `newton_asset://` | no `package://` map here |
| Robotiq SDF (`robotiq_arg85_parallel_grippers.sdf`) | `2f85.xml` MJCF | Newton has no SDF importer |
| board via SDF | board via URDF | same reason; the board was ported SDF → URDF |
| UR10 glTF-textureless URDF | NVIDIA `universal_robots_ur10` USD | the Drake glTFs carry no images; see `assets/README.md` |
| (no equivalent) | `ur10_wrist_3_link_drake` frame | absorbs the USD-vs-URDF `wrist_3_link` frame difference so every weld/frame downstream of it stays at the Drake numbers |
| Franka starts at `franka.dmd.yaml`'s "ready" pose | starts at `q_init_franka`/`q_init_franka_hand` | this scene never loads `franka.dmd.yaml`; it reproduces what the Drake *simulation* actually seeds (`round_belt_simulation_params.yaml`), not the directives file's own default |
| n/a | `static`, `kind`, `color`/`link_colors`/`component_colors`/`split_components`, `importer_options`, `urdf_fixups`, `gravity_compensation` on `add_model`; `add_tabletop_collision`, `add_rod_ellipse`, `add_ground_plane` as directives | Newton-native extensions with no Drake equivalent (§1, §4, §5) |
