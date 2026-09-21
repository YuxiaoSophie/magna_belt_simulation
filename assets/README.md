# Ported robot/holder assets

These assets were copied out of the bazel-managed `magna` checkout so the
Newton scene can run without that checkout present. Robot bodies/meshes were
authored for Drake; Newton's URDF importer resolves paths and mesh up-axes
differently, so each set was normalised on the way in (see "Edits applied"
below). The vendored copies are byte-identical to their sources except for the edits listed below.

## Franka (`assets/common/franka/`)

- **Source:** `magna/external/+drake_models_extension+my_drake_models/franka_description/`
  (`urdf/panda_arm.urdf`, `urdf/panda_hand_with_long_fingers.urdf`, `meshes/visual/`)
- **Upstream project:** `drake_models` (Frankaemika `franka_description`, redistributed
  via Drake's model repository)
- **License:** `assets/common/franka/LICENSE`, `assets/common/franka/LICENSE.TXT`
- **Edits applied:**
  - Rewrote every `package://drake_models/franka_description/meshes/visual/` mesh URI
    to the relative path `../meshes/visual/` (Newton has no `package://` resolution
    that matches the Drake package root; relative paths are resolved against the
    URDF's own directory).
  - Set `<robot name="panda_arm">` in `panda_arm.urdf` and `<robot name="panda_hand">`
    in `panda_hand_with_long_fingers.urdf` (Newton labels bodies/joints
    `<robot name>/<link name>`).
  - glTF up-axis fix: every `<visual>` referencing a `.gltf` mesh (`link0`..`link7` in
    the arm, `hand.gltf` in the hand — 9 visuals total) had `+pi/2` (1.5707963267948966
    rad) added to the roll component of its `<origin rpy>`. Drake applies an
    `Rx(pi/2)` rotation to every glTF mesh at render time
    (`RotationMatrixd::MakeXRotation(M_PI/2)` in
    `drake+/geometry/render_gltf_client/internal_render_engine_gltf_client.cc`)
    because glTF is Y-up and Drake's convention is Z-up; Newton's trimesh-based
    loader applies no such correction, so the rotation is baked into the URDF
    instead. `.obj` visuals/collisions (`finger_holder.obj`, `long_finger.obj`)
    were left untouched — Drake does not apply the glTF fix to OBJ meshes.
  - All other content (inertials, `drake:` attributes, transmissions, joint
    limits, non-gltf origins) is byte-identical to the source.

## UR10 (`assets/common/ur10/`)

- **Source:** `magna/external/drake-ur-driver+/models/` (`ur10.urdf`, `ur10/visual/`,
  `ur10/collision/`)
- **Upstream project:** `drake-ur-driver` (Universal Robots UR10 description, Drake
  fork of the ROS-Industrial `ur_description`)
- **License:** not vendored with the source directory; see the upstream
  `drake-ur-driver` repository for licensing terms.
- **Edits applied:**
  - `<robot name="ur10-test">` renamed to `<robot name="ur10">`.
  - glTF up-axis fix: the 7 `.gltf` visuals (`base`, `shoulder`, `upperarm`,
    `forearm`, `wrist1`, `wrist2`, `wrist3`) each had `+pi/2` added to the roll
    component of their `<origin rpy>`, for the same Drake-Y-up-glTF reason as the
    Franka assets above. The `.obj` collision origins were left untouched.
  - Mesh paths were already relative (`ur10/visual/...`, `ur10/collision/...`) and
    keep resolving correctly because the `ur10/` subfolder layout was preserved
    verbatim; no path rewrite was needed.
  - Everything else is byte-identical to the source.
- **NOT used by `round_belt_task_simulation.py` any more (appearance).** The scene now
  builds the UR10 from the textured NVIDIA USD asset
  (`newton.utils.download_asset("universal_robots_ur10")/usd/ur10_instanceable.usda`,
  the same one `round_belt.py` uses). Reason, measured: all 7 Drake
  `ur10/visual/*.gltf` files contain **zero images**. They carry only
  `baseColorFactor` values -- greys `0.223 / 0.371 / 0.464 / 0.656` plus the pale
  UR blue `(0.392, 0.543, 0.640)` linear -- so Newton imports them correctly and
  the arm still renders flat grey/white. That is the asset, not the pipeline.
  This URDF stays on disk and stays under `scripts/check_round_belt_task_poses.py`
  because it remains the **kinematic source of truth**:
  `scripts/check_round_belt_task_poses.py` check 7 walks it with an independent numpy
  FK and asserts the USD arm lands in the same world pose.
- **USD vs URDF frame conventions (measured, both at `UR10_DEFAULT_Q`):**
  - `base_link`: the USD root frame is **identical** to the URDF `base_link`
    frame (constant = identity to 1e-6), so the Drake weld `X_W_UR10` is reused
    unchanged.
  - `wrist_3_link`: the USD body frame is **not** the ROS/Drake URDF frame for the
    same link. `inv(X_usd_wrist3) . X_urdf_wrist3 = T(0, 0.0922, 0) . Rx(-90 deg)`,
    where `0.0922` is exactly the `wrist_3_joint` origin translation in
    `ur10.urdf:259`. The physical link occupies the same space; only the frame
    differs. `round_belt_task.constants.X_USDWRIST3_URDFWRIST3` encodes this, and the
    2F-85 weld goes through it so the gripper, its pads and the belt interaction
    geometry land at exactly the pose they had on the Drake URDF (verified: 2f85
    root world position error 3e-5 mm, pads still 98.600 mm apart on wrist3 +Y
    with 0.000 mm on +X, mean pad projection 122.72 mm on +Z).
  - The USD has 8 bodies (`base_link`, `shoulder_link`, `upper_arm_link`,
    `forearm_link`, `wrist_1..3_link`, `ee_link`) and the same six joint names in
    the same order, so `UR10_DEFAULT_Q` applies unchanged. The URDF-only massless
    frames (`base_link_inertia`, `base`, `ft_frame`, `flange`, `tool0`) do not
    exist in it. USD body labels are prim paths (`/ur10/<link>`), not `ur10/<link>`.

## Franka appearance: textures DO import, but `--viewer gl` tints them

Investigated because the Franka was reported as rendering wrong. It is **not** an
asset or import problem, and no URDF was edited:

- Each of the 9 Franka visual glTFs declares its baseColor twice --
  `textures[i].source` -> the `.png`, and `extensions.KHR_texture_basisu.source`
  -> the `.ktx2`. trimesh does not implement `KHR_texture_basisu`, so it decodes
  the **PNG**; there is no silent KTX2 failure.
- Verified end to end in the real scene (not a bare builder): all 9 Franka
  visuals reach `Model.shape_source` after `finalize()` with a `(2048, 2048, 3)`
  texture **and** per-vertex UVs, and `ViewerViser.log_mesh` receives all 9 and
  adds them through the textured `add_batched_meshes_trimesh` path
  (`trimesh_built=True`). The `batched_opacities` warning viser emits concerns
  opacity only; textures are unaffected. **On viser the Franka renders its real
  white-and-black livery, which is correct.**
- On `--viewer gl` the same meshes render tinted in saturated debug colours. Root
  cause, exact: `trimesh.load(..., force="mesh")` returns a `TextureVisuals`,
  which has no `main_color` attribute, and the PBR material has a
  `baseColorTexture` but no `baseColorFactor`. So in
  `external/newton/newton/_src/utils/mesh.py:1526-1550` (the "single-material mesh
  fallback") `color` stays `None`. That branch is missing the
  `if texture is not None and color is None: color = (1, 1, 1)` guard that the
  sibling `add_mesh_from_faces` helper has at `mesh.py:1404-1405`. `Mesh.color =
  None` makes `ModelBuilder` assign from its rotating **debug palette**, and
  `viewer_gl` `shaders.py:332-337` computes `albedo = ObjectColor^2.2 *
  texture^2.2`, i.e. it multiplies the real texture by that debug colour.
- This is a Newton bug under `external/newton/` and is out of scope here. It does
  not affect viser, and it no longer affects the UR10 (the USD asset sets
  `Mesh.color`, so the palette fallback never fires for it).
- **Do not "fix" this by adding `<material><color rgba>` to the URDFs.** A URDF
  material colour becomes `override_color`, and `mesh.py:1421-1431` takes the
  uniform-override branch which sets `mesh_texture = override_texture` (`None`)
  and skips per-material splitting entirely -- it would delete the Franka's
  working 2048x2048 textures.

## Belt chain holder (`assets/common/belt_chain_holder/`)

- **Source:** `magna/models/round_belt_task/belt_chain_holder/` (`belt_chain_holder.urdf`,
  `belt_chain_holder_half.obj`, `bottom_plate.obj`, `quarter_ellipse_bottom.obj`,
  `quarter_ellipse_top.obj`)
- **Upstream project:** authored in-repo for the `magna` round-belt task
- **License:** none provided upstream (project-internal asset)
- **Edits applied:** added `quarter_ellipse_rim.obj` (collisions `quarter_ellipse_rim_1` /
  `_2` on each half, placed like `quarter_ellipse_bottom_*`, z 0.005-0.010): a closed slab of
  the visual mesh's z 0.010 rim outline for one quarter, cornered at (-0.02, 0.02), so the
  rim's four 4 cm slots (|x| < 0.02, |y| < 0.02) stay open. Everything else copied verbatim —
  mesh paths are already relative and all meshes are `.obj`, so no glTF up-axis fix is needed.
  `<robot name>` was already `belt_chain_holder`.

## ZED camera (`assets/common/zed_camera/`)

- **Source:** `magna` branch `hien/timing_belt_task`, `models/common/zed_camera/`
  (`zed_camera.urdf`, `ZEDM.obj`, `ZEDM.mtl`)
- **Upstream project:** Stereolabs ZED Mini mesh, via the `magna` round-belt task
- **License:** none provided upstream
- **Edits applied:** none. Copied verbatim. The link frame is the camera's optical frame;
  the visual sits 0.305 m behind it, as in Drake. The camera poses and intrinsics live in
  `assets/common/directives/zed_cameras.yaml`.

## Trap: the two Newton mesh loaders disagree on glTF node transforms

Newton has two paths into a mesh file and they do **not** agree on the glTF
scene-graph:

- `newton.Mesh.create_from_file(...)` (via `newton._src.geometry.utils.load_mesh`)
  **ignores** the glTF node/root transforms and returns the raw mesh vertices.
- `builder.add_urdf(...)` (via `newton._src.utils.mesh.load_meshes_from_file`)
  **applies** them.

Measured on the copied files (AABB of the returned vertices, metres):

| file | `create_from_file` | `add_urdf` path | implied extra rotation |
| --- | --- | --- | --- |
| `ur10/ur10/visual/base.gltf` | min `(-0.075, -0.092, 0.0)` / max `(0.075, 0.0751, 0.038)` | min `(-0.075, 0.0, -0.0751)` / max `(0.075, 0.038, 0.092)` | `Rx(-90°)` (glTF root node) |
| `franka/meshes/visual/link0.gltf` | min `(-0.1167, -0.2307, -0.0715)` / max `(0.1577, 0.14, 0.1541)` | min `(-0.1541, -0.2307, -0.1167)` / max `(0.0715, 0.14, 0.1577)` | `Ry(-90°)` (glTF node) |

So: anyone who loads these copied glTFs with `create_from_file` (as
`utils.meshes.load_meshes` does for the *static* URDFs added by `utils.urdf` — table,
board, holder, all of which are `.obj`/`.stl` and therefore unaffected) gets a
**silently wrong orientation**, with no warning and no error. The robot URDFs in
this folder go through `add_urdf`, which is why the current scene is correct on
both counts; the `roll += pi/2` edits recorded above are the separate Drake
Y-up→Z-up visual fix and are unrelated to this loader difference.

If you ever need one of these glTFs outside `add_urdf`, apply the node transform
yourself (or load via `load_meshes_from_file`) before trusting the geometry.

## Robotiq gripper — intentionally NOT ported

The Newton scene uses the existing `2f85.xml` MJCF already in this repo for the
Robotiq 2F-85 gripper. The Drake Robotiq SDF model
(`drake_models` Robotiq description) was **intentionally not copied**, because
Newton has no SDF importer. The 2F-85 STLs referenced by `2f85.xml` predate this port;
they now live in `common/robotiq_2f85/` (its `meshdir`).

### ALOHA-style fingers, baked into `2f85.xml`

`2f85.xml` is shared (`round_belt.py`, `round_belt_two_arms.py`, `timing_belt.py` and
`ur10_2f85.yaml` all load it). Each pad body carries a black finger visual and collider in
place of the original pad geoms:

* Meshes: `common/robotiq_2f85/fingers/{left,right}_finger.obj`, from the Drake Robotiq SDF.
  They are in metres, hence `scale="1 1 1"` instead of the `2f85` class.
* Pose: the SDF finger links sit at `(+/-0.047285310862444, 0, 0.1148045193817614)` in the
  MJCF import frame; on the pad that is `pos="0 -0.002014689137556 -0.0079154806182386"`,
  quat Rz(+90 deg). Drake's `left_finger` mesh rides `right_pad`, and vice versa.
* Mass: `*_silicone_pad` keeps the inertial its removed mesh used to give it.

## LCM message types (`lcmtypes/`)

- **Source:** `magna/bazel-magna/external/dairlib+/lcmtypes/lcmt_robot_{input,output}.lcm`,
  `magna/bazel-magna/external/drake+/lcmtypes/lcmt_{schunk_wsg_status,schunk_wsg_command,
  viewer_link_data,viewer_geometry_data}.lcm`, `magna/bazel-magna/external/robotiq-driver+/lcmtypes/
  lcmt_robotiq_{command,status}.lcm`
- **Copies:** byte-identical.
- **Regenerated with:** `scripts/gen_lcmtypes.sh` (emits the `dairlib/`, `drake/`, `robotiq/`
  Python packages at the repo root via the venv's `lcm-gen`).

## LCM simulation parameters (`round_belt_task/round_belt_lcm_sim.yaml`)

- **`belt_trigger`:** transcribed from `magna/systems/simulation/magna_simulation.cc` (target
  point) and `round_belt_controller_params_sim.yaml` `predefined_motion_position_tolerance`
  (0.005 m); `grasp_depth` and `nearest_body_radius` are Newton-side anchor choices.

## Table and task board — not copied here

The table and task board models are not part of this port. The Newton scene
uses the copies already present in `task_board_urdf/` (`common/scene.urdf`,
`round_belt_task/round_belt_task_board.urdf`).

`round_belt_task/round_belt_task_board.urdf` edit: the two pulley joints are `continuous` axles
at the pulley centres (link frames and geometry re-expressed there) with a thin
`rotation_marker_strip` visual each, as magna commit `2d9b0ca` does to the SDF; unlike that
commit the axles carry `damping="0.001"` (N m s/rad) so a kicked pulley does not spin forever.
