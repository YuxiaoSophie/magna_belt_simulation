"""Task-agnostic scaffolding shared by the belt tasks.

    directives.py   shared extension directives (COMMON_DIRECTIVES) and the params helper
    scene.py        make_builder, SceneInfo and build_task_scene
    joint_state.py  seeding the finalized Model with default joint state and gains
    simulation.py   BeltTaskSimulation: solver, stepping, CUDA graph, diagnostics
    cameras.py      CameraSpec and RgbdCameras: the scene's RGBD cameras on SensorTiledCamera
    point_cloud.py  PointCloudSpec and CroppedPointCloud: merged, cropped, voxelized world cloud

REPO_ROOT is the repository root every package derives its paths from. Importing this package
puts lcmtypes/ on sys.path, so the generated dairlib/drake/robotiq LCM packages import by their
top-level names.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/task_common/__init__.py

# Top-level names, not lcmtypes.drake: lcm-gen output imports itself as `drake` and matches magna.
LCMTYPES_DIR = REPO_ROOT / "lcmtypes"
if str(LCMTYPES_DIR) not in sys.path:
    sys.path.insert(0, str(LCMTYPES_DIR))
