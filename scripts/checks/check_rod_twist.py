"""CPU check: ``add_rod_ellipse`` optional ``twist_stiffness``/``twist_damping`` (V12-117).

T1: without the keys every rod joint's twist dof gets the bend values (Newton's default, so scenes
without the keys build bit-identical gains). T2: with the keys the twist dof takes them and the
bend dofs are unchanged. Run: uv run --frozen python scripts/checks/check_rod_twist.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import round_belt_task

SCENE = ROOT / "assets/round_belt_task/round_belt_scene_wr625b15.yaml"


def rod_gains(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ke, kd) of the bend and twist dofs per rod joint, shape (n_joints, 2)."""
    b = round_belt_task.make_builder()
    info = round_belt_task.build_scene(b, path)
    ke, kd = [], []
    for j in info.belt_joints:
        s = b.joint_qd_start[j]
        e = b.joint_qd_start[j + 1] if j + 1 < b.joint_count else b.joint_dof_count
        if e - s < 6:  # skip the rod's free root joint
            ke.append(b.joint_target_ke[s + 2:e])  # dofs: stretch, shear, bend, twist
            kd.append(b.joint_target_kd[s + 2:e])
    return np.array(ke), np.array(kd)


def main() -> int:
    ke, kd = rod_gains(SCENE)
    assert ke.shape[1] == 2 and np.allclose(ke, 0.225) and np.allclose(kd, 0.1), (ke[0], kd[0])
    print(f"[PASS] T1 default: {ke.shape[0]} rod joints x {ke.shape[1]} dofs, ke 0.225 kd 0.1 "
          "(twist = bend)")
    text = SCENE.read_text().replace("      bend_damping: 1.0e-1\n",
                                     "      bend_damping: 1.0e-1\n      twist_stiffness: 0.9\n"
                                     "      twist_damping: 0.3\n")
    text = text.replace("file: ../", f"file: {SCENE.parent.parent}/").replace(
        "file: round_belt", f"file: {SCENE.parent}/round_belt")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "scene.yaml"
        p.write_text(text)
        ke2, kd2 = rod_gains(p)
    changed = ~np.isclose(ke2, ke)
    assert changed.any(), ke2[0]
    assert np.allclose(ke2[changed], 0.9) and np.allclose(kd2[changed], 0.3), ke2[0]
    assert np.allclose(ke2[~changed], 0.225) and np.allclose(kd2[~changed], 0.1)
    assert (changed.sum(1) == 1).all(), "one twist dof per joint"
    print(f"[PASS] T2 twist keys: dof {np.flatnonzero(changed[0])[0]} of each joint -> ke 0.9 "
          "kd 0.3, bend dofs unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
