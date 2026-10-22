"""GPU check: opt-in ``add_rod_ellipse`` ``pin_stiffness`` (V12-122).

T1: an unchanged scene (S, wr625b15) builds the same model and VBD solver arrays, bit for bit, as
with the pin step disabled. T2: with ``pin_stiffness: true`` the model is unchanged and only the
rod stretch slots of ``joint_penalty_k``/``joint_penalty_k_min`` change, to the authored k (shear
stays on the ramp). T3: bend/twist penalties already sit at the authored k (angular k_start
10 > 0.225), pin or not.
T4: a hanging 48-link rod with 5 N of hanging mass, 10 VBD iterations, pinned through the sim's
own method, stretches < 2x the nominal F/k series. Each scene builds in a child process (the
scene is chosen by ROUND_BELT_SCENE at import). Run under the sim lock:
uv run --frozen python scripts/checks/check_belt_pin.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCENE = ROOT / "assets/round_belt_task/round_belt_scene_wr625b15.yaml"
PIN_KEYS = {"vbd.joint_penalty_k", "vbd.joint_penalty_k_min"}


def dump(out: str, disable_pin: bool) -> None:
    """Child: build the sim of $ROUND_BELT_SCENE; save model + VBD wp.arrays and rod slots."""
    import newton
    import newton.examples
    import warp as wp

    from round_belt_task.simulation import RoundBeltTaskSimulation

    if disable_pin:
        RoundBeltTaskSimulation._pin_belt_stretch_stiffness = lambda self, info: None
    parser = RoundBeltTaskSimulation.create_parser()
    parser.set_defaults(viewer="null", cameras=False)
    sys.argv = sys.argv[:1]  # init() parses argv
    viewer, args = newton.examples.init(parser)
    sim = RoundBeltTaskSimulation(viewer, args)
    vbd = sim.solver.solver("vbd")
    arrays = {}
    for prefix, obj in (("model", sim.model), ("vbd", vbd)):
        for name, value in vars(obj).items():
            if isinstance(value, wp.array):
                arrays[f"{prefix}.{name}"] = value.numpy()
    jt = vbd.model.joint_type.numpy()
    arrays["rod_slots"] = vbd.joint_constraint_start.numpy()[jt == int(newton.JointType.ROD)]
    np.savez(out, **arrays)


def build(scene: Path, out: Path, disable_pin: bool = False) -> dict:
    env = {**os.environ, "ROUND_BELT_SCENE": str(scene)}
    cmd = [sys.executable, __file__, "--dump", str(out)] + (["--no-pin"] if disable_pin else [])
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with np.load(out) as z:
        return dict(z)


def diff_keys(a: dict, b: dict) -> list[str]:
    assert a.keys() == b.keys(), a.keys() ^ b.keys()
    return sorted(k for k in a if a[k].shape != b[k].shape or a[k].tobytes() != b[k].tobytes())


def pinned_pull(
        force: float = 5.0, iterations: int = 10, seconds: float = 4.0) -> tuple[float, float]:
    """Stretch [m] and nominal [m] of S's rod hanging from a fixed top body with a mass F/g."""
    import newton
    import warp as wp
    import yaml
    from newton.solvers import SolverVBD

    from round_belt_task.directives import estimate_belt_density
    from task_common import defaults
    from task_common.scene import make_builder
    from task_common.simulation import BeltTaskSimulation

    rod = next(d["add_rod_ellipse"] for d in yaml.safe_load(SCENE.read_text().replace("!Rpy", ""))[
        "directives"] if "add_rod_ellipse" in d)
    a, b = rod["semi_axes"]
    n = rod["num_elements"]
    t = 2 * np.pi * np.arange(n + 1) / n
    ring = np.c_[a * np.cos(t), b * np.sin(t)]
    seg = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum()) / n
    builder = make_builder()
    pts = [wp.vec3(0.0, 0.0, 1.0 - i * seg) for i in range(n + 1)]
    bodies, joints = builder.add_rod(
        positions=pts, quaternions=newton.utils.create_parallel_transport_cable_quaternions(pts),
        radius=rod["radius"], cfg=newton.ModelBuilder.ShapeConfig(density=estimate_belt_density()),
        stretch_stiffness=rod["stretch_stiffness"], stretch_damping=rod["stretch_damping"],
        bend_stiffness=rod["bend_stiffness"], bend_damping=rod["bend_damping"],
        closed=False, body_frame_origin="com", label="belt")
    top, bottom = bodies[0], bodies[-1]
    builder.body_mass[top] = builder.body_inv_mass[top] = 0.0
    builder.body_inertia[top] = builder.body_inv_inertia[top] = wp.mat33(0.0)
    # hanging mass: body_f would enter VBD's initial guess
    scale = 1.0 + force / 9.81 / builder.body_mass[bottom]
    builder.body_mass[bottom] *= scale
    builder.body_inv_mass[bottom] /= scale
    builder.body_inertia[bottom] = builder.body_inertia[bottom] * scale
    builder.body_inv_inertia[bottom] = builder.body_inv_inertia[bottom] / scale
    builder.color()
    model = builder.finalize()
    solver = SolverVBD(
        model=model, iterations=iterations, rigid_avbd_beta=defaults.VBD_RIGID_AVBD_BETA,
        rigid_contact_k_start=defaults.VBD_RIGID_CONTACT_K_START, rigid_contact_history=True,
        rigid_body_contact_buffer_size=defaults.VBD_RIGID_CONTACT_BUFFER_SIZE,
        rigid_joint_linear_ke=defaults.VBD_RIGID_JOINT_LINEAR_KE,
        rigid_joint_angular_ke=defaults.VBD_RIGID_JOINT_ANGULAR_KE,
        rigid_joint_linear_kd=defaults.VBD_RIGID_JOINT_LINEAR_KD,
        rigid_joint_angular_kd=defaults.VBD_RIGID_JOINT_ANGULAR_KD)
    fake = SimpleNamespace(solver=SimpleNamespace(solver=lambda name: solver), model=model)
    BeltTaskSimulation._pin_belt_stretch_stiffness(
        fake, SimpleNamespace(belt_pin_stiffness=True, belt_joints=list(joints)))
    is_rod = model.joint_type.numpy() == int(newton.JointType.ROD)
    rod_slots = solver.joint_constraint_start.numpy()[is_rod]
    k = solver.joint_penalty_k.numpy()[rod_slots]
    assert np.allclose(k, rod["stretch_stiffness"]), k[:3]
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    span = lambda s: float(np.linalg.norm(np.diff(s.body_q.numpy()[[top, bottom], :3], axis=0)))
    span0, dt, spans = span(s0), 0.0025, []
    for i in range(int(seconds / dt)):
        solver.step(s0, s1, ctl, None, dt)
        s0, s1 = s1, s0
        if i * dt >= seconds - 1.0 and i % 20 == 0:
            spans.append(span(s0))
    weight = float(model.body_mass.numpy().sum()) * 9.81 - force
    nominal = sum(force + weight * (n - 1 - i) / n for i in range(n - 1)) / rod["stretch_stiffness"]
    return float(np.mean(spans)) - span0, nominal


def main() -> int:
    text = SCENE.read_text()
    text = text.replace("file: ../", f"file: {SCENE.parent.parent}/").replace(
        "file: round_belt", f"file: {SCENE.parent}/round_belt")
    with tempfile.TemporaryDirectory() as d:
        pin = Path(d) / "scene_pin.yaml"
        pin.write_text(text.replace("      bend_damping: 1.0e-1\n",
                                    "      bend_damping: 1.0e-1\n      pin_stiffness: true\n"))
        a = build(SCENE, Path(d) / "a.npz")
        # pointers, uninitialised buffers
        noise = set(diff_keys(a, build(SCENE, Path(d) / "a2.npz")))
        base = build(SCENE, Path(d) / "base.npz", disable_pin=True)
        c = build(pin, Path(d) / "c.npz")
    assert len(noise) < 5, noise
    assert not set(diff_keys(a, base)) - noise, diff_keys(a, base)
    print(f"[PASS] T1 unchanged scene: {len(a) - 1 - len(noise)} model + VBD arrays bit-identical "
          f"to the pin step disabled ({len(noise)} differ between two identical builds: "
          f"{sorted(noise)})")
    changed = set(diff_keys(a, c)) - noise
    assert set(changed) == PIN_KEYS, changed
    slots = a["rod_slots"]
    for key in PIN_KEYS:
        moved = np.flatnonzero(a[key] != c[key])
        assert set(moved) <= set(slots), sorted(set(moved) - set(slots))
        assert np.array_equal(c[key][slots], c["vbd.joint_material_k"][slots]), key
    print(f"[PASS] T2 pin_stiffness: model identical; only {sorted(PIN_KEYS)} change, at the "
          f"{len(slots)} rod stretch slots, to k = "
          f"{float(np.median(c['vbd.joint_material_k'][slots])):g} "
          f"(was {float(np.median(a['vbd.joint_penalty_k'][slots])):g})")
    bend = np.concatenate([a["rod_slots"] + 2, a["rod_slots"] + 3])
    for x in (a, c):
        assert np.array_equal(x["vbd.joint_penalty_k"][bend], x["vbd.joint_material_k"][bend])
    k_bend = float(np.median(a["vbd.joint_material_k"][bend]))
    print(f"[PASS] T3 bend/twist penalty = authored {k_bend:g} from step 0, pin or not")
    stretch, nominal = pinned_pull()
    assert stretch < 2.0 * nominal, (stretch, nominal)
    print(f"[PASS] T4 pinned static pull 5 N, 10 iters: {stretch * 1e3:.1f} mm vs nominal "
          f"{nominal * 1e3:.1f} mm ({stretch / nominal:.2f}x)")
    return 0


if __name__ == "__main__":
    if "--dump" in sys.argv:
        dump(sys.argv[sys.argv.index("--dump") + 1], "--no-pin" in sys.argv)
        sys.exit(0)
    sys.exit(main())
