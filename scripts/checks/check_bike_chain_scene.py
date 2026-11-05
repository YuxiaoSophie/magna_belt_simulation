"""CPU check of the bike-chain scene (V13-04): directives, counts, numbers, round-belt identity.

Run:
    uv run --frozen python scripts/checks/check_bike_chain_scene.py [--chain-links N]
(--chain-links overrides the expected chain body count; the negative flip uses 63.)
"""

from __future__ import annotations

import argparse
import contextlib
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import newton
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ROUND = ROOT / "assets" / "round_belt_task"
ROUND_SCENES = (ROUND / "round_belt_scene.yaml", ROUND / "round_belt_scene_wr_pin640b1.yaml")

from utils.directives import parse_directives

# handoffs/research/bike-chain-facts.md, "Welds" (branch scene column)
BOARD_T, BOARD_RPY = (0.67206688, -0.19727797, 0.01063954), (-0.69055534, -0.44629307, 89.58609652)
UR_T, UR_RPY = (1.3256854, -0.17264792, 0.02969374), (-0.91339513, 0.81993092, 179.09576213)

OK = True


def rep(name: str, ok: bool, msg: str = "") -> None:
    global OK
    OK &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}")


def raises(exc: type, fn) -> str:
    try:
        fn()
    except exc as e:
        return type(e).__name__
    except Exception as e:  # noqa: BLE001
        return f"wrong type {type(e).__name__}: {e}"
    return "no error"


def c1_parse(scene: Path) -> None:
    d = parse_directives(scene)
    chains = [e for e, _ in d.entries if getattr(e, "kind", "") == "add_roller_chain"]
    p = chains[0].params if len(chains) == 1 else {}
    rep("C1 chain", p.get("name") == "bike_chain" and p["num_links"] == 64
        and abs(p["pitch"] - 0.009525) < 1e-12 and not {"pin_stiffness", "variant"} & p.keys(),
        f"{len(chains)} add_roller_chain, num_links {p.get('num_links')}, pitch {p.get('pitch')}")
    err = 0.0
    for child, t, rpy in (("board::board", BOARD_T, BOARD_RPY), ("ur10::base_link", UR_T, UR_RPY)):
        w = d.weld(child).X_PC
        err = max(err, *(abs(a - b) for a, b in zip(w.translation, t)),
                  *(abs(a - b) for a, b in zip(w.rpy_deg, rpy)))
    rep("C1 welds", err < 1e-9, f"board + UR10 max |delta| {err:.1e}")


def c2_load(scene: Path, n_links: int) -> None:
    from bike_chain_task import constants as K
    from bike_chain_task.scene import build_scene
    from task_common.scene import make_builder

    b = make_builder()
    info = build_scene(b)
    jt, art = b.joint_type, b.joint_articulation
    bj = info.belt_joints
    rep("C2 bodies", len(info.belt_bodies) == n_links, f"{len(info.belt_bodies)} (want {n_links})")
    types = [int(jt[j]) for j in bj]
    in_art = [j for j in bj if art[j] != -1]
    rep("C2 joints", types[0] == int(newton.JointType.FREE)
        and all(t == int(newton.JointType.REVOLUTE) for t in types[1:])
        and len(in_art) == len(bj) - 1 and len({art[j] for j in in_art}) == 1
        and art[bj[-1]] == -1,
        f"{len(bj)} joints: FREE + {len(bj) - 2} REVOLUTE in one articulation + 1 closure")
    group = K._D.custom("add_roller_chain", "bike_chain").params.get("collision_group", -2)
    in_group = {s for s, g in enumerate(b.shape_collision_group) if g == group}
    chain_bodies = set(info.belt_bodies)
    collide = {s for s in info.belt_shapes if b.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)}
    rep("C2 collision_group", group < 0 and in_group == collide
        and all(b.shape_body[s] in chain_bodies for s in in_group),
        f"{len(in_group)} shapes in group {group}, all chain collision shapes, none else")
    labels = [b.joint_label[j] for j in info.pulley_joints]
    teeth = [sum(1 for s in info.pulley_shapes
                 if b.shape_label[s].startswith(f"board/{leaf}/collision")
                 and int(b.shape_type[s]) == int(newton.GeoType.MESH))
             for leaf in (x.split("/")[1].removesuffix("_joint") for x in K.SPROCKET_JOINT_LABELS)]
    rep("C2 sprockets", len(info.pulley_bodies) == 3 and labels == K.SPROCKET_JOINT_LABELS
        and teeth == list(K.SPROCKET_TEETH), f"labels ok, tooth shapes {teeth}")
    st = info.static_shapes
    rep("C2 statics", st == list(range(st[0], st[-1] + 1)), f"shapes {st[0]}..{st[-1]} contiguous")
    q = np.array(b.body_q, dtype=np.float64)[info.belt_bodies]
    c, ax = K.CHAIN_CENTER, K.CHAIN_SEMI_AXES
    if ax is None:
        resid = np.abs(np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) - K.CHAIN_RADIUS)
    else:
        resid = np.abs(np.hypot((q[:, 0] - c[0]) / ax[0], (q[:, 1] - c[1]) / ax[1]) - 1) * min(ax)
    step = np.linalg.norm(np.roll(q[:, :2], -1, axis=0) - q[:, :2], axis=1)
    mass = sum(b.body_mass[i] for i in info.belt_bodies)
    # ellipse residual is the directive's chord-walk tolerance (2.4e-6 m measured), not 1e-6
    rep("C2 geometry", resid.max() < 1e-5 and np.abs(step[:-1] - K.CHAIN_PITCH).max() < 1e-6
        and abs(step[-1] - K.CHAIN_PITCH) < 1e-4 and abs(mass - 0.20) < 0.002,
        f"on-curve {resid.max():.1e} m, pitch err {np.abs(step - K.CHAIN_PITCH).max():.1e} m "
        f"(closing {step[-1] - K.CHAIN_PITCH:+.1e}), mass {mass:.4f} kg")


@contextlib.contextmanager
def edited_scene(scene: Path, old: str, new: str):
    text = scene.read_text()
    assert old in text, old
    tmp = scene.with_name("_check_tmp_scene.yaml")  # relative file: paths resolve next to it
    tmp.write_text(text.replace(old, new))
    try:
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)


def c3_guards(scene: Path) -> None:
    from bike_chain_task.scene import build_scene
    from task_common.scene import make_builder

    def load(old, new):
        with edited_scene(scene, old, new) as tmp:
            build_scene(make_builder(), tmp)

    pitch = "      pitch: 0.009525\n"
    got = {
        "unknown key": raises(ValueError, lambda: load(pitch, pitch + "      bogus: 1\n")),
        "odd num_links": raises(ValueError, lambda: load("num_links: 64", "num_links: 63")),
        "group in use": raises(ValueError, lambda: load(pitch, pitch + "      collision_group: 0\n")),
    }
    rep("C3 guards", all(v == "ValueError" for v in got.values()), str(got))


def c4_round_belt() -> None:
    before = [repr(parse_directives(p).entries) for p in ROUND_SCENES]
    import round_belt_task.constants as rc

    k0 = {k: v for k, v in vars(rc).items() if k.isupper() and isinstance(v, (int, float, str))}
    import bike_chain_task.constants
    import bike_chain_task.simulation  # noqa: F401

    after = [repr(parse_directives(p).entries) for p in ROUND_SCENES]
    k1 = {k: v for k, v in vars(rc).items() if k.isupper() and isinstance(v, (int, float, str))}
    rep("C4 parse identity", before == after and k0 == k1,
        f"{len(ROUND_SCENES)} round-belt scenes, {len(k0)} scalar constants unchanged")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/checks/check_scene_directives.py"),
                        "--device", "cpu"], check=False, capture_output=True, text=True)
    rep("C4 check_scene_directives", r.returncode == 0 and "ALL DIRECTIVE CHECKS PASSED" in r.stdout,
        f"exit {r.returncode}, {time.time() - t0:.0f} s")


def c5_constants() -> None:
    from bike_chain_task import constants as K

    ref = [K.CHAIN_PITCH / (2 * math.sin(math.pi / n)) * 1e3 for n in (16, 32, 16)]
    rep("C5 pitch radius", np.allclose(K.SPROCKET_PITCH_RADIUS_MM, ref, atol=1e-9, rtol=0),
        f"{[round(x, 3) for x in ref]} mm")
    urdf = {c.get("name"): c.find("origin").get("xyz").split()
            for c in ET.parse(K.BOARD_URDF).getroot().iter("collision")}
    rims = [float(urdf[f"{s}_sprocket_rim"][2]) * 1e3 for s in ("small", "large", "tensioner")]
    rep("C5 tooth plane z", np.allclose(K.TOOTH_PLANE_Z_MM, rims, atol=1e-9, rtol=0)
        and K.BOARD_URDF.name == "bike_chain_task_board_aligned.urdf", f"{rims} mm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chain-links", type=int, default=64)
    args = ap.parse_args()
    wp.set_device(args.device)
    scene = ROOT / "assets/bike_chain_task/bike_chain_scene.yaml"
    t0 = time.time()
    c1_parse(scene)
    c2_load(scene, args.chain_links)
    c3_guards(scene)
    c5_constants()
    c4_round_belt()
    print(f"{'ALL PASS' if OK else 'FAILED'} ({time.time() - t0:.0f} s)")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
