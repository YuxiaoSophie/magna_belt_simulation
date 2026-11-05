#!/usr/bin/env python3
"""Checks of the opt-in gripper-aware encoder and grasp gate (V12-131).

R0 sources: mppi_lcs / mppi_warp / the MPPI node / eval minus the listed V12-131 hunks == pre.
R1 off identical (``--pre-dir`` = v12_131/pre): the pre and tree ``latent_encoder`` give equal z
   on the reference clouds of non-grip deploys, and equal latent-message bytes for a 40-d proprio.
   Pre and tree ``TorchLcs`` give equal step / rollout on deploys without ``grip_gate``.
   With ``--cuda`` the fused rollout is compared too.
R2 gate: ``set_gate(ones)`` gives the ungated step; ``set_gate(g)`` gives the ungated rollout of
   ``g * u``, torch (and fused with ``--cuda``).
R3 features: magna's ``recorded_grip_features`` equals lcs_learning's on W3 episodes. The bits
   are (1, 1) on stage 1 and (1, 0) on stage 2, and ``grip_gate_vector`` is checked.
R4 grip encoder and node: a synthetic grip deploy (V59f + zero grip columns) gives V59f's z.
   Missing grip is refused, and a 44-d proprio message is accepted. The node subscribes the
   gripper channels only for grip deploys, holds back until status arrives, and infers unseen
   commands as in recorded data.

R5 (``--ctx-deploy``, a seg_ctx deploy dir): magna's ``BeltCtxEncoder`` == the exporter's z,
   ``ctx_kinematics`` (magna FK) == lcs_learning's ctx vector, z_std inf on exactly the z_ctx
   dims, so ``TorchLcs`` / ``LearnedLcs`` whitened distances (goal / stage cost) ignore z_ctx;
   EE pose +-10 mm / +-10 deg on a goal frame's state moves z_ctx but not z_belt nor the goal
   distance; the decoder takes the full z; a node on that deploy accepts the sim board pose and
   refuses another.

    uv run --frozen python scripts/checks/check_grip_encoder.py --pre-dir \\
        data/lcs/realfirst/cmp/v12_131/pre [--cuda] [--ctx-deploy DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from task_common import latent_encoder as le
from task_common import mppi_lcs as tree

LCS_LEARNING = Path("/home/hienbui/git/lcs_learning")
ES = REPO / "data/lcs/realfirst/cmp/encoder_study/sim"
DEPLOYS = [ES / "V59f/deploy_v27f/deploy.npz", ES / "V59f_s2/deploy_v27f/deploy.npz",
           ES / "V50b/deploy_v18b/deploy.npz"]
W3 = REPO / "data/lcs/realfirst/cmp/v12_128/tiers/W3/buffer.json"


# V12-131 hunks (tree text, pre text) per file; older source-identity checks strip them too
V131_HUNKS = {
    "src/task_common/mppi_lcs.py": [
        ("""        # V12-131 grip gate: u~ = g * u in B u and H u (set_gate per latent); off = deploys without
        self.grip_gate = bool(_scalar(d["grip_gate"])) if "grip_gate" in d else False
        self._gate = None
""", ""),
        ("""    def set_gate(self, g) -> None:
        \"\"\"Input gate ``g`` (n_u,): B and H columns scaled in place (CUDA-graph safe; the fused
        rollout's copies too), so every rollout sees ``g * u``; constant over the horizon.\"\"\"
        g = np.asarray(g, dtype=np.float64).reshape(self.n_u)
        if self._gate is not None and np.array_equal(g, self._gate):
            return
        self._gate = g.copy()
        gt = torch.as_tensor(g, dtype=self.dtype, device=self.device)
        self.B.copy_(torch.as_tensor(self.np["B"], dtype=self.dtype, device=self.device) * gt)
        self.H.copy_(torch.as_tensor(self.np["H"], dtype=self.dtype, device=self.device) * gt)
        self._sH.copy_(self.step_size * self.H)
        if self._fused is not None:
            self._fused.set_gate(g)

""", ""),
        ("""            if self._gate is not None:
                self._fused.set_gate(self._gate)
""", "")],
    "src/task_common/mppi_warp.py": [
        ("""    def set_gate(self, g) -> None:
        \"\"\"``TorchLcs.set_gate``: B and s H columns times ``g``, written into the same arrays.\"\"\"
        n, s, ty = self.lcs.np, self.lcs.step_size, self.ty
        for k, m, t in ((1, n["B"], "mxu"), (6, s * n["H"], "mlu")):
            src = self.wp.array([np.asarray(m * np.asarray(g)[None, :], dtype=np.float32)],
                                dtype=ty[t], device=self.dev)
            self.wp.copy(self.mats[k], src)

""", "")],
    "scripts/lcs/mppi_controller_node.py": [
        ("    GRIP_DIM,\n    LATENT_STATE_CHANNEL,", "    LATENT_STATE_CHANNEL,"),
        ("    grip_gate_vector,\n    parse_latent", "    parse_latent"),
        ("import STATE_DIM, delta_rotvec", "import delta_rotvec"),
        ("""        if self.lcs.grip_gate:  # V12-131: the measured grasp bits gate u over the whole horizon
            if len(_prop) != STATE_DIM + GRIP_DIM:
                raise ValueError(f"grip-gated deploy: proprio has {len(_prop)} entries, needs "
                                 f"{STATE_DIM} + {GRIP_DIM} gripper features")
            self.lcs.set_gate(grip_gate_vector(_prop[STATE_DIM:]))
""", "")],
    "scripts/lcs/eval_learned_mpc.py": [
        ("    FRANKA_CLOSED_MM,\n    UR_CLOSED_BYTE,\n", ""),
        ("    BeltCtxEncoder,\n", ""),
        ("    grip_features,\n", ""),
        ("""        prop = frame["prop"]
        if getattr(ctx.encoder, "grip_dim", 0):  # V12-131: measured gripper state + own commands
            hand_cmd, ur_cmd = sim.gripper_commands()
            grip = grip_features(grasp.franka_width_mm, grasp.ur_status,
                                 hand_cmd is None or hand_cmd < FRANKA_CLOSED_MM,
                                 (grasp.ur_status if ur_cmd is None else ur_cmd) >= UR_CLOSED_BYTE)
            kw = {"scene": frame["cloud"]} if isinstance(ctx.encoder, BeltCtxEncoder) else {}
            z = ctx.encoder.encode(encoder_input(ctx.encoder, frame), prop, frame["belt_pts"],
                                   grip=grip, **kw)
            prop = np.r_[prop, grip]  # the 44-d proprio block: the MPPI node's gate bits
        else:
            z = ctx.encoder.encode(encoder_input(ctx.encoder, frame), prop, frame["belt_pts"])
""", """        z = ctx.encoder.encode(encoder_input(ctx.encoder, frame), frame["prop"], frame["belt_pts"])
"""),
        ("latent_state_message(utime, t, z, ee_f, ee_u, prop))",
         'latent_state_message(utime, t, z, ee_f, ee_u, frame["prop"]))')],
}


# V12-132 --franka-grip-n hunks (eval harness); undone before the V12-131 ones
V132_HUNKS = {
    "scripts/lcs/eval_learned_mpc.py": [
        ("""# --franka-grip-n (V12-132): force-limited Franka squeeze in the sim, dropped at default.
GRIP_ARG_DEFAULTS = {"franka_grip_n": None}
""", ""),
        ("""    p.add_argument("--franka-grip-n", type=float, default=GRIP_ARG_DEFAULTS["franka_grip_n"],
                   metavar="N", help="sim Franka squeeze force [N, both fingers], force-limited "
                                     "(RoundBeltLcmSimulation --franka-grip-n); default the yaml's")
""", ""),
        ("""    for key, value in {**PREFIX_ARG_DEFAULTS, **STAGE2_ARG_DEFAULTS, **PLANAGE_ARG_DEFAULTS,
                       **GRIP_ARG_DEFAULTS}.items():""",
         """    for key, value in {**PREFIX_ARG_DEFAULTS, **STAGE2_ARG_DEFAULTS, **PLANAGE_ARG_DEFAULTS}.items():"""),
        ("""    grip_kw = {} if args.franka_grip_n is None else {"franka_grip_n": args.franka_grip_n}
""", ""),
        ("""                                       viewer=_live_viewer(args) if teleop else None, **grip_kw)""",
         """                                       viewer=_live_viewer(args) if teleop else None)""")],
}


# V12-133 V61e hunks (MPPI node: goals re-encoded with the current robot vector)
V133_HUNKS = {
    "scripts/lcs/mppi_controller_node.py": [
        ("    RV_TYPE,\n    DemoGoals,\n    LatentEncoder,\n", "    DemoGoals,\n"),
        ("        self.rv_goals = self._rv_goal_base(cli)\n", ""),
        ("""    def _rv_goal_base(self, cli):
        \"\"\"V61e (scene_rv deploys): the waypoints' rv-free scene parts; each latent re-adds the
        CURRENT robot part to every goal (``_rv_goals``). None for every other deploy.\"\"\"
        with np.load(cli.deploy, allow_pickle=False) as d:
            if "encoder_type" not in d.files or str(d["encoder_type"]) != RV_TYPE:
                return None
        rv_free = cli.goal.startswith("waypoints:") and bool(
            np.load(cli.goal.split(":", 1)[1]).get("rv_free", False))
        if self.wp is None or cli.goal_mode != "stage" or not rv_free \\
                or getattr(cli, "waypoint_bank", None) is not None:
            raise ValueError(f"{RV_TYPE} deploy needs --goal waypoints:<rv_free npz> --goal-mode "
                             "stage (no bank)")
        self.rv_enc = LatentEncoder.load(cli.deploy)
        return self.wp.z_traj.copy(), self.wp.z_wp.copy()

    def _rv_goals(self, prop) -> None:
        zr = self.rv_enc.robot_part(prop[:STATE_DIM], prop[STATE_DIM:STATE_DIM + GRIP_DIM])
        self.wp.z_traj[:] = self.rv_goals[0] + zr
        self.wp.z_wp[:] = self.rv_goals[1] + zr
        for st in self.stages:
            st.goal[:] = self.wp.z_wp[st.index]

""", ""),
        ("""        if self.rv_goals is not None:  # V61e: goals = scene part + the current robot part
            self._rv_goals(_prop)
""", "")],
}


def strip_v131(rel: str, txt: str) -> str:
    """``txt`` of tree file ``rel`` with the V12-133, V12-132 and V12-131 hunks undone (each once)."""
    for new, old in V133_HUNKS.get(rel, []) + V132_HUNKS.get(rel, []) + V131_HUNKS.get(rel, []):
        if txt.count(new) != 1:
            raise AssertionError(f"{rel}: V12-131 hunk not found once: {new[:60]!r}")
        txt = txt.replace(new, old)
    return txt


def r0(pre_dir: Path | None) -> str:
    if pre_dir is None:
        return "no --pre-dir: skipped"
    for rel in V131_HUNKS:
        _require(strip_v131(rel, (REPO / rel).read_text()) == (pre_dir / Path(rel).name)
                 .read_text(), f"{rel}: tree minus the V12-131 hunks != pre")
    return f"{len(V131_HUNKS)} sources == pre minus the V12-131 hunks"


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _inputs(lcs, rng, K=64, H=7):
    z0 = rng.normal(size=lcs.n_x)
    U = torch.as_tensor(rng.normal(size=(K, H, lcs.n_u)) * 2e-3, dtype=torch.float32,
                        device=lcs.device)
    return z0, np.zeros(6), U


def r1(pre_dir: Path | None, cuda: bool) -> str:
    if pre_dir is None:
        return "no --pre-dir: skipped"
    pre_le = _load("pre_latent_encoder", pre_dir / "latent_encoder.py")
    pre_ml = _load("pre_mppi_lcs", pre_dir / "mppi_lcs.py")
    rng = np.random.default_rng(0)
    n_z = n_msg = 0
    for dep in DEPLOYS:
        a, b = pre_le.LatentEncoder.load(dep), le.LatentEncoder.load(dep)
        _require(not getattr(b, "grip_dim", 0), f"{dep}: grip_dim set")
        with np.load(dep.parent / "reference_vectors.npz", allow_pickle=True) as r:
            pcs = list(r["pc_raw"])
            props = r["prop"] if "prop" in r.files else [None] * len(pcs)
            belts = r["belt_raw"] if "belt_raw" in r.files else [None] * len(pcs)
        for pc, p, bt in zip(pcs, props, belts, strict=True):
            _require(np.array_equal(a.encode(pc, p, bt), b.encode(pc, p, bt)), f"{dep}: z")
            n_z += 1
        _require(np.array_equal(a.encode_batch(pcs, props, belts),
                                b.encode_batch(pcs, props, belts)), f"{dep}: encode_batch")
        z, prop = rng.normal(size=a.latent_dim), rng.normal(size=40)
        pose = np.r_[rng.normal(size=3), 1.0, 0, 0, 0]
        _require(pre_le.latent_state_message(5, 5e-6, z, pose, pose, prop).encode()
                 == le.latent_state_message(5, 5e-6, z, pose, pose, prop).encode(), "msg bytes")
        n_msg += 1
    devs = ["cpu"] + (["cuda"] if cuda else [])
    n_lcs = 0
    for dev in devs:
        for dep in DEPLOYS:
            a, b = (m.TorchLcs.from_deploy(dep, dev) for m in (pre_ml, tree))
            _require(not b.grip_gate, f"{dep}: grip_gate on")
            z0, p0, U = _inputs(a, np.random.default_rng(1))
            for x, y in zip(a.rollout(z0, p0, U), b.rollout(z0, p0, U), strict=True):
                _require(torch.equal(x, y), f"{dep} {dev}: rollout")
            if dev == "cuda":
                for x, y in zip(a.rollout_fused(z0, p0, U), b.rollout_fused(z0, p0, U),
                                strict=True):
                    _require(torch.equal(x, y), f"{dep}: rollout_fused")
            n_lcs += 1
    return (f"{n_z} reference z and {len(DEPLOYS)} batches ==, {n_msg} message bytes ==; "
            f"TorchLcs rollout{' + fused' if cuda else ''} == pre on {n_lcs} deploy/device pairs")


def r2(cuda: bool) -> str:
    out = []
    for dev in ["cpu"] + (["cuda"] if cuda else []):
        dep = DEPLOYS[0]
        ref, gated = (tree.TorchLcs.from_deploy(dep, dev) for _ in range(2))
        z0, p0, U = _inputs(ref, np.random.default_rng(2))
        Z = ref.tensor(np.random.default_rng(3).normal(size=(8, ref.n_x)))
        gated.set_gate(np.ones(12))
        for x, y in zip(ref.step(Z, U[:8, 0]), gated.step(Z, U[:8, 0]), strict=True):
            _require(torch.equal(x, y), f"{dev}: ones gate != ungated step")
        g = le.grip_gate_vector([0.4, 0.75, 1.0, 0.0])
        _require(g.tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], f"gate vector {g}")
        roll = [ref.rollout] + ([ref.rollout_fused] if dev == "cuda" else [])
        groll = [gated.rollout] + ([gated.rollout_fused] if dev == "cuda" else [])
        if dev == "cuda":
            gated.set_gate(np.ones(12))
            gated.rollout_fused(z0, p0, U)  # the fused copies exist before the gate changes
        gated.set_gate(g)
        Ug = U * ref.tensor(g)
        for f_ref, f_gated in zip(roll, groll, strict=True):
            Zr, Zg = f_ref(z0, p0, Ug)[0], f_gated(z0, p0, U)[0]
            err = float((Zr - Zg).abs().max())
            _require(err <= 1e-5, f"{dev} {f_gated.__name__}: gated != rollout(g*u) {err:.1e}")
            out.append(f"{dev} {f_gated.__name__} {err:.1e}")
        _require(float((ref.rollout(z0, p0, U)[0] - groll[0](z0, p0, U)[0]).abs().max()) > 1e-4,
                 "gate has no effect")
    return "ones == ungated; gated rollout == rollout(g*u): " + ", ".join(out)


def r3() -> str:
    sys.path.insert(0, str(LCS_LEARNING))
    from lcs_learning import belt_cloud_ae as bca

    sys.path.pop(0)
    for k in ("GRIP_DIM", "FRANKA_HELD_MM", "FRANKA_CLOSED_MM", "UR_CLOSED_BYTE",
              "UR_FULL_BYTE", "FRANKA_U", "UR_U"):
        _require(getattr(bca, k) == getattr(le, k), f"constant {k} differs")
    rows = json.loads(W3.read_text())["rollouts"]
    picks = {"s1_chain": [], "stage2_mpc": []}
    for r in rows:
        if r["kind"] in picks and len(picks[r["kind"]]) < 6:
            picks[r["kind"]].append(r["episode"])
    bits, n = {}, 0
    for key, eps in picks.items():
        _require(eps, f"no {key} episodes in W3")
        for ep in eps:
            with np.load(ep, allow_pickle=True) as z:
                args = (z["sim_hand_mm"], z["sim_robotiq_byte"], z["sim_cmd_source"])
            a, b = le.recorded_grip_features(*args), bca.recorded_grip_features(*args)
            _require(np.array_equal(a, b), f"{ep}: magna != lcs_learning")
            bits.setdefault(key, []).append(a[:, 2:].mean(0))
            n += len(a)
    mb = {k: np.round(np.mean(v, 0), 3).tolist() for k, v in bits.items()}
    _require(mb["s1_chain"] == [1.0, 1.0] and mb["stage2_mpc"][1] == 0.0, f"mean bits {mb}")
    return f"{n} frames magna == lcs_learning; mean (franka, ur) bits {mb}"


def _grip_deploy(td: Path) -> Path:
    with np.load(DEPLOYS[0], allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    w = d[f"{le._ENC}fuse_mlp__0__weight"]
    d[f"{le._ENC}fuse_mlp__0__weight"] = np.concatenate([w, np.zeros((len(w), 4), w.dtype)], 1)
    d["encoder_type"], d["encoder_grip_dim"] = np.array(le.GRIP_TYPE), np.array(4)
    d["grip_gate"] = np.array(True)
    out = td / "deploy.npz"
    np.savez(out, **d)
    return out


class _FakeLcm:
    def __init__(self) -> None:
        self.subs: dict[str, object] = {}
        self.sent: list[tuple[str, bytes]] = []

    def subscribe(self, ch, fn) -> None:
        self.subs[ch] = fn

    def publish(self, ch, data) -> None:
        self.sent.append((ch, data))


def r4() -> str:
    import latent_encoder_node as ln

    from drake import lcmt_schunk_wsg_command, lcmt_schunk_wsg_status
    from robotiq import lcmt_robotiq_command, lcmt_robotiq_status

    with tempfile.TemporaryDirectory() as td:
        dep = _grip_deploy(Path(td))
        e0, eg = le.LatentEncoder.load(DEPLOYS[0]), le.LatentEncoder.load(dep)
        _require(eg.grip_dim == 4 and tree.TorchLcs.from_deploy(dep, "cpu").grip_gate, "keys")
        with np.load(DEPLOYS[0].parent / "reference_vectors.npz", allow_pickle=True) as r:
            pcs = list(r["pc_raw"])
        g = le.grip_features(8.0, 250, True, True)
        _require(all(np.array_equal(e0.encode(pc), eg.encode(pc, grip=g)) for pc in pcs),
                 "zero grip columns change z")
        _require(np.array_equal(e0.encode_batch(pcs), eg.encode_batch(pcs, grips=np.stack(
            [g] * len(pcs)))), "encode_batch")
        try:
            eg.encode(pcs[0])
            _require(False, "missing grip accepted")
        except ValueError:
            pass
        pose = np.r_[0.0, 0, 0, 1, 0, 0, 0]
        msg = le.latent_state_message(1, 1e-6, np.zeros(24), pose, pose, np.arange(44.0))
        back = le.parse_latent_state_message(type(msg).decode(msg.encode()))[-1]
        _require(np.array_equal(back, np.arange(44.0)), "44-d proprio round trip")
        nodes = {}
        for name, d in (("off", DEPLOYS[0]), ("grip", dep)):
            cli = ln.create_parser().parse_args(["--lcm-url", "udpm://239.255.78.59:7859?ttl=0",
                                                 "--deploy", str(d), "--belt-seg"])
            nodes[name] = ln.LatentEncoderNode(cli, _FakeLcm())
    grip_ch = {"PANDA_HAND_STATUS", "ROBOTIQ_STATUS", "PANDA_HAND_COMMAND", "ROBOTIQ_COMMAND"}
    _require(not grip_ch & set(nodes["off"].lc.subs), "non-grip node subscribes gripper")
    n = nodes["grip"]
    _require(grip_ch <= set(n.lc.subs) and n.grip() is None, "grip node subs / initial grip")

    def send(ch, msg_type, **kw):
        m = msg_type()
        for k, v in kw.items():
            setattr(m, k, v)
        n.lc.subs[ch](ch, m.encode())

    send("PANDA_HAND_STATUS", lcmt_schunk_wsg_status, actual_position_mm=8.0)
    _require(n.grip() is None, "grip before the Robotiq status")
    send("ROBOTIQ_STATUS", lcmt_robotiq_status, position=250)
    _require(np.array_equal(n.grip(), le.grip_features(8.0, 250, True, True)), "inferred grip")
    send("ROBOTIQ_STATUS", lcmt_robotiq_status, position=191)
    _require(n.grip()[3] == 0.0, "open robotiq bit")
    send("ROBOTIQ_STATUS", lcmt_robotiq_status, position=250)
    send("ROBOTIQ_COMMAND", lcmt_robotiq_command, position=191)
    send("PANDA_HAND_COMMAND", lcmt_schunk_wsg_command, target_position_mm=60.0)
    _require(n.grip()[2:].tolist() == [0.0, 0.0], "open commands do not clear the bits")
    return ("synthetic grip deploy: z == V59f, missing grip refused, 44-d message ok; node: "
            "gripper subs only for grip deploys, waits for status, infers / uses commands")


def _rotz(q, deg: float) -> np.ndarray:
    h = np.radians(deg) / 2.0
    a = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def r5(d: Path | None) -> str:
    if d is None:
        return "no --ctx-deploy: skipped"
    import latent_encoder_node as ln

    enc = le.LatentEncoder.load(d / "deploy.npz")
    _require(isinstance(enc, le.BeltCtxEncoder), "not a ctx deploy")
    if isinstance(enc, le.SceneRvEncoder):
        return "scene_rv deploy: see check_scene_rv.py"
    r = dict(np.load(d / "reference_vectors.npz", allow_pickle=True))
    K, nb = len(r["z"]), enc.belt_dim
    z = np.stack([enc.encode(r["pc_raw"][k], r["prop"][k], grip=r["grip"][k],
                             scene=r["scene_raw"][k]) for k in range(K)])
    z_err = float(np.abs(z - r["z"]).max())
    _require(z_err <= 1e-5 * max(1.0, float(np.abs(r["z"]).max())), f"z vs exporter {z_err:.1e}")
    v = np.stack([le.ctx_kinematics(p, enc.frames["X_BW"]) for p in r["prop"]])
    v_err = float(np.abs(v - r["ctx_vec"][:, :le.CTX_KIN_DIM]).max())
    _require(v_err <= 1e-6, f"ctx_kinematics magna vs lcs_learning {v_err:.1e}")
    tl = tree.TorchLcs.from_deploy(d / "deploy.npz", "cpu")
    ll = le.LearnedLcs.load(d / "deploy.npz")
    _require(np.isinf(tl.z_std_np[nb:]).all() and np.isfinite(tl.z_std_np[:nb]).all(), "z_std")
    goal = tl.np["z_goal"]
    moved = np.r_[goal[:nb], goal[nb:] + 7.0]
    _require(tl.whitened_dist(moved, goal) == 0.0 == ll.whitened_dist(moved, goal),
             "z_ctx weighs in the whitened distance")
    k = 0
    zg = enc.encode(r["pc_raw"][k], r["prop"][k], grip=r["grip"][k], scene=r["scene_raw"][k])
    worst_b, ctx_moved = 0.0, 0.0
    for dp, deg in ((0.01, 0.0), (-0.01, 0.0), (0.0, 10.0), (0.0, -10.0)):
        p = np.asarray(r["prop"][k], np.float64).copy()
        for sl in (slice(26, 33), slice(33, 40)):
            p[sl][:3] += dp
            p[sl][3:] = _rotz(p[sl][3:], deg)
        zp = enc.encode(r["pc_raw"][k], p, grip=r["grip"][k], scene=r["scene_raw"][k])
        worst_b = max(worst_b, float(np.abs(zp[:nb] - zg[:nb]).max()),
                      abs(tl.whitened_dist(zp, goal) - tl.whitened_dist(zg, goal)))
        ctx_moved = max(ctx_moved, float(np.abs(zp[nb:] - zg[nb:]).max(initial=0.0)))
    _require(worst_b == 0.0 and (ctx_moved > 0.0 or enc.ctx_dim == 0),  # V61d: no z_ctx
             f"EE perturbation: belt {worst_b} ctx {ctx_moved}")
    if (d / "decoder.npz").is_file():
        _require(le.LatentDecoder.load(d / "decoder.npz").decode(z[0]).ndim == 2, "decoder")
    cli = ln.create_parser().parse_args(["--lcm-url", "udpm://239.255.78.59:7859?ttl=0",
                                         "--deploy", str(d / "deploy.npz"), "--belt-seg"])
    node = ln.LatentEncoderNode(cli, _FakeLcm())
    _require(isinstance(node.encoder, le.BeltCtxEncoder), "node encoder")
    try:
        node.encoder.set_board(np.eye(4))
        _require(False, "another board pose accepted")
    except ValueError:
        pass
    return (f"K={K}: z vs exporter {z_err:.1e}, ctx_kinematics magna vs lcs_learning {v_err:.1e}; "
            f"z_std inf on the {enc.ctx_dim} z_ctx dims, whitened distances ignore z_ctx; EE "
            f"+-10 mm / +-10 deg: z_belt and goal distance change 0, z_ctx max {ctx_moved:.2e}; "
            "node accepts the sim board, refuses another")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, default=None)
    ap.add_argument("--cuda", action="store_true", help="also check the fused (Warp) rollout")
    ap.add_argument("--ctx-deploy", type=Path, default=None, help="a seg_ctx deploy dir (R5)")
    a = ap.parse_args()
    ok = True
    for name, fn in (("R0", lambda: r0(a.pre_dir)), ("R1", lambda: r1(a.pre_dir, a.cuda)), ("R2", lambda: r2(a.cuda)),
                     ("R3", r3), ("R4", r4), ("R5", lambda: r5(a.ctx_deploy))):
        try:
            print(f"[PASS] {name} {fn()}", flush=True)
        except AssertionError as e:
            ok = False
            print(f"[FAIL] {name} {e}", flush=True)
    print("ALL GRIP-ENCODER CHECKS PASSED" if ok else "GRIP-ENCODER CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
