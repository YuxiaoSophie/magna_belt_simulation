"""Recreate apps/demo_gui/tests/fixtures from recorded data (CPU-only, deterministic)."""

import json
import re
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "fixtures"
CMP = REPO / "data/lcs/realfirst/cmp"
E2E = CMP / "v12_152/e2e_V63j"
RUN = E2E / "V63j_tol50b_nominal"
REC = RUN / "recordings/episode_pre_place_1_osc_0"
DEMO = CMP / "v12_148/tiers/eps/s1/v12_147_demo__episode_gv_07_0.npz"
MILES = CMP / "v12_147/miles1_d3"
TRAIN = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v12_20261006/v12_152/J152_fpz0r05")
T = 40
NPTS = 128


def _cut(a, n):
    return a[:n] if a.ndim and a.shape[0] > n else a


def _cloud(a):
    if a.dtype == object:  # per-frame point arrays
        out = np.empty(a.shape, dtype=object)
        for i, x in enumerate(a):
            out[i] = x[:NPTS] if isinstance(x, np.ndarray) and x.ndim else x
        return out
    return a


def save_npz(src, dst, firsts):
    """Truncate arrays whose axis 0 is in `firsts` to T frames and clouds to NPTS points."""
    z = np.load(src, allow_pickle=True)
    out = {}
    for k in z.files:
        a = z[k]
        if a.ndim and a.shape[0] in firsts:
            a = _cloud(_cut(a, T))
            if k.startswith(("pcd_belt", "pcd_kin")) and a.dtype != object and a.ndim >= 2:
                a = a[:, :NPTS]
        out[k] = a
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def describe(path):
    z = np.load(path, allow_pickle=True)
    return [(k, z[k].shape, str(z[k].dtype)) for k in z.files]


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    ep = "episode_pre_place_1_osc_0.npz"
    run_dir = OUT / "e2e_run"
    cp(RUN / "index.json", run_dir / "index.json")
    n_ep = len(np.load(RUN / ep, allow_pickle=True)["utime"])
    save_npz(RUN / ep, run_dir / ep, {n_ep})
    rec = run_dir / "recordings/episode_pre_place_1_osc_0"
    cp(REC / "meta.json", rec / "meta.json")
    ch = json.loads((REC / "meta.json").read_text())["chunks"][0]
    save_npz(
        REC / "chunks/chunk_00000.npz",
        rec / "chunks/chunk_00000.npz",
        {ch["steps"], ch["state_frames"]},
    )
    for f in ("hard_suite.json", "hard_suite.md", "summarize.out", "suite.json"):
        cp(E2E / f, OUT / "e2e_suite" / f)
    n_demo = len(np.load(DEMO, allow_pickle=True)["utime"])
    save_npz(DEMO, OUT / "demo/episode_gv_07_0.npz", {n_demo})

    for f in ("miles_report.md",):
        cp(MILES / f, OUT / "miles" / f)
    b = json.loads((MILES / "buffer.json").read_text())
    b["rollouts"] = b["rollouts"][:5]
    (OUT / "miles/buffer.json").write_text(json.dumps(b))
    for p in sorted((MILES / "miles").glob("*.npz"))[:3]:  # tiny (2 frames), copied whole
        cp(p, OUT / "miles/miles" / p.name)

    cp(TRAIN / "config.yaml", OUT / "train/config.yaml")
    lines = (TRAIN / "train.log").read_text().splitlines()
    (OUT / "train/train.log").write_text("\n".join(lines[:30] + lines[-30:]) + "\n")
    r = json.loads((TRAIN / "refit_report.json").read_text())
    r.pop("split", None)
    (OUT / "train/refit_report.json").write_text(json.dumps(r))

    b = json.loads((CMP / "v12_152/tiers/W150all/buffer.json").read_text())
    b["rollouts"] = b["rollouts"][:5]
    (OUT / "buffer_W150all_head.json").write_text(json.dumps(b))
    cp(REPO / "data/lcs/start_states/grasp_variants/set2/index.json", OUT / "grasp_set2_index.json")
    cp(REPO / "data/lcs/start_states/pre_place_1.npz", OUT / "pre_place_1.npz")
    cp(CMP / "v12_151/arms.json", OUT / "arms.json")
    write_keys(rec)


def write_keys(rec):
    L = ["# Fixture key inventory", "", "Shapes/dtypes are from the ORIGINAL files (full T).", ""]
    srcs = {
        "e2e_run/episode_pre_place_1_osc_0.npz": RUN / "episode_pre_place_1_osc_0.npz",
        "e2e_run/recordings/.../chunks/chunk_00000.npz": REC / "chunks/chunk_00000.npz",
        "demo/episode_gv_07_0.npz": DEMO,
        "pre_place_1.npz (whole copy)": REPO / "data/lcs/start_states/pre_place_1.npz",
        "miles/miles/*.npz (3 whole copies, 2 frames each)": min((MILES / "miles").glob("*.npz")),
    }
    for name, p in srcs.items():
        L += [f"## {name}", "", "| key | shape | dtype |", "|---|---|---|"]
        L += [f"| {k} | {s} | {d} |" for k, s, d in describe(p)]
        L.append("")
    ep = {k for k, _, _ in describe(srcs["e2e_run/episode_pre_place_1_osc_0.npz"])}
    L += ["## Episode npz key presence (e2e run; demo identical except no sim_chain_from)", ""]
    L += ["| asked name | present | actual name |", "|---|---|---|"]
    alias = {"pc": "pcd_belt", "ctrl_hand_mm": "sim_cmd_ctrl_hand_mm", "time": "sim_time"}
    asked = "sim_ee_franka sim_ee_ur sim_belt_xyz pc state u actions sim_wrap_deg"
    asked += " sim_wrap_small_deg sim_grasp_ok sim_phase sim_cmd_source ctrl_hand_mm time utime"
    for k in [*asked.split(), "sim_chain_from"]:
        a = alias.get(k)
        if k in ep:
            L.append(f"| {k} | yes | {k} |")
        elif a and a in ep:
            note = " (+ pcd: per-frame object clouds)" if k == "pc" else ""
            L.append(f"| {k} | renamed | {a}{note} |")
        else:
            L.append(f"| {k} | NO | - |")
    L.append("")
    m = json.loads((REC / "meta.json").read_text())
    L += ["## meta.json", ""]
    L.append(f"- franka_hand.coords: `{m['robot_io'].get('franka_hand', {}).get('coords')}`")
    L.append(f"- driver_coords: `{m['driver_coords']}`")
    L.append(f"- effort_layout: `{m['effort_layout']}`")
    L.append("- chunks entry still lists steps=2000/state_frames=500; fixture chunk is cut to 40")
    L.append("")
    ix = json.loads((RUN / "index.json").read_text())
    L += ["## index.json", "", f"- episodes[0] keys: `{list(ix['episodes'][0])}`"]
    L += [f"- phase_labels: `{ix['phase_labels']}`", ""]
    keys = []
    for line in (TRAIN / "train.log").read_text().splitlines():
        if line.startswith("[epoch"):
            for k in re.findall(r"(\S+)=", line):
                if k not in keys:
                    keys.append(k)
    L += ["## train.log `[epoch N]` k=v keys", "", "`" + "`, `".join(keys) + "`", ""]
    L += ["## Not copied (paths + sizes)", ""]
    for p in (
        CMP / "encoder_study/sim/V63j/deploy_v39j/deploy.npz",
        CMP / "encoder_study/sim/V63j/decoder_decfit/decoder.npz",
    ):
        L.append(f"- `{p}` ({p.stat().st_size} bytes)")
    (OUT / "KEYS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
