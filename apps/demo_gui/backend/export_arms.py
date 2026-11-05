"""configs148.py + make_wp.py (x0.5 both stages) of post152.sh with every output under ``<R>``:
``<R>/configs/<n>_{stage1,stage2,ch_stage2}.yaml``, ``<R>/wp/<n>_tol50b_s{1,2}_latent_waypoints.npz``,
``<R>/arms.json`` (``<n>`` and ``<n>_tol50b``) and ``<R>/arm.json`` = ``{<n>: the tol50b arm}``.

usage: export_arms.py <R> <S (parent of the <n>/ and <n>_s2/ prep dirs)> <n>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

CMP = Path(__file__).resolve().parents[3] / "data/lcs/realfirst/cmp"
sys.path.insert(0, str(CMP / "v12_110"))
import suite_configs as sc


def configs(R: Path, S: Path, n: str) -> dict:
    """configs148.py for one model: the stage yamls + its arms entry."""
    (R / "configs").mkdir(parents=True, exist_ok=True)
    sc.config(sc.BASE1, S / n, R / "configs" / f"{n}_stage1.yaml")
    s2 = R / "configs" / f"{n}_stage2.yaml"
    sc.config(sc.BASE2, S / f"{n}_s2", s2)
    doc = yaml.safe_load(s2.read_text())
    doc["knot0_chain"] = {"mode": "shift", "cap_mm": 15.0, "cap_deg": 5.0}
    ch = R / "configs" / f"{n}_ch_stage2.yaml"
    ch.write_text(f"# demo GUI {n}_ch: {s2.name} + knot0_chain\n" + yaml.safe_dump(doc, sort_keys=False))
    d1 = json.loads((S / n / "deploy.json").read_text())["deploy"]
    d2 = json.loads((S / f"{n}_s2" / "deploy.json").read_text())["deploy"]
    return {"dep": str(Path(d2).parent), "deploy1": d1, "wp1": str(S / n / "latent_waypoints.npz"),
            "cfg1": str(R / "configs" / f"{n}_stage1.yaml"), "deploy2": d2,
            "wp2": str(S / f"{n}_s2" / "latent_waypoints.npz"), "cfg2": str(ch)}


def scale_wp(R: Path, arm: dict, name: str, s: float) -> dict:
    """make_wp.py: tol[k] = max(s * tol[k], floor) for tol > 0, in both stages."""
    arm = dict(arm)
    for key in ("wp2", "wp1"):
        with np.load(arm[key], allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        tol = d["tol"].copy()
        tol[tol > 0] = np.maximum(s * tol[tol > 0], float(d["floor"]))
        d["tol"], d["tol_scale"] = tol, np.float64(s)
        out = R / "wp" / f"{name}_s{key[-1]}_latent_waypoints.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **d)
        arm[key] = str(out)
    return arm


def main(R: str, S: str, n: str) -> None:
    R, S = Path(R), Path(S)
    base = configs(R, S, n)
    tol = scale_wp(R, base, f"{n}_tol50b", 0.5)
    (R / "arms.json").write_text(json.dumps({n: base, f"{n}_tol50b": tol}, indent=1) + "\n")
    (R / "arm.json").write_text(json.dumps({n: tol}, indent=1) + "\n")
    print(n, tol)


if __name__ == "__main__":
    main(*sys.argv[1:])
