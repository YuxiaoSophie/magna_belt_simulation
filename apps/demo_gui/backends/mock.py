"""Sim-free backend: copies / replays recorded data into the session; no GPU, lock or LCM."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from apps.demo_gui import jobs, telemetry
from apps.demo_gui.config import MOCK_MILES, MOCK_RUN, MOCK_START, MOCK_TRAIN, REPO
from apps.demo_gui.telemetry import EpisodeStreamer

MOCK_ARMS = REPO / "data/lcs/realfirst/cmp/v12_151/arms.json"


def _next(d: Path, prefix: str) -> Path:
    k = 0
    while (d / f"{prefix}{k:02d}").exists():
        k += 1
    return d / f"{prefix}{k:02d}"


def _copy_run(src: Path, dst: Path, start: dict | None = None) -> None:
    """``src``'s first episode (npz + recordings/<episode>) and an index.json listing only it."""
    idx = json.loads((src / "index.json").read_text())
    ep = idx["episodes"][0]
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / ep["file"], dst / ep["file"])
    rec = src / "recordings" / Path(ep["file"]).stem
    if rec.is_dir():
        shutil.copytree(rec, dst / "recordings" / rec.name, dirs_exist_ok=True)
    if start:
        ep["start_state"] = {**ep["start_state"], **start}
    idx["episodes"] = [ep]
    telemetry.write(dst / "index.json", idx)  # last, so a listed run is complete


def _stream(run_dir: Path, path: Path, stop, pace: float) -> None:
    s = EpisodeStreamer(run_dir, path, pace)
    s.start()
    while s.is_alive():
        if stop.wait(0.1):
            s.stop()
    s.join()


class MockBackend:
    """``pace`` scales every replay delay (1.0 = real time)."""

    supports_teleop = True

    def __init__(self, pace: float = 1.0) -> None:
        self.pace = pace
        self._demo: tuple[Path, jobs.Job] | None = None

    def make_start(self, session, franka: dict, ur: dict) -> jobs.Job:
        def fn(stop):
            stop.wait(self.pace)
            p = session.starts / "index.json"
            idx = (json.loads(p.read_text()) if p.is_file()
                   else {"set": "demo_gui", "variants": []})
            sid = f"s{len(idx['variants']):02d}"
            dst = session.starts / f"{sid}_osc.npz"
            shutil.copy2(MOCK_START, dst)
            idx["variants"].append({
                "id": sid,
                "commanded": {"franka_slide_mm": franka["slide_mm"],
                              "franka_roll_deg": franka["roll_deg"],
                              "ur_slide_mm": ur["slide_mm"], "ur_roll_deg": ur["roll_deg"]}
                | {k: float(v) for k, v in (  # Variant.commanded(): only when non-zero
                    ("ur_depth_mm", ur.get("depth_mm")), ("ur_pivot_deg", ur.get("pivot_deg")),
                    ("franka_pivot_deg", franka.get("pivot_deg")),
                    ("franka_depth_mm", franka.get("depth_mm"))) if v},
                "held": True, "reason": None, "measured": None, "file": str(dst),
                "notes": f"mock: copy of {MOCK_START.name}"})
            telemetry.write(p, idx)
        return jobs.run_thread(session, "make_start", fn)

    def record_demo(self, session, start_id: str, mode: str = "scripted") -> jobs.Job:
        d = _next(session.demos, "demo_")
        start = {"id": start_id, "file": str(session.starts / f"{start_id}_osc.npz")}

        def fn(stop):
            _copy_run(MOCK_RUN, d, start)
            _stream(d, session.telemetry_path, stop, self.pace)
        job = jobs.run_thread(session, "demo", fn)
        self._demo = (d, job)
        return job

    def stop_demo(self, session) -> None:
        if self._demo:
            jobs.stop(self._demo[1])

    def reset_demo(self, session) -> None:
        if self._demo:
            self.stop_demo(session)
            shutil.rmtree(self._demo[0], ignore_errors=True)
            self._demo = None

    def run_miles(self, session, demos: list[dict], cycles: int = 4, wp_stride: int = 5) -> jobs.Job:
        d = _next(session.miles, "miles_")

        def fn(stop):
            (d / "miles").mkdir(parents=True)
            shutil.copy2(MOCK_MILES / "miles_report.md", d)
            buf = json.loads((MOCK_MILES / "buffer.json").read_text())
            rows = []
            for r in buf["rollouts"][:20]:
                if stop.is_set():
                    return
                src = Path(r["episode"])
                shutil.copy2(src, d / "miles" / src.name)
                rows.append({**r, "episode": str(d / "miles" / src.name)})
                telemetry.write(d / "buffer.json", {**buf, "n": len(rows), "rollouts": rows})
                stop.wait(0.5 * self.pace)
        return jobs.run_thread(session, "miles", fn)

    def train(self, session, tag: str, rows: list[dict], epochs: int, seed: int = 5) -> jobs.Job:
        d = session.train / tag
        d.mkdir(exist_ok=True)
        telemetry.write(d / "buffer.json", {"version": 2, "n": len(rows), "rollouts": rows})

        def fn(stop):
            # shortcut: replays the recorded log whatever `epochs` is; fine for a mock.
            with open(d / "train.log", "w") as log:
                for line in (MOCK_TRAIN / "train.log").read_text().splitlines(keepends=True):
                    if stop.wait(0.2 * self.pace):
                        return
                    log.write(line)
                    log.flush()
            shutil.copy2(MOCK_TRAIN / "refit_report.json", d)
        return jobs.run_thread(session, f"train_{tag}", fn)

    def export(self, session, tag: str) -> jobs.Job:
        def fn(stop):
            d = session.train / tag / "export"
            d.mkdir(parents=True, exist_ok=True)
            if stop.wait(self.pace):
                return
            arm = json.loads(MOCK_ARMS.read_text())["V63j_tol50b"]
            telemetry.write(d / "arm.json", {tag: arm})  # mock: the pre-exported V63j arm
            (d / "status").write_text(f"mock export {tag} ok: V63j_tol50b\n")
        return jobs.run_thread(session, f"export_{tag}", fn)

    def evaluate(self, session, name: str, model: dict, starts: list[str],
                 repeats: int = 1) -> jobs.Job:
        from apps.demo_gui.evalrun import new_out

        fx = Path(__file__).resolve().parents[1] / "tests/fixtures"
        d = new_out(session, name)
        d.mkdir(parents=True)

        def fn(stop):
            # shortcut: one fixture episode as <name>_nominal whatever starts / repeats; a mock.
            if stop.wait(3 * self.pace):
                return
            _copy_run(fx / "e2e_run", d / f"{name}_nominal")
            for f in (fx / "e2e_suite").iterdir():
                shutil.copy2(f, d)
        return jobs.run_thread(session, f"eval_{name}", fn)
