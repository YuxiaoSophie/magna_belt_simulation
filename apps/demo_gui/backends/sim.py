"""Real sim backend: Newton + magna controllers as detached jobs on the app's LCM port."""

from __future__ import annotations

import os
import shlex
import shutil
import time
from pathlib import Path

from apps.demo_gui import buffer, episodes, jobs
from apps.demo_gui.backends.mock import _next
from apps.demo_gui.config import REPO, Cfg
from apps.demo_gui.session import read_index
from task_common.magna_process import CONTROLLER_REL

# v12_147/e2e_demo.sh with the e2etune set3_gv_07 stage-1 yaml + walk (a v12_148 tier demo)
DEMO_PARAMS = REPO / "data/lcs/realfirst/cmp/v12_147/e2etune/yaml/set3_gv_07.yaml"
DEMO_SCRIPTED_SET = ["hover_h_mm=36"]
STOP_WAIT_S = 120.0  # teleop Stop: release + save after the final d

# v12_147/run11.sh MILES settings; --wp-motion-mm sets the waypoints, so --wp-stride is ignored
MILES_FLAGS = shlex.split(
    "--wp-motion-mm 5 --wp-hold 4 --dev-s 0.8 --ret-s 0.8 --hold-s 0.2 --box-free 30,8 "
    "--box-contact 10,4 --emphasis-frac 0 --modes both --seed 20261123 --tau-mm 1.82 "
    "--belt-ref arrival --reach-mm 4 --reach-deg 2 --on-unreachable skip-waypoint "
    "--on-disturb skip-waypoint --franka-grip-n 30")

LCS = Path("/home/hienbui/git/lcs_learning")
# data/lcs/realfirst/cmp/v12_152/train.sh variant q (latent 32), minus buffer/out/tag/epochs/seed
TRAIN_FLAGS = shlex.split(
    "--init-checkpoint none --mode joint --lr 0.001 --latent-dim 32 --num-points 544 "
    "--pc-recon-loss chamfer --encoder-inputs seg_ctx --belt-ctx-dim 16 --belt-ctx-keep pose "
    "--belt-grip-gate --recon-max-mm 6.0 --batch-size 154 --aux-feature-weight 0 "
    "--n-dyn-heads 2 --mode-gated --mode-gate-mats A,B,d --mode-delta-l2 1e-5 "
    "--mode-delta-init like_base --device cuda --train-only-kinds s1_chain,stage2_mpc,stage2_demo "
    "--heldout-frac 0.05 --no-violation-grad-to-encoder")


def train_argv(session, tag: str, epochs: int, seed: int) -> list[str]:
    return [str(LCS / ".venv/bin/python"), "scripts/refit_from_buffer.py",
            "--buffer", str(buffer.path(session, tag)), *TRAIN_FLAGS, "--epochs", str(int(epochs)),
            "--seed", str(int(seed)), "--out-dir", str(session.train), "--tag", tag]


def make_start_argv(cfg: Cfg, session, sid: str, franka: dict, ur: dict) -> list[str]:
    return ["uv", "run", "--frozen", "python", "apps/demo_gui/backend/make_start.py",
            "--out", str(session.starts), "--id", sid,
            "--franka-slide-mm", str(float(franka["slide_mm"])),
            "--franka-roll-deg", str(float(franka["roll_deg"])),
            "--ur-slide-mm", str(float(ur["slide_mm"])), "--ur-roll-deg", str(float(ur["roll_deg"])),
            "--franka-pivot-deg", str(float(franka.get("pivot_deg", 0.0))),
            "--ur-pivot-deg", str(float(ur.get("pivot_deg", 0.0))),
            "--franka-depth-mm", str(float(franka.get("depth_mm", 0.0))),
            "--ur-depth-mm", str(float(ur.get("depth_mm", 0.0))),
            "--lcm-url", cfg.lcm_url]


def demo_argv(cfg: Cfg, session, start_id: str, out: Path, mode: str = "scripted") -> list[str]:
    """The ``e2e_demo.sh`` harness line on the app's port, publishing body poses; ``teleop``: both
    stages by SpaceMouse (UR exit scripted), keys from ``session.cmd_path``."""
    rel = os.path.relpath(DEMO_PARAMS, cfg.magna_root)
    s2 = (["--stage2-scripted", *(a for kv in DEMO_SCRIPTED_SET for a in ("--scripted-set", kv))]
          if mode == "scripted" else
          ["--teleop-stage1", "--stage2-teleop", "--teleop-cmd-file", str(session.cmd_path)])
    return ["uv", "run", "--frozen", "python", "scripts/lcs/eval_learned_mpc.py",
            "--mode", "baseline", "--magna-root", str(cfg.magna_root), "--params", rel,
            "--lcm-url", cfg.lcm_url, "--record",
            "--max-episode-s", "15", "--settle-s", "0.5", "--pace", "1",
            "--start-states", str(session.starts / f"{start_id}_osc.npz"), "--repeats", "1",
            "--belt-seg", "--cloud-noise", "off", "--label", f"demo_gui_{out.name}",
            "--out", str(out), "--outcome-pulley", "both", "--post-mpc", "stage2_release",
            "--stage2-chain", *s2, "--stage2-start-snapshot", "--stage2-max-episode-s", "30",
            "--franka-width-max", "20", "--end-snapshot", "--ur-exit-final-settle-s", "3", "--log-plan-age",
            "--franka-grip-n", "30", "--render-every", str(cfg.render_every),
            "--publish-body-poses", "--realtime"]


def telemetry_argv(cfg: Cfg, session, run_dir: Path, until_pid: int | None) -> list[str]:
    return ["uv", "run", "--frozen", "python", "-m", "apps.demo_gui.telemetry_lcm",
            "--url", cfg.lcm_url, "--out", str(session.telemetry_path), "--run-dir", str(run_dir),
            *(["--until-pid", str(until_pid)] if until_pid else [])]


class SimBackend:
    supports_teleop = True

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self._demo: tuple[Path, jobs.Job, jobs.Job] | None = None
        self._teleop = self._stage2 = False

    def argv_for(self, op: str, **kw) -> list[str]:
        """argv of ``make_start`` / ``record_demo`` / ``telemetry`` without running anything."""
        fn = {"make_start": make_start_argv, "record_demo": demo_argv,
              "telemetry": telemetry_argv}[op]
        return fn(self.cfg, **kw)

    def make_start(self, session, franka, ur):
        p = session.jobs / "make_start.json"
        if p.is_file() and jobs.status(jobs.load(p)) == "running":
            raise RuntimeError("a make_start job is still running")
        sid = f"s{len(read_index(session.starts)):02d}"
        return jobs.start(session, "make_start", self.argv_for(
            "make_start", session=session, sid=sid, franka=franka, ur=ur), sim=True,
            lcm_url=self.cfg.lcm_url)

    def record_demo(self, session, start_id, mode="scripted"):
        if mode not in ("scripted", "teleop"):
            raise ValueError(f"demo mode {mode!r}: scripted or teleop")
        start = session.starts / f"{start_id}_osc.npz"
        if not start.is_file():
            raise FileNotFoundError(f"start {start_id} has no held state ({start})")
        ctrl = self.cfg.magna_root / CONTROLLER_REL
        if mode == "scripted" and not ctrl.is_file():  # a teleop stage 1 runs no controller
            raise FileNotFoundError(f"baseline controller {ctrl} missing (the scripted demo runs "
                                    "magna's waypoint baseline; build it or pass --magna-root)")
        self._kill()
        session.cmd_path.write_text("")  # no stale key from an earlier run
        self._teleop, self._stage2 = mode == "teleop", False
        d = _next(session.demos, "demo_")
        d.mkdir()  # claims the name while the job waits for the sim lock
        job = jobs.start(session, "demo", self.argv_for(
            "record_demo", session=session, start_id=start_id, out=d, mode=mode),
            env={"ROUND_BELT_SCENE": str(self.cfg.scene)}, sim=True, lcm_url=self.cfg.lcm_url)
        tel = jobs.start(session, "telemetry", self.argv_for(
            "telemetry", session=session, run_dir=d, until_pid=job.pid),
            env={"CUDA_VISIBLE_DEVICES": ""})
        self._demo = (d, job, tel)
        return job

    def teleop_key(self, session, ch: str) -> None:
        """One teleop key (``d`` done, ``a`` abort, ``p`` pause) for the harness to pop."""
        session.cmd_path.write_text(ch)
        self._stage2 |= self._teleop and ch == "d"

    def _kill(self) -> None:
        if self._demo:
            for job in self._demo[1:]:
                jobs.stop(job)

    def stop_demo(self, session):
        """Teleop after "Done stage 1": ``d`` (release + save), wait for the job; then kill."""
        if self._demo and self._stage2:
            self.teleop_key(session, "d")
            end = time.monotonic() + STOP_WAIT_S
            while jobs.status(self._demo[1]) == "running" and time.monotonic() < end:
                time.sleep(0.5)
        self._kill()

    def reset_demo(self, session):
        if self._demo:
            if self._teleop:
                self.teleop_key(session, "a")
            self._kill()
            shutil.rmtree(self._demo[0], ignore_errors=True)
            self._demo = None

    def run_miles(self, session, demos, cycles: int = 4, wp_stride: int = 5):
        """MILES around ``demos`` (``{name, episode, start}``); whole-run demos as stage-1 cuts."""
        d = _next(session.miles, "miles_")
        argv = ["uv", "run", "--frozen", "python", "scripts/lcs/miles_seed.py", "--out", d,
                "--lcm-url", self.cfg.lcm_url,
                # an --extra-demo only registers a name; miles_seed runs the --demos list
                "--demos", ",".join(x["name"] for x in demos)]
        for x in demos:
            ep, marks = episodes.stage1_input(x["episode"], d.with_name(d.name + "_in"), x["name"])
            argv += ["--extra-demo", f"{x['name']}={ep},{x['start']}" + (f",{marks}" if marks else "")]
        argv += ["--cycles", cycles, "--wp-stride", wp_stride, "--record", *MILES_FLAGS]
        return jobs.start(session, d.name, argv, env={"ROUND_BELT_SCENE": str(self.cfg.scene)},
                          sim=True, lcm_url=self.cfg.lcm_url)

    def train(self, session, tag, rows, epochs, seed: int = 5):
        """v12_152 train.sh variant q on ``buffer.path(session, tag)`` (built by ``buffer.build``;
        ``rows`` unused) -> ``<s>/train/<tag>``."""
        buf = buffer.path(session, tag)
        if not buf.is_file():
            raise ValueError(f"no training buffer {buf}")
        if (session.train / tag).exists():
            raise ValueError(f"{session.train / tag} exists; pick a new tag")
        return jobs.start(session, f"train_{tag}", train_argv(session, tag, epochs, seed), cwd=LCS,
                          env={"PYTHONPATH": str(LCS)}, gpu=True)

    def export(self, session, tag):
        return jobs.start(session, f"export_{tag}", [
            "bash", REPO / "apps/demo_gui/backend/post_export.sh", session.root, tag], gpu=True)

    def evaluate(self, session, name, model, starts, repeats):
        """``run_hard_suite.py write`` (offline, waited for), then ``run.sh`` + summarize as a sim
        job on the app's port."""
        import time

        from apps.demo_gui import evalrun, jobs

        out = evalrun.new_out(session, name)
        w = jobs.start(session, f"evalwrite_{name}",
                       evalrun.write_argv(self.cfg, out, name, model, repeats, starts))
        end = time.monotonic() + 120
        while jobs.status(w) == "running" and time.monotonic() < end:
            time.sleep(0.2)
        if jobs.status(w) == "running":
            jobs.stop(w)
        if not (out / "run.sh").is_file():
            raise RuntimeError(f"run_hard_suite write failed, see {w.log}")
        return jobs.start(session, f"eval_{name}", evalrun.run_argv(out), sim=True,
                          lcm_url=self.cfg.lcm_url)
