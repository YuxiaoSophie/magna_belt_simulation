"""magna binaries as child processes on a PRIVATE LCM URL (own session, log file, own-pid stop).

:class:`MagnaProcess` is the generic launcher; :class:`AssemblyControllerProcess` runs
``run_round_belt_assembly_controller`` (from the ``magna-deploy-learned-lcs`` worktree: only it
has ``--local_lcm_url``) and reads its stdout markers. magna binaries resolve
``systems/parameters/*.yaml`` relative to their CWD, so each runs from its magna root.
:class:`MppiControllerProcess` runs ``scripts/lcs/mppi_controller_node.py`` (same markers) with
the same log API.
"""

from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

MAGNA_WORKTREE = Path("/home/hienbui/git/magna-deploy-learned-lcs")
CONTROLLER_REL = Path("bazel-bin/systems/controllers/run_round_belt_assembly_controller")
REPO_ROOT = Path(__file__).resolve().parents[2]
MPPI_NODE = REPO_ROOT / "scripts" / "lcs" / "mppi_controller_node.py"
SHARED_LCM_MARKERS = ("239.255.76.67", "7667")
CONTROLLER_STARTED = "Assembly controller started"
# Substrings of the controller's stdout lines that mark phase/stage transitions.
CONTROLLER_MARKERS = (
    CONTROLLER_STARTED,
    "No pre-MPC targets found. Initial phase set to MPC.",
    "Pre-MPC target ",
    "All pre-MPC targets completed!",
    "[learned-mpc] stage ",
    "MPC completed!",
    "Switching to Terminate",
)
_STAGE_LINE = re.compile(r"\[learned-mpc\] stage (\d+) (reached|timeout) t=(\S+) dist=(\S+)")
_PRE_MPC_LINE = re.compile(r"Pre-MPC target (\d+) reached at t=(\S+)")


def check_private_url(lcm_url: str) -> str:
    """``lcm_url`` if it is a non-empty URL other than magna's shared group, else ValueError."""
    url = str(lcm_url or "").strip()
    if not url:
        raise ValueError("an explicit private --lcm_url is required")
    if any(marker in url for marker in SHARED_LCM_MARKERS):
        raise ValueError(f"{url!r} is magna's shared LCM group; use a private one")
    return url


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MagnaProcess:
    """One child process: ``launch`` / ``alive`` / ``stop`` (SIGINT, then SIGKILL, own pid)."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.argv: list[str] = []
        self.cwd: Path | None = None
        self.lcm_url: str | None = None
        self.binary: Path | None = None
        self.log_path: Path | None = None
        self._log = None

    def launch(self, binary: Path, args, cwd: Path, log_path: Path, lcm_url: str,
               build_hint: str = "") -> int:
        """Start ``binary args...`` in ``cwd``; stdout+stderr appended to ``log_path``."""
        if self.alive():
            raise RuntimeError(f"{self.binary} already running (pid {self.pid})")
        self.lcm_url = check_private_url(lcm_url)
        self.binary, self.cwd = Path(binary), Path(cwd)
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary}{f' ({build_hint})' if build_hint else ''}")
        self.argv = [str(self.binary), *[str(a) for a in args]]
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")  # noqa: SIM115 - held by the child until stop()
        self.proc = subprocess.Popen(self.argv, cwd=self.cwd, stdout=self._log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        return self.proc.pid

    @property
    def pid(self) -> int | None:
        return None if self.proc is None else self.proc.pid

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace_s: float = 3.0) -> int | None:
        """Stop our own child by pid; returns its exit code (None if never started)."""
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            try:
                os.kill(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                os.kill(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=grace_s)
            except ProcessLookupError:
                pass
        code = self.proc.wait()
        if self._log is not None:
            self._log.close()
            self._log = None
        return code

    def log_text(self) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        return self.log_path.read_text(errors="replace")

    def describe(self) -> dict:
        resolved = None if self.binary is None else self.binary.resolve()
        return {
            "binary": None if self.binary is None else str(self.binary),
            "binary_resolved": None if resolved is None else str(resolved),
            "binary_sha256": None if resolved is None else sha256(resolved),
            "argv": list(self.argv), "cwd": None if self.cwd is None else str(self.cwd),
            "lcm_url": self.lcm_url, "pid": self.pid,
            "log": None if self.log_path is None else str(self.log_path),
        }


class _ControllerLog(MagnaProcess):
    """Incremental reading of this run's part of an append-mode controller log."""

    def __init__(self) -> None:
        super().__init__()
        self._log_offset = 0
        self._read_offset = 0
        self._partial = b""

    def _mark_log(self, log_path: Path) -> None:
        self._log_offset = Path(log_path).stat().st_size if Path(log_path).exists() else 0
        self._read_offset, self._partial = self._log_offset, b""

    def own_log(self) -> str:
        """This run's part of the (append-mode) log."""
        text = self.log_text()
        return text[self._log_offset:] if len(text) >= self._log_offset else text

    def read_new_lines(self) -> list[str]:
        """Complete log lines written since the last call (incremental, cheap per step)."""
        if self.log_path is None or not self.log_path.exists():
            return []
        with open(self.log_path, "rb") as f:
            f.seek(self._read_offset)
            data = f.read()
        self._read_offset += len(data)
        data = self._partial + data
        *lines, self._partial = data.split(b"\n")
        return [line.decode(errors="replace") for line in lines]

    def wait_for(self, substring: str, timeout_s: float) -> float:
        """Wall seconds until ``substring`` is in this run's log; RuntimeError on exit/timeout."""
        t0 = time.perf_counter()
        while True:
            if substring in self.own_log():
                return time.perf_counter() - t0
            if not self.alive():
                raise RuntimeError(f"controller exited ({self.proc.returncode}) before "
                                   f"{substring!r}; log {self.log_path}")
            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError(f"no {substring!r} after {timeout_s:g} s; log {self.log_path}")
            time.sleep(0.02)


class AssemblyControllerProcess(_ControllerLog):
    """magna's round-belt assembly controller with both LCM URLs set to one private URL."""

    def __init__(self) -> None:
        super().__init__()
        self.params_yaml: Path | None = None

    def start(self, lcm_url: str, params_yaml, log_path: Path, magna_root: Path = MAGNA_WORKTREE,
              extra_args=(), binary: Path | None = None) -> int:
        """``params_yaml`` is relative to ``magna_root`` (the controller's CWD)."""
        root = Path(magna_root)
        rel = Path(params_yaml)
        if rel.is_absolute():
            rel = rel.relative_to(root)
        if not (root / rel).is_file():
            raise FileNotFoundError(root / rel)
        self.params_yaml = rel
        url = check_private_url(lcm_url)
        args = [f"--lcm_url={url}", f"--local_lcm_url={url}",
                f"--assembly_controller_params={rel}", *extra_args]
        self._mark_log(log_path)
        return self.launch(root / CONTROLLER_REL if binary is None else binary, args, root,
                           log_path, url, "bazel build //systems/controllers:"
                           "run_round_belt_assembly_controller in the worktree")

    def describe(self) -> dict:
        info = super().describe()
        params = (None if self.cwd is None or self.params_yaml is None
                  else self.cwd / self.params_yaml)
        info.update(params_yaml=None if params is None else str(params),
                    params_sha256=None if params is None else sha256(params))
        return info


class MppiControllerProcess(_ControllerLog):
    """``mppi_controller_node.py`` (this repo's venv python, CWD = repo) on one private URL."""

    def __init__(self) -> None:
        super().__init__()
        self.node: Path = MPPI_NODE
        self.config: Path | None = None
        self.deploy: Path | None = None

    def start(self, lcm_url: str, deploy, config, log_path: Path, goal: str = "deploy",
              stages: str = "last", max_duration_s: float = 30.0, extra_args=()) -> int:
        """``goal``: ``deploy`` or ``demo_goals:PATH`` (the node's ``--goal``)."""
        self.deploy, self.config = Path(deploy).resolve(), Path(config).resolve()
        for path in (self.node, self.deploy, self.config):
            if not path.is_file():
                raise FileNotFoundError(path)
        url = check_private_url(lcm_url)
        args = [str(self.node), "--lcm-url", url, "--deploy", str(self.deploy), "--config",
                str(self.config), "--goal", goal, "--stages", stages, "--max-duration-s",
                f"{float(max_duration_s):g}", *extra_args]
        self._mark_log(log_path)
        return self.launch(Path(sys.executable), args, REPO_ROOT, log_path, url)

    def describe(self) -> dict:
        info = super().describe()
        info.update(node=str(self.node), node_sha256=sha256(self.node),
                    config=None if self.config is None else str(self.config),
                    config_sha256=None if self.config is None else sha256(self.config),
                    deploy=None if self.deploy is None else str(self.deploy),
                    deploy_sha256=None if self.deploy is None else sha256(self.deploy))
        return info


def marker_lines(text: str) -> list[tuple[int, str]]:
    return [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1)
            if any(m in line for m in CONTROLLER_MARKERS)]


def parse_stage_line(line: str) -> tuple[int, str, float, float] | None:
    """``(stage, "reached"|"timeout", t, dist)`` of a ``[learned-mpc] stage`` marker."""
    m = _STAGE_LINE.search(line)
    if m is None:
        return None
    return int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))


def parse_pre_mpc_line(line: str) -> tuple[int, float] | None:
    """``(target index, t)`` of a ``Pre-MPC target <i> reached at t=`` marker."""
    m = _PRE_MPC_LINE.search(line)
    return None if m is None else (int(m.group(1)), float(m.group(2)))
