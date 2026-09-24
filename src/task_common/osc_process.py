"""magna's ``franka_cartesian_osc_controller`` as a child process on a PRIVATE LCM URL.

The binary resolves ``systems/parameters/*.yaml`` relative to its CWD, so it runs from the magna
root. It reads ``FRANKA_STATE`` + ``TARGET_CARTESIAN_POSE_TRAJECTORY`` and publishes one
``FRANKA_INPUT`` per state message. Side effect of the binary: it writes its diagram to
``<magna root>/../diagrams/franka_cartesian_osc_controller{,.svg}``.
"""

from __future__ import annotations

from pathlib import Path

from task_common.magna_process import (  # noqa: F401  (re-exported)
    SHARED_LCM_MARKERS,
    MagnaProcess,
    check_private_url,
    sha256,
)

MAGNA_ROOT = Path("/home/hienbui/git/magna")
OSC_PARAMS_REL = Path("systems") / "parameters" / "franka_cartesian_osc_controller_params.yaml"
OSC_BINARY = (MAGNA_ROOT / "bazel-bin" / "systems" / "controllers"
              / "franka_cartesian_osc_controller")
OSC_PARAMS_YAML = MAGNA_ROOT / OSC_PARAMS_REL
DEFAULT_ARGS = ("--input_mode=1", "--osc_debug_level=warn")


class OscProcess(MagnaProcess):
    """One OSC child process: ``start`` / ``alive`` / ``stop`` (SIGINT, then SIGKILL)."""

    def start(self, lcm_url: str, log_path: Path, cwd: Path = MAGNA_ROOT,
              binary: Path = OSC_BINARY, extra_args=DEFAULT_ARGS) -> int:
        url = check_private_url(lcm_url)
        return self.launch(binary, [f"--lcm_url={url}", *extra_args], cwd, log_path, url,
                           "bazel build //systems/controllers:franka_cartesian_osc_controller")

    def describe(self) -> dict:
        info = super().describe()
        params = OSC_PARAMS_YAML if self.cwd is None else Path(self.cwd) / OSC_PARAMS_REL
        info.update(params_yaml=str(params), params_sha256=sha256(params))
        return info
