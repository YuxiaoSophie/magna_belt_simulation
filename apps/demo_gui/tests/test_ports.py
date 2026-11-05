"""LCM port policy and the 7667 literal guard; plain asserts, pytest-compatible.

    uv run --frozen python apps/demo_gui/tests/test_ports.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.demo_gui import config

REPO = config.REPO
APP = REPO / "apps/demo_gui"


def test_lcm_url_ranges():
    assert config.lcm_url(7850) == "udpm://239.255.76.250:7850?ttl=0"
    assert config.lcm_url(7855) == "udpm://239.255.76.255:7855?ttl=0"
    assert config.lcm_url(7856) == "udpm://239.255.78.56:7856?ttl=0"
    assert config.lcm_url(7859) == "udpm://239.255.78.59:7859?ttl=0"
    for bad in (7667, 7779, 7860):
        try:
            config.lcm_url(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


def test_launcher_refuses():
    assert set(config.REFUSED_SIM_PORTS) == {7667, 7840, 7841, 7855, 7857}
    for port in config.REFUSED_SIM_PORTS:
        r = subprocess.run([sys.executable, REPO / "scripts/demo_gui.py", "--sim-port", str(port)],
                           capture_output=True, text=True, timeout=60, check=False)
        assert r.returncode == 2 and "refused" in r.stderr, (port, r.stderr)  # argparse exit, no server


def test_viewer_hz():
    assert [config.Cfg(viewer_hz=hz).render_every for hz in (60, 30, 200, 1000)] == [3, 7, 1, 1]
    r = subprocess.run([sys.executable, REPO / "scripts/demo_gui.py", "--help"],
                       capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0 and "--viewer-hz" in r.stdout, r.stderr
    r = subprocess.run([sys.executable, REPO / "scripts/demo_gui.py", "--viewer-hz", "0"],
                       capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 2 and "--viewer-hz" in r.stderr, r.stderr


def test_no_7667_literal():
    hits = []
    for p in sorted(APP.rglob("*.py")):
        if p.name == "test_ports.py" or "__pycache__" in p.parts:
            continue
        hits += [f"{p.relative_to(REPO)}:{i}" for i, ln in enumerate(p.read_text().splitlines(), 1)
                 if "7667" in ln]
    allowed = ("apps/demo_gui/config.py:", "apps/demo_gui/tests/")
    bad = [h for h in hits if not h.startswith(allowed)]  # config refusals + tests asserting absence
    assert not bad, bad


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
