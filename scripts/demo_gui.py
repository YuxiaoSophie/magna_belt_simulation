#!/usr/bin/env python3
"""Demo GUI launcher (apps/demo_gui): prints the URL, never opens a browser.

    uv run --frozen python scripts/demo_gui.py --backend mock
    uv run --frozen python scripts/demo_gui.py --backend sim --session data/demo_gui/sessions/<stamp>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.demo_gui import config
from apps.demo_gui.app import build_app
from apps.demo_gui.session import Session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backend", choices=("mock", "sim"), default="mock",
                    help="mock replays recorded data (no GPU / LCM)")
    ap.add_argument("--session", default="new",
                    help=f"session dir, or 'new' for {config.SESSION_ROOT}/<YYYYmmdd-HHMM>")
    ap.add_argument("--http-port", type=int, default=config.HTTP_PORT, help="GUI port")
    ap.add_argument("--sim-port", type=int, default=config.SIM_PORT, help="private LCM port")
    ap.add_argument("--viewer-port", type=int, default=config.VIEWER_PORT,
                    help="live Newton viewer port")
    ap.add_argument("--replay-port", type=int, default=config.REPLAY_PORT, help="replay viewer port")
    ap.add_argument("--viewer-hz", type=float, default=60.0,
                    help="live viewer render rate (rounded to a whole number of 200 Hz sim steps)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address")
    ap.add_argument("--magna-root", type=Path, default=config.MAGNA_ROOT,
                    help="magna checkout with the baseline controller + OSC builds (sim backend)")
    a = ap.parse_args()
    if a.sim_port in config.REFUSED_SIM_PORTS or a.sim_port < config.MIN_SIM_PORT:
        ap.error(f"--sim-port {a.sim_port} refused: 7667 is magna's shared port, < "
                 f"{config.MIN_SIM_PORT} is outside the private range, 7840/7841/7855/7857 are "
                 "taken (teleop / other runs)")
    try:
        config.lcm_url(a.sim_port)
    except ValueError as e:
        ap.error(str(e))
    if not a.viewer_hz > 0:
        ap.error(f"--viewer-hz {a.viewer_hz}: must be > 0")

    cfg = config.Cfg(backend=a.backend, host=a.host, http_port=a.http_port,
                     viewer_port=a.viewer_port, replay_port=a.replay_port, sim_port=a.sim_port,
                     magna_root=a.magna_root, viewer_hz=a.viewer_hz)
    session = Session.new() if a.session == "new" else Session(a.session)
    if a.backend == "mock":
        from apps.demo_gui.backends.mock import MockBackend
        backend = MockBackend()
    else:
        from apps.demo_gui.backends.sim import SimBackend
        backend = SimBackend(cfg)
    app = build_app(backend, session, cfg)
    print(f"session {session.root}\nopen http://{a.host}:{a.http_port}", flush=True)
    app.run(host=a.host, port=a.http_port, debug=False)


if __name__ == "__main__":
    main()
