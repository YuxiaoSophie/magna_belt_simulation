"""Serve a ``.viser`` recording (e.g. ``bike_chain_spike.py --record-viser``) with viser's
timeline/scrub player: open ``http://localhost:<port>/?playbackPath=recording.viser``.

uv run --frozen python scripts/debug/play_viser.py data/bike_chain/v13_02b/view/seated.viser
"""

from __future__ import annotations

import argparse
import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from newton.viewer import ViewerViser


class Handler(SimpleHTTPRequestHandler):
    """viser's client build, plus the recording at ``/recording.viser``."""

    recording = b""

    def do_GET(self) -> None:
        if self.path.split("?")[0] != "/recording.viser":
            super().do_GET()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(self.recording)))
        self.end_headers()
        self.wfile.write(self.recording)

    def log_message(self, *args) -> None:
        pass


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        # Browsers drop connections mid-download (reload, cancelled fetch): not an error.
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help=".viser recording")
    ap.add_argument("--port", type=int, default=8084)
    args = ap.parse_args()
    Handler.recording = args.path.read_bytes()
    handler = functools.partial(Handler, directory=str(ViewerViser.get_viser_client_dir()))
    with Server(("", args.port), handler) as server:
        print(f"http://localhost:{args.port}/?playbackPath=recording.viser  "
              f"({args.path.name}; Ctrl-C to quit)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
