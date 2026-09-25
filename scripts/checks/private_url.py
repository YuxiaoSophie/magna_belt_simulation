"""Port + leftover-process helpers for checks that take a private ``--lcm-url``."""

from __future__ import annotations

import os
import subprocess
from urllib.parse import urlparse


def url_port(url: str) -> str:
    """``udpm://239.255.76.99:7699?ttl=0`` -> ``"7699"``."""
    port = urlparse(url).port
    if port is None:
        raise ValueError(f"{url}: no port")
    return str(port)


def _ancestors(pid: int) -> set[int]:
    out = set()
    while pid > 1:
        out.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def pgrep_others(pattern: str) -> list[int]:
    """``pgrep -f pattern`` minus this process and its ancestors (their argv holds the URL)."""
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True,
                         check=False).stdout.split()
    mine = _ancestors(os.getpid())
    return sorted(int(p) for p in out if p.isdigit() and int(p) not in mine)
