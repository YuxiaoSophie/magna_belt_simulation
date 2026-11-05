"""Run every test_*.py as a plain script (no pytest in the venv); exit 1 on any failure.

    uv run --frozen python apps/demo_gui/tests/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
failed = []
for f in sorted(here.glob("test_*.py")):
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, f], capture_output=True, text=True, check=False)
    print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {f.name} ({time.monotonic() - t0:.1f} s)")
    if r.returncode:
        failed.append(f.name)
        print((r.stdout + r.stderr)[-2000:])
sys.exit(1 if failed else 0)
