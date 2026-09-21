#!/usr/bin/env bash
# Regenerate the dairlib/drake/robotiq/magna Python packages in place, next to their
# lcmtypes/*/*.lcm sources. Run from anywhere.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# Clean regen (stale modules from a removed .lcm would linger); the .lcm sources share these dirs.
for pkg in dairlib drake robotiq magna; do
  rm -f "$pkg"/*.py
  rm -rf "$pkg/__pycache__"
done
uv run lcm-gen -p --ppath . */*.lcm
