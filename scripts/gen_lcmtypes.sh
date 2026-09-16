#!/usr/bin/env bash
# Regenerate the dairlib/drake/robotiq Python packages from lcmtypes/*/*.lcm. Run from repo root.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Clean regen: stale modules from a removed .lcm source would otherwise linger.
for pkg in dairlib drake robotiq; do
  [[ -d "$pkg" && -f "$pkg/__init__.py" ]] && rm -r "$pkg"
done
uv run lcm-gen -p --ppath . lcmtypes/*/*.lcm
