#!/usr/bin/env bash
cd "$(dirname "$0")/.." && exec uv run python scripts/round_belt_lcm_simulation.py "$@"
