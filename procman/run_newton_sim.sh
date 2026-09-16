#!/usr/bin/env bash
cd "$(dirname "$0")/.." && exec uv run python round_belt_lcm_simulation.py "$@"
