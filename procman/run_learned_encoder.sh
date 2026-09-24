#!/usr/bin/env bash
cd "$(dirname "$0")/.." && exec uv run python scripts/lcs/latent_encoder_node.py "$@"
