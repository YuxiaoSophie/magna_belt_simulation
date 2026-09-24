#!/usr/bin/env bash
# Default root = the deploy worktree (only its controller has --local_lcm_url).
export LCM_DEFAULT_URL="${LEARNED_LCM_URL:-udpm://239.255.76.95:7695?ttl=0}"
cd "${MAGNA_ROOT:-/home/hienbui/git/magna-deploy-learned-lcs}" && exec "$@"
