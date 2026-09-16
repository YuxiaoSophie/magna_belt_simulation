#!/usr/bin/env bash
# Procman does not cd: the magna binaries resolve systems/parameters/... from the magna root.
cd "${MAGNA_ROOT:-/home/hienbui/git/magna}" && exec "$@"
