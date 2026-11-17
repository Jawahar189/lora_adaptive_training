#!/usr/bin/env bash
# A checkout's one-time setup: the IMU submodule, the venv and its requirements.
#
#   ./setup.sh
#
# The IMU repo (external/imu_based_scale_resolution) is private. It is cloned with
# GIT_TOKEN from third_party/.env (git-ignored) when that has one, for this run only:
# the token goes to git through its environment, not its command line, and nothing is
# written to .git/config. Without one, your own git login is used.
#
# The venv is built with uv when it is installed, else with python -m venv and pip.
# PYTHON= the interpreter it is built from, VENV= where it goes, ENV_FILE= the .env.
# Re-running is safe: an existing venv is reused and only what is missing is installed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python3.10}"
VENV="${VENV:-$REPO/.venv}"
ENV_FILE="${ENV_FILE:-$REPO/third_party/.env}"

# the submodule first: it is the step that needs credentials, so it fails before the long install
token=""
if [ -f "$ENV_FILE" ]; then
    token="$(sed -n 's/^GIT_TOKEN=//p' "$ENV_FILE" | tail -1 | tr -d "\"'\r")"
fi
if [ -n "$token" ]; then
    echo "=== IMU submodule, with GIT_TOKEN from $ENV_FILE"
    GIT_TERMINAL_PROMPT=0 GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraHeader \
        GIT_CONFIG_VALUE_0="Authorization: Basic $(printf 'x-access-token:%s' "$token" | base64 | tr -d '\n')" \
        git submodule update --init
else
    echo "=== IMU submodule, with your git login (no GIT_TOKEN in $ENV_FILE)"
    git submodule update --init
fi
unset token

echo "=== venv $VENV"
if command -v uv >/dev/null; then
    # uv's venv has no pip of its own; unsafe-best-match lets the torch index serve torch only,
    # as pip does, instead of every package it also mirrors
    [ -x "$VENV/bin/python" ] || uv venv --python "$PYTHON" "$VENV"
    uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match -r requirements.txt
else
    [ -x "$VENV/bin/pip" ] || "$PYTHON" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r requirements.txt
fi

echo "=== done: $("$VENV/bin/python" --version), $(git submodule status external/imu_based_scale_resolution)"
