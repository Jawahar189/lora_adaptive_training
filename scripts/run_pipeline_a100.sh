#!/usr/bin/env bash
# The A100 pipeline, stage by stage, from bevaligner/configs/pipeline_a100.yaml.
#
#   scripts/run_pipeline_a100.sh                    splits, preprocess, smoke, train, eval
#   scripts/run_pipeline_a100.sh smoke              one stage on its own
#   scripts/run_pipeline_a100.sh resume             continue an interrupted run
#   scripts/run_pipeline_a100.sh train --dry-run    print the call, do nothing
#   scripts/run_pipeline_a100.sh eval eval.qualitative=0
#
# Anything after the stage is handed to the pipeline as-is, so dotted overrides
# such as preprocess.workers=8 work here exactly as they do on the module.
#
# CONFIG= and PYTHON= override the config file and the interpreter. The default
# interpreter is .venv-a100, not .venv: the repo sits on a shared disk, and the
# .venv built on the T4 box is Python 3.10, which this machine does not have.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv-a100/bin/python}"
CONFIG="${CONFIG:-bevaligner/configs/pipeline_a100.yaml}"
export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
# the token `hf auth login` saved, which a repo-local HF_HOME would not see;
# DINOv3 is gated and needs it
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"

if ! "$PYTHON" -c "import torch, lightning, transformers" 2>/dev/null; then
    echo "$PYTHON cannot import torch, lightning and transformers. Build it with" >&2
    echo "  python3 -m venv .venv-a100 && .venv-a100/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# The stages this config lists, in order. `resume` exists too, but is run by
# name rather than as part of a full pass.
STAGES=(splits preprocess smoke train eval)

run_stage() {
    local stage="$1"
    shift
    echo
    echo "=== $stage  $(date '+%Y-%m-%d %H:%M:%S')"
    "$PYTHON" -m bevaligner.pipeline "$stage" --config "$CONFIG" "$@"
}

# A first argument that is neither a flag nor a key=value override is the stage.
if [ $# -gt 0 ] && [[ "$1" != -* && "$1" != *=* ]]; then
    stage="$1"
    shift
    run_stage "$stage" "$@"
else
    for stage in "${STAGES[@]}"; do
        run_stage "$stage" "$@"
    done
fi
