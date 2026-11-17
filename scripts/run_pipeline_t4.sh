#!/usr/bin/env bash
# The T4 pipeline, stage by stage, from bevaligner/configs/pipeline_t4.yaml.
#
#   scripts/run_pipeline_t4.sh                    splits, preprocess, train, eval
#   scripts/run_pipeline_t4.sh preprocess         one stage on its own
#   scripts/run_pipeline_t4.sh resume             continue an interrupted run
#   scripts/run_pipeline_t4.sh train --dry-run    print the call, do nothing
#   scripts/run_pipeline_t4.sh eval eval.qualitative=0
#
# Anything after the stage is handed to the pipeline as-is, so dotted overrides
# such as preprocess.workers=8 work here exactly as they do on the module.
#
# CONFIG= and PYTHON= override the config file and the interpreter.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
CONFIG="${CONFIG:-bevaligner/configs/pipeline_t4.yaml}"
export HF_HOME="${HF_HOME:-$REPO/cache/hf}"

# The stages this config lists, in order. `smoke` and `resume` exist too, but
# are run by name rather than as part of a full pass.
STAGES=(splits preprocess train eval)

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
