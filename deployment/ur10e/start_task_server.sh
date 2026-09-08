#!/usr/bin/env bash
set -euo pipefail

# Start the 5090 FastWAM server with a matched 80k checkpoint, normalization
# statistics file, and the exact instruction stored in the task dataset.

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly ARTIFACT_ROOT="${FASTWAM_ARTIFACT_ROOT:-/media/wbjsamuel/data/FastWAM-checkpoints/extend80-276515}"
readonly DEFAULT_VAE_PATH="/home/wbjsamuel/projects/FastWAM-ToolUse/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
readonly DEFAULT_TEXT_ENCODER_PATH="/home/wbjsamuel/projects/motionforesight/assets/models_pretrained/wan_models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
readonly DEFAULT_TOKENIZER_PATH="/home/wbjsamuel/projects/motionforesight/assets/models_pretrained/wan_models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

usage() {
    cat <<'EOF'
Usage:
  start_task_server.sh <task> [options]
  start_task_server.sh --list
  start_task_server.sh --check-all

Tasks: brush, cup, hammer, knife, screwdriver, spoon

Options:
  --port N                       TCP port (default: 9999)
  --device DEVICE                Torch device (default: cuda:0)
  --max-response-steps N         Confirmed-step response cap (default: 10)
  --max-stream-replan-steps N    Continuous-client queue cap (default: 10)
  --check-only                   Validate paths without loading the model
  -h, --help                     Show this help

Environment overrides:
  FASTWAM_ARTIFACT_ROOT, FASTWAM_VAE_PATH, FASTWAM_TEXT_ENCODER_PATH,
  FASTWAM_TOKENIZER_PATH, FASTWAM_PYTHON
EOF
}

task_prompt() {
    case "$1" in
        brush) echo "Use the brush to clean the target surface." ;;
        cup) echo "Pour the water from the cup into the kettle." ;;
        hammer) echo "Use the hammer to drive the nail into the target surface." ;;
        knife) echo "Use the knife to cut the target object." ;;
        screwdriver) echo "Use the screwdriver to tighten the screw." ;;
        spoon) echo "Use the spoon to transfer the food into the target container." ;;
        *) return 1 ;;
    esac
}

print_task() {
    local task="$1"
    printf '%-12s checkpoint=%s/%s/step_080000.pt\n' "$task" "$ARTIFACT_ROOT" "$task"
    printf '%-12s stats=%s/%s/%s-stats.json\n' "" "$ARTIFACT_ROOT" "$task" "$task"
    printf '%-12s prompt=%s\n' "" "$(task_prompt "$task")"
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        printf 'ERROR: %s file not found: %s\n' "$label" "$path" >&2
        return 1
    fi
}

require_dir() {
    local label="$1"
    local path="$2"
    if [[ ! -d "$path" ]]; then
        printf 'ERROR: %s directory not found: %s\n' "$label" "$path" >&2
        return 1
    fi
}

check_task_artifacts() {
    local task="$1"
    require_file "$task checkpoint" "$ARTIFACT_ROOT/$task/step_080000.pt"
    require_file "$task stats" "$ARTIFACT_ROOT/$task/$task-stats.json"
}

case ${1:-} in
    -h|--help)
        usage
        exit 0
        ;;
    --list)
        for listed_task in brush cup hammer knife screwdriver spoon; do
            print_task "$listed_task"
        done
        exit 0
        ;;
    --check-all)
        status=0
        for checked_task in brush cup hammer knife screwdriver spoon; do
            check_task_artifacts "$checked_task" || status=1
        done
        exit "$status"
        ;;
esac

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

task="$1"
shift
if ! prompt="$(task_prompt "$task")"; then
    printf 'ERROR: unknown task: %s\n' "$task" >&2
    usage >&2
    exit 2
fi

port=9999
device=cuda:0
max_response_steps=10
max_stream_replan_steps=10
check_only=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            port="${2:?--port requires a value}"
            shift 2
            ;;
        --device)
            device="${2:?--device requires a value}"
            shift 2
            ;;
        --max-response-steps)
            max_response_steps="${2:?--max-response-steps requires a value}"
            shift 2
            ;;
        --max-stream-replan-steps)
            max_stream_replan_steps="${2:?--max-stream-replan-steps requires a value}"
            shift 2
            ;;
        --check-only)
            check_only=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

checkpoint="$ARTIFACT_ROOT/$task/step_080000.pt"
stats="$ARTIFACT_ROOT/$task/$task-stats.json"
vae_path="${FASTWAM_VAE_PATH:-$DEFAULT_VAE_PATH}"
text_encoder_path="${FASTWAM_TEXT_ENCODER_PATH:-$DEFAULT_TEXT_ENCODER_PATH}"
tokenizer_path="${FASTWAM_TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}"

check_task_artifacts "$task"
require_file "VAE" "$vae_path"
require_file "text encoder" "$text_encoder_path"
require_dir "tokenizer" "$tokenizer_path"
require_file "FastWAM server" "$SCRIPT_DIR/fastwam_server.py"

printf 'FASTWAM_TASK_SERVER_CONFIG task=%s\n' "$task"
printf '  checkpoint=%s\n' "$checkpoint"
printf '  stats=%s\n' "$stats"
printf '  prompt=%s\n' "$prompt"
printf '  port=%s device=%s response_steps=%s stream_steps=%s\n' \
    "$port" "$device" "$max_response_steps" "$max_stream_replan_steps"

if [[ "$check_only" == true ]]; then
    echo "FASTWAM_TASK_SERVER_CHECK_OK task=$task"
    exit 0
fi

if [[ -n ${FASTWAM_PYTHON:-} ]]; then
    python_bin="$FASTWAM_PYTHON"
elif [[ -x /home/wbjsamuel/miniconda3/envs/fastwam/bin/python ]]; then
    python_bin=/home/wbjsamuel/miniconda3/envs/fastwam/bin/python
else
    python_bin=python
fi

cd "$REPO_ROOT"
exec "$python_bin" -u deployment/ur10e/fastwam_server.py \
    --mode inference-dry-run \
    --host 0.0.0.0 \
    --port "$port" \
    --checkpoint "$checkpoint" \
    --dataset-stats "$stats" \
    --vae-path "$vae_path" \
    --text-encoder-path "$text_encoder_path" \
    --tokenizer-path "$tokenizer_path" \
    --task-config ur_robotiq_uncond_1cam224 \
    --model-config fastwam_joint \
    --prompt "$prompt" \
    --max-response-steps "$max_response_steps" \
    --max-stream-replan-steps "$max_stream_replan_steps" \
    --device "$device"
