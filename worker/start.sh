#!/bin/sh
set -eu

MODEL_REPOSITORY="${MODEL_REPOSITORY:-Jerome0207/penclaw-glm-5.3-abliterated-q4-k-m}"
MODEL_CACHE="${MODEL_CACHE:-/runpod-volume/huggingface-cache/hub}"
MODEL_ALIAS="${MODEL_ALIAS:-penclaw-glm-5.3-abliterated-q4-k-m}"
CONTEXT_SIZE="${CONTEXT_SIZE:-131072}"
GPU_COUNT="${GPU_COUNT:-4}"
SPLIT_MODE="${SPLIT_MODE:-layer}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1,1,1,1}"

log() {
    printf '[penclaw-glm53] %s\n' "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[ "$visible" = "$GPU_COUNT" ] || die "Expected ${GPU_COUNT} GPUs, found ${visible}"

namespace="${MODEL_REPOSITORY%%/*}"
repository="${MODEL_REPOSITORY#*/}"
snapshot_root="${MODEL_CACHE}/models--${namespace}--${repository}/snapshots"

model_path="${MODEL_PATH:-}"
if [ -z "$model_path" ]; then
    for candidate in "$snapshot_root"/*/*-00001-of-*.gguf; do
        if [ -r "$candidate" ]; then
            model_path="$candidate"
            break
        fi
    done
fi

[ -n "$model_path" ] || die "No first GGUF shard found in RunPod Cached Models"
[ -r "$model_path" ] || die "Model is not readable: ${model_path}"

log "Model: ${model_path}"
log "Context: ${CONTEXT_SIZE}; split mode: ${SPLIT_MODE}; tensor split: ${TENSOR_SPLIT}"
nvidia-smi topo -m >&2 || true

exec /opt/runpod/health-proxy \
    --model "$model_path" \
    --alias "$MODEL_ALIAS" \
    --ctx-size "$CONTEXT_SIZE" \
    --n-gpu-layers 999 \
    --split-mode "$SPLIT_MODE" \
    --tensor-split "$TENSOR_SPLIT" \
    --main-gpu 0 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --parallel 1 \
    --cont-batching \
    --jinja \
    --metrics \
    --no-webui \
    --temp 1.0 \
    --top-p 0.95 \
    --n-predict 32768
