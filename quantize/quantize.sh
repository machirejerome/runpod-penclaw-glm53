#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_REPO="${SOURCE_REPO:-audnai/penclaw-GLM-5.3-abliterated}"
TARGET_REPO="${TARGET_REPO:-Jerome0207/penclaw-glm-5.3-abliterated-q4-k-m}"
LLAMA_COMMIT="${LLAMA_COMMIT:-81ff93ea1d48c0482508d30e75b814df99c25bf0}"
IMATRIX_REPO="${IMATRIX_REPO:-AesSedai/GLM-5.3-GGUF}"
WORK_ROOT="${WORK_ROOT:-/workspace/penclaw-quant}"
STATUS_ROOT="${STATUS_ROOT:-/status}"

SOURCE_DIR="${WORK_ROOT}/source"
F16_DIR="${WORK_ROOT}/bf16-gguf"
OUTPUT_DIR="${WORK_ROOT}/q4-k-m"
LLAMA_DIR="${WORK_ROOT}/llama.cpp"
IMATRIX_FILE="${WORK_ROOT}/imatrix.gguf"
LOG_FILE="${STATUS_ROOT}/log.txt"

mkdir -p "$STATUS_ROOT" "$WORK_ROOT" "$SOURCE_DIR" "$F16_DIR" "$OUTPUT_DIR"
touch "$LOG_FILE"

status() {
    local stage="$1"
    shift
    printf '%s\n' "$stage" > "${STATUS_ROOT}/stage.txt"
    printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_FILE"
}

fail() {
    local rc=$?
    printf 'failed\n' > "${STATUS_ROOT}/stage.txt"
    printf '[%s] FAILED rc=%s line=%s\n' "$(date -u +%FT%TZ)" "$rc" "${BASH_LINENO[0]:-unknown}" | tee -a "$LOG_FILE"
    printf '%s\n' '--- last log lines ---' >&2
    tail -n 120 "$LOG_FILE" >&2 || true
    exit "$rc"
}
trap fail ERR

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required" >&2
    exit 2
fi

export DEBIAN_FRONTEND=noninteractive
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

status bootstrap "Installing conversion and upload dependencies"
apt-get update >>"$LOG_FILE" 2>&1
apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git libcurl4-openssl-dev \
    python3 python3-pip python3-venv >>"$LOG_FILE" 2>&1

if [[ ! -x /opt/penclaw-venv/bin/python ]]; then
    python3 -m venv /opt/penclaw-venv
fi
/opt/penclaw-venv/bin/pip install -U pip 'huggingface_hub[cli,hf_xet]' >>"$LOG_FILE" 2>&1

status build "Building pinned llama.cpp conversion and quantization tools"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" >>"$LOG_FILE" 2>&1
fi
git -C "$LLAMA_DIR" checkout --detach "$LLAMA_COMMIT" >>"$LOG_FILE" 2>&1
/opt/penclaw-venv/bin/pip install -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt" >>"$LOG_FILE" 2>&1
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release >>"$LOG_FILE" 2>&1
cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" \
    --target llama-quantize llama-gguf-split >>"$LOG_FILE" 2>&1

status download "Downloading only the final root BF16 checkpoint (not research subdirectories)"
/opt/penclaw-venv/bin/hf download "$SOURCE_REPO" \
    --local-dir "$SOURCE_DIR" \
    --include 'model-*.safetensors' \
    --include 'model.safetensors.index.json' \
    --include 'config.json' \
    --include 'generation_config.json' \
    --include 'tokenizer.json' \
    --include 'tokenizer_config.json' \
    --include 'chat_template.jinja' \
    >>"$LOG_FILE" 2>&1

/opt/penclaw-venv/bin/hf download "$IMATRIX_REPO" imatrix.gguf \
    --local-dir "$WORK_ROOT/imatrix-source" >>"$LOG_FILE" 2>&1
cp "$WORK_ROOT/imatrix-source/imatrix.gguf" "$IMATRIX_FILE"

expected_shards=282
actual_shards="$(find "$SOURCE_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
if [[ "$actual_shards" != "$expected_shards" ]]; then
    echo "Expected ${expected_shards} BF16 shards, found ${actual_shards}" >&2
    exit 3
fi

status convert "Converting BF16 Safetensors to sharded BF16 GGUF"
/opt/penclaw-venv/bin/python "$LLAMA_DIR/convert_hf_to_gguf.py" "$SOURCE_DIR" \
    --outfile "$F16_DIR/Penclaw-GLM-5.3-Abliterated-BF16.gguf" \
    --outtype bf16 --split-max-size 45G >>"$LOG_FILE" 2>&1

first_f16="$(find "$F16_DIR" -maxdepth 1 -type f -name '*-00001-of-*.gguf' | sort | head -n 1)"
if [[ -z "$first_f16" ]]; then
    echo "Conversion produced no first GGUF shard" >&2
    exit 4
fi

status reclaim "Removing downloaded Safetensor shards after verified conversion"
find "$SOURCE_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' -delete

status quantize "Quantizing with transferred GLM-5.3 importance matrix to Q4_K_M"
"$LLAMA_DIR/build/bin/llama-quantize" \
    --imatrix "$IMATRIX_FILE" --keep-split \
    "$first_f16" "$OUTPUT_DIR/Penclaw-GLM-5.3-Abliterated-Q4_K_M.gguf" Q4_K_M \
    >>"$LOG_FILE" 2>&1

q_shards="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.gguf' | wc -l | tr -d ' ')"
if [[ "$q_shards" -lt 2 ]]; then
    echo "Quantization produced only ${q_shards} shard(s); expected a split model" >&2
    exit 5
fi

status verify "Recording checksums and repository metadata"
(
    cd "$OUTPUT_DIR"
    sha256sum ./*.gguf > checksums.sha256
)
cp "$SOURCE_DIR/config.json" "$SOURCE_DIR/generation_config.json" \
    "$SOURCE_DIR/tokenizer.json" "$SOURCE_DIR/tokenizer_config.json" \
    "$SOURCE_DIR/chat_template.jinja" "$OUTPUT_DIR/"

cat > "$OUTPUT_DIR/README.md" <<EOF
---
base_model: audnai/penclaw-GLM-5.3-abliterated
library_name: llama.cpp
pipeline_tag: text-generation
license: other
tags:
  - gguf
  - abliterated
  - glm-5.3
  - private-evaluation
---

# Penclaw GLM-5.3 Abliterated Q4_K_M

Private evaluation quantization produced from the final root BF16 checkpoint at
\`${SOURCE_REPO}\` using llama.cpp commit \`${LLAMA_COMMIT}\` and the published
GLM-5.3 importance matrix from \`${IMATRIX_REPO}\`.

This repository intentionally contains exactly one quantization so RunPod
Cached Models does not stage unrelated variants.
EOF

status upload "Uploading the single-quant repository to Hugging Face"
/opt/penclaw-venv/bin/hf upload-large-folder "$TARGET_REPO" "$OUTPUT_DIR" \
    --repo-type model >>"$LOG_FILE" 2>&1

status done "Quantization and private upload completed successfully"
touch "${STATUS_ROOT}/done"
