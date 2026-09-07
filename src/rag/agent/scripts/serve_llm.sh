#!/usr/bin/env bash
#
#   bash src/rag/agent/scripts/serve_llm.sh         
#   bash src/rag/agent/scripts/serve_llm.sh unsloth/other-GGUF:Q4_K_M  # ghi đè nhanh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}" # Trường hợp build llama.cpp ngoài repo (ví dụ home/)

find_llama_server() {
    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server
        return 0
    fi
    for candidate in \
        "$LLAMA_CPP_DIR/build/bin/llama-server" \
        "$LLAMA_CPP_DIR/llama-server"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if ! LLAMA_SERVER="$(find_llama_server)"; then
    echo "LỖI: không tìm thấy llama-server." >&2
    echo "  đã dò: \$PATH, $LLAMA_CPP_DIR/build/bin/, $LLAMA_CPP_DIR/" >&2
    echo "  repo llama.cpp ở chỗ khác thì: LLAMA_CPP_DIR=/duong/dan bash $0" >&2
    exit 1
fi

MODEL_HF="${1:-$(cd "$REPO_ROOT" && python -c 'from src.rag.config import CITATION_AGENT_MODEL; print(CITATION_AGENT_MODEL)')}"
MODEL_ALIAS="citation-agent"
CONFIGURED_BASE_URL="$(cd "$REPO_ROOT" && python -c 'from src.rag.config import CITATION_AGENT_BASE_URL; print(CITATION_AGENT_BASE_URL)')"


NGL="${LLAMA_NGL:-99}"
CTX="${LLAMA_CTX:-10132}"
PORT="${LLAMA_PORT:-8080}"
RUNTIME_BASE_URL="http://127.0.0.1:${PORT}/v1"

echo "binary : ${LLAMA_SERVER}"
echo "model  : ${MODEL_HF}"
echo "alias  : ${MODEL_ALIAS}"
echo "cấu hình: -ngl ${NGL}  -c ${CTX}  --parallel 1  --reasoning off  -> ${RUNTIME_BASE_URL}"
echo "cache  : ${LLAMA_CACHE:-$HOME/.cache/llama.cpp} (nơi -hf tải GGUF về)"

if [[ "${RUNTIME_BASE_URL%/}" != "${CONFIGURED_BASE_URL%/}" ]]; then
    echo "CẢNH BÁO: runtime URL ${RUNTIME_BASE_URL} khác CITATION_AGENT_BASE_URL ${CONFIGURED_BASE_URL}" >&2
fi

# --jinja: bắt buộc để server dùng chat template thật của model
# --reasoning off: Qwen3.5 là reasoning model, mặc định `auto` dò từ template -> bật thinking
# -ngl 99: đẩy toàn bộ layer lên GPU
exec "$LLAMA_SERVER" -hf "$MODEL_HF" \
    --alias "$MODEL_ALIAS" \
    --jinja \
    --reasoning off \
    --parallel 1 \
    -ngl "$NGL" \
    -c "$CTX" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --flash-attn on
