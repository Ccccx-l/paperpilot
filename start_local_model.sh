#!/usr/bin/env bash
# 启动本地 Qwen3-VL-8B 的 OpenAI 兼容服务（vLLM）
# mini-harness 通过 http://127.0.0.1:8000/v1 调用它
set -euo pipefail

# 机器相关路径不提供默认值，必须通过环境变量传入：
#   MODEL_PATH  Qwen3-VL-8B-Instruct 模型目录（必须存在）
#   VLLM_VENV   提供 bin/vllm 的虚拟环境目录
# 本地媒体目录默认为本脚本所在目录（不随当前工作目录变化），可用环境变量覆盖：
#   ALLOWED_LOCAL_MEDIA_PATH  允许 vLLM 读取的本地媒体目录
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b}"
PORT="${PORT:-8000}"
VLLM_VENV="${VLLM_VENV:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM="${GPU_MEM:-0.70}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$SCRIPT_DIR}"

if [ -z "$MODEL_PATH" ]; then
  echo "error: 未设置环境变量 MODEL_PATH，请将其指向 Qwen3-VL-8B-Instruct 模型目录，例如：MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct $0" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "error: 环境变量 MODEL_PATH 指向的目录不存在，请检查 MODEL_PATH（当前值无效）" >&2
  exit 1
fi
if [ -z "$VLLM_VENV" ]; then
  echo "error: 未设置环境变量 VLLM_VENV，请将其指向包含 bin/vllm 的虚拟环境目录，例如：VLLM_VENV=/path/to/venv $0" >&2
  exit 1
fi
if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
  echo "error: 环境变量 VLLM_VENV 下缺少可执行的 bin/vllm，请检查 VLLM_VENV 指向的虚拟环境" >&2
  exit 1
fi
if [ ! -d "$ALLOWED_LOCAL_MEDIA_PATH" ]; then
  echo "error: 环境变量 ALLOWED_LOCAL_MEDIA_PATH 指向的目录不存在，请检查 ALLOWED_LOCAL_MEDIA_PATH（不设置时默认为脚本所在目录）" >&2
  exit 1
fi

exec "$VLLM_VENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH"
