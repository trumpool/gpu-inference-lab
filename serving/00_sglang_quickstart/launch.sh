#!/bin/bash
# 在 5070 Ti 上启动 SGLang 服务（OpenAI 兼容接口），Ctrl-C 停止。
# 用法：bash serving/00_sglang_quickstart/launch.sh [其他 sglang 参数...]
#   换模型：MODEL=Qwen/Qwen2.5-0.5B-Instruct bash serving/00_sglang_quickstart/launch.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
PORT=${PORT:-30000}
LOG=${LOG:-/tmp/sglang_server.log}

# FlashInfer 运行时要用 nvcc 现场编译 kernel：系统的 CUDA 12.9 要在 PATH 上，
# ninja 在 venv 的 bin/ 里，所以必须先激活 venv
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
source ~/venvs/sglang/bin/activate

echo "model=$MODEL port=$PORT log=$LOG"
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  "$@" 2>&1 | tee "$LOG"
