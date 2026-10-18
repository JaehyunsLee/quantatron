#!/usr/bin/env bash
# Serve Nemotron 3.5 Lightning (base or GRPO-trained) behind an OpenAI-compatible API with
# Red Hat AI Inference Server (Red Hat's supported vLLM build), for baseline and test evaluation.
#
#   bash serving/serve_vllm.sh                                   # base model, Red Hat vLLM container
#   MODEL=/abs/path/to/merged_hf_model bash serving/serve_vllm.sh   # trained checkpoint
#   RUNTIME=pip bash serving/serve_vllm.sh                       # plain `vllm serve` in the current env
#
# Needs `podman login registry.redhat.io` once (or docker). The image's vLLM must know the
# `nemotron_v3` reasoning parser and the `qwen3_coder` tool parser (NVIDIA's cookbook uses vLLM 0.27.1);
# if the Red Hat tag you have is older, set RHAIIS_IMAGE to a newer tag, or as a fallback
# RHAIIS_IMAGE=docker.io/vllm/vllm-openai:v0.27.1.
set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-nemotron-lightning}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-32768}"
RUNTIME="${RUNTIME:-container}"
ENGINE="${ENGINE:-$(command -v podman || command -v docker)}"
RHAIIS_IMAGE="${RHAIIS_IMAGE:-registry.redhat.io/rhaiis/vllm-cuda-rhel9:latest}"

VLLM_ARGS=(
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser nemotron_v3
  --trust-remote-code
)

if [[ "$RUNTIME" == "pip" ]]; then
  exec vllm serve "$MODEL" "${VLLM_ARGS[@]}"
fi

MOUNTS=(-v "$HOME/.cache/huggingface:/opt/app-root/src/.cache/huggingface:Z")
MODEL_ARG="$MODEL"
if [[ -d "$MODEL" ]]; then  # local checkpoint -> mount it
  MOUNTS+=(-v "$MODEL:/models/served:Z")
  MODEL_ARG=/models/served
fi

exec "$ENGINE" run --rm -it \
  --device nvidia.com/gpu=all --security-opt=label=disable --shm-size=8g \
  -p "$PORT:$PORT" "${MOUNTS[@]}" \
  -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  "$RHAIIS_IMAGE" \
  --model "$MODEL_ARG" "${VLLM_ARGS[@]}"
