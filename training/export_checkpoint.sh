#!/usr/bin/env bash
# Merge the GRPO LoRA adapter into the base model and export a HuggingFace checkpoint that
# serving/serve_vllm.sh (Red Hat vLLM) can load.
#
#   NEMO_RL_DIR=/path/to/RL STEP=60 bash training/export_checkpoint.sh
set -euo pipefail

NEMO_RL_DIR="${NEMO_RL_DIR:?set NEMO_RL_DIR}"
STEP="${STEP:?set STEP to the checkpoint step to export}"
BASE_MODEL="${BASE_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
RESULTS="${RESULTS:-$NEMO_RL_DIR/results/grpo-active-space-lightning35}"
OUT="${OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/checkpoints/lightning35-active-space-step$STEP}"

cd "$NEMO_RL_DIR"
uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
    --base-ckpt "$HOME/.cache/huggingface/nemo_rl/$BASE_MODEL/iter_0000000" \
    --adapter-ckpt "$RESULTS/step_$STEP/policy/weights/iter_0000000" \
    --hf-model-name "$BASE_MODEL" \
    --hf-ckpt-path "$OUT"
echo "exported to $OUT"
