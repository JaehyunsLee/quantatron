#!/usr/bin/env bash
# Steps 10 and 12: collect NeMo Gym rollouts for a served model on a split.
#
#   bash serving/serve_vllm.sh &                     # base Nemotron 3.5 Lightning (Red Hat vLLM)
#   TAG=base bash scripts/run_eval.sh                # -> results/base_test.jsonl
#   MODEL=checkpoints/... bash serving/serve_vllm.sh &
#   TAG=grpo bash scripts/run_eval.sh                # -> results/grpo_test.jsonl
#   python scripts/05_evaluate.py results/base_test.jsonl results/grpo_test.jsonl
#
# Requires env.yaml (copy env.yaml.example) and `uv` on PATH (Gym builds one venv per server).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${TAG:-base}"
SPLIT="${SPLIT:-test}"
REPEATS="${REPEATS:-8}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
CONFIGS=(--config resources_servers/active_space/configs/active_space.yaml
         --config responses_api_models/vllm_model/configs/vllm_model.yaml)

mkdir -p results logs
gym env start "${CONFIGS[@]}" > "logs/gym_servers_${TAG}.log" 2>&1 &
SERVERS_PID=$!
trap 'kill $SERVERS_PID 2>/dev/null || true' EXIT

echo "waiting for Gym servers (log: logs/gym_servers_${TAG}.log)"
until gym env status 2>/dev/null | grep -q "active_space"; do
  kill -0 $SERVERS_PID 2>/dev/null || { echo "Gym servers exited; see log"; exit 1; }
  sleep 5
done

gym eval run --no-serve \
  --agent active_space_simple_agent \
  --input "resources_servers/active_space/data/${SPLIT}.jsonl" \
  --output "results/${TAG}_${SPLIT}.jsonl" \
  --num-repeats "$REPEATS" \
  --concurrency "$CONCURRENCY" \
  --temperature 1.0 --top-p 0.95 \
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
echo "wrote results/${TAG}_${SPLIT}.jsonl"
