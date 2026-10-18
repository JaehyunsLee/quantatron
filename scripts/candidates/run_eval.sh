#!/usr/bin/env bash
# Collect NeMo Gym rollouts for a served model on the active_space_candidates test split (8 held-out materials).
#
#   bash serving/serve_vllm.sh &                                # Nemotron 3.5 Lightning behind an OpenAI-style API
#   TAG=base bash scripts/candidates/run_eval.sh                # -> results/candidates_base_test.jsonl
#   python scripts/candidates/evaluate.py results/candidates_base_test.jsonl
#
# Needs env.yaml in the repo root (copy env.yaml.example) and `uv` on PATH (Gym builds one venv per server).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root: Gym resolves resources_servers/ and env.yaml from here

TAG="${TAG:-base}"
REPEATS="${REPEATS:-16}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.6}"   # below the training rollouts' 1.0: test the policy's preferred answer, less sampling noise
SERVER_CFG=resources_servers/active_space_candidates/configs/active_space_candidates.yaml
EVAL_CFG=resources_servers/active_space_candidates/configs/active_space_candidates_eval.yaml   # eval-only environment server
MODEL_CFG="$(python3 -c 'import nemo_gym,os;print(os.path.join(os.path.dirname(nemo_gym.__file__),"..","responses_api_models/vllm_model/configs/vllm_model.yaml"))' 2>/dev/null || echo responses_api_models/vllm_model/configs/vllm_model.yaml)"

[ -f env.yaml ] || cp env.yaml.example env.yaml   # Gym reads env.yaml from the working directory
mkdir -p results logs
gym env start --config "$SERVER_CFG" --config "$EVAL_CFG" --config "$MODEL_CFG" > "logs/gym_candidates_servers_${TAG}.log" 2>&1 &
SERVERS_PID=$!
trap 'kill $SERVERS_PID 2>/dev/null || true' EXIT

echo "waiting for Gym servers (log: logs/gym_candidates_servers_${TAG}.log)"
until grep -q "4 / 4 servers ready" "logs/gym_candidates_servers_${TAG}.log" 2>/dev/null || gym env status 2>/dev/null | grep -q "active_space_candidates"; do
  kill -0 $SERVERS_PID 2>/dev/null || { echo "Gym servers exited; see log"; exit 1; }
  sleep 5
done

gym eval run --no-serve \
  --agent active_space_candidates_simple_agent \
  --input "resources_servers/active_space_candidates/data/test.jsonl" \
  --output "results/candidates_${TAG}_test.jsonl" \
  --num-repeats "$REPEATS" \
  --concurrency "$CONCURRENCY" \
  --temperature "$TEMPERATURE" --top-p 0.95 \
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
echo "wrote results/candidates_${TAG}_test.jsonl"
