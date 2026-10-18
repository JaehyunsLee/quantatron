#!/usr/bin/env bash
# GRPO training on the active_space_candidates environment (2 GPUs): Nemotron 3.5 Lightning by default, Qwen3 with
# CONFIG_NAME=grpo_active_space_candidates_qwen3.yaml policy.model_name=Qwen/Qwen3-4B (brev/run_on_brev.sh sets both
# from MODEL=...). Run it inside the NeMo RL container (see brev/run_on_brev.sh) or on any box with a NeMo RL checkout:
#
#   NEMO_RL_DIR=/opt/nemo-rl bash training/launch_grpo_candidates.sh [extra hydra overrides]
#
# Smoke test first:  bash training/launch_grpo_candidates.sh grpo.max_num_steps=2 grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_DIR="${NEMO_RL_DIR:?set NEMO_RL_DIR to a NeMo RL checkout (submodules initialised)}"
GYM_DIR="$NEMO_RL_DIR/3rdparty/Gym-workspace/Gym"
SERVER=active_space_candidates
CONFIG_NAME="${CONFIG_NAME:-grpo_active_space_candidates_lightning35.yaml}"   # grpo_active_space_candidates_qwen3.yaml for the Qwen comparison runs

# 1. Put the server directory (app.py, requirements.txt, configs/, data/) where Gym looks for servers.
rm -rf "$GYM_DIR/resources_servers/$SERVER"
cp -r "$PROJECT_DIR/resources_servers/$SERVER" "$GYM_DIR/resources_servers/$SERVER"
# A .venv left by the eval-side Gym (pip nemo-gym) would be reused as-is by this older pinned Gym; let it build its own.
rm -rf "$GYM_DIR/resources_servers/$SERVER"/.venv*
# Both candidates configs, since the Qwen config inherits from the Lightning one through `defaults:`.
cp "$PROJECT_DIR"/training/configs/grpo_active_space_candidates_*.yaml "$NEMO_RL_DIR/examples/nemo_gym/"

# 2. Collate the train split with the Gym version pinned by NeMo RL (stamps routing info on every row).
#    The test split is not declared in the Gym config, so it can never end up in this file.
COLLATED="$GYM_DIR/data/$SERVER"
rm -rf "$COLLATED"
GYM_CONFIGS="resources_servers/$SERVER/configs/$SERVER.yaml,responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
(cd "$GYM_DIR" && {
    if uv run which ng_prepare_data >/dev/null 2>&1; then
        uv run ng_prepare_data "+config_paths=[$GYM_CONFIGS]" "+output_dirpath=$COLLATED" +mode=train_preparation +should_download=false
    else
        uv run gym dataset collate "+config_paths=[$GYM_CONFIGS]" "+output_dirpath=$COLLATED" +mode=train_preparation +should_download=false
    fi
})
test -s "$COLLATED/train.jsonl" || { echo "collation produced no train.jsonl in $COLLATED"; exit 1; }
export ACTIVE_SPACE_CANDIDATES_DATA="$COLLATED"
echo "collated $(wc -l < "$COLLATED/train.jsonl") train rows -> $COLLATED"

# 3. Validate the merged config (defaults + this file + overrides) the same way run_grpo_nemo_gym.py does, so a schema
#    mismatch with the pinned NeMo RL release fails here in seconds instead of after the container/venv spin-up.
cd "$NEMO_RL_DIR"
uv run python - "examples/nemo_gym/$CONFIG_NAME" "$@" <<'EOF'
import sys
from omegaconf import OmegaConf
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers

register_omegaconf_resolvers()
config = load_config(sys.argv[1])
if sys.argv[2:]:
    config = parse_hydra_overrides(config, sys.argv[2:])
MasterConfig(**OmegaConf.to_container(config, resolve=True))
print("config OK:", sys.argv[1], *sys.argv[2:])
EOF
[[ "${PREPARE_ONLY:-0}" == 1 ]] && { echo "PREPARE_ONLY=1: wiring OK, not training"; exit 0; }

# 4. Train.
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "examples/nemo_gym/$CONFIG_NAME" \
    "$@"
