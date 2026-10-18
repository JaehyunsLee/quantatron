#!/usr/bin/env bash
# Step 11: GRPO in NeMo RL with the active-space NeMo Gym environment.
#
#   git clone --recursive https://github.com/NVIDIA-NeMo/RL.git $NEMO_RL_DIR   (then `uv sync` per its README)
#   NEMO_RL_DIR=/path/to/RL bash training/launch_grpo.sh [extra hydra overrides]
#
# Smoke test first:  bash training/launch_grpo.sh grpo.max_num_steps=2 grpo.num_prompts_per_step=2
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_DIR="${NEMO_RL_DIR:?set NEMO_RL_DIR to a NeMo RL checkout}"
GYM_DIR="$NEMO_RL_DIR/3rdparty/Gym-workspace/Gym"
CONFIG_NAME=grpo_active_space_lightning35.yaml

# 1. Make the environment visible to NeMo RL's Gym and put the config next to the example it inherits from.
ln -sfn "$PROJECT_DIR/resources_servers/active_space" "$GYM_DIR/resources_servers/active_space"
cp "$PROJECT_DIR/training/configs/$CONFIG_NAME" "$NEMO_RL_DIR/examples/nemo_gym/$CONFIG_NAME"

# 2. Collate train/validation with the Gym version pinned by NeMo RL (stamps routing info on every row).
export ACTIVE_SPACE_DATA="$GYM_DIR/data/active_space"
(cd "$GYM_DIR" && uv run ng_prepare_data \
    "+config_paths=[resources_servers/active_space/configs/active_space.yaml,responses_api_models/vllm_model/configs/vllm_model_for_training.yaml]" \
    +output_dirpath=data/active_space +mode=train_preparation +should_download=false)

# 3. Train.
cd "$NEMO_RL_DIR"
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "examples/nemo_gym/$CONFIG_NAME" \
    "$@"
