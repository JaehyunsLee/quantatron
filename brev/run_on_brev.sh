#!/usr/bin/env bash
# Drive the active_space_candidates experiment on a Brev GPU box from your laptop. Brev auth is whatever `brev login` stored
# (see brev/BREV_TOKEN.md); this script never reads or prints a token.
#
#   bash brev/run_on_brev.sh up          # create the instance (shows the matching type + price first)
#   bash brev/run_on_brev.sh setup       # sync project, pull NeMo RL container + vLLM image, clone NeMo RL, fetch model
#   bash brev/run_on_brev.sh baseline    # serve untrained Lightning with vLLM, run test rollouts, score them
#   bash brev/run_on_brev.sh prepare     # no training: GPUs visible in the NeMo RL container + server/data wiring + collation
#   bash brev/run_on_brev.sh convert     # one-off HF -> Megatron conversion of the model (smoke/train/all run it if needed)
#   bash brev/run_on_brev.sh smoke       # 2-step GRPO smoke test (waits; stops the job on a fatal error or stall)
#   bash brev/run_on_brev.sh train       # full GRPO run (32 steps, background; use `logs` to follow)
#   bash brev/run_on_brev.sh logs        # tail the training log (JOB=candidates-smoke for the smoke test)
#   bash brev/run_on_brev.sh detach S..  # run steps ON the box (laptop can close), e.g. detach all "MODEL=qwen3-1.7b all"
#   bash brev/run_on_brev.sh pipeline    # show the detached queue's log (~/pipeline.log on the box)
#   bash brev/run_on_brev.sh evaltrained # export LoRA -> HF, serve it, run test rollouts, score vs base
#   bash brev/run_on_brev.sh all         # baseline (if stale) -> train -> wait (fail-fast) -> evaltrained: 4-row table
#   bash brev/run_on_brev.sh score       # reprint the oracle / random / base / GRPO table from saved rollouts
#   bash brev/run_on_brev.sh pull        # copy results/ and logs/ back to your laptop
#   bash brev/run_on_brev.sh shell       # interactive shell on the box
#   bash brev/run_on_brev.sh down        # delete the instance (asks you to type the name)
#
# MODEL=qwen3-4b (or qwen3-1.7b, qwen3-8b) before any command runs the same pipeline for a small Qwen3 comparison model,
# e.g. `MODEL=qwen3-4b bash brev/run_on_brev.sh all`; results land next to Lightning's and `score` lists them all.
set -euo pipefail

NAME="${BREV_INSTANCE:-active-space-gpu}"
TYPE="${BREV_TYPE:-massedcompute_A100_80Gx2}"   # 2x A100 80GB, ~$3.24/h; `brev search --gpu-name A100` for others
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_REF="${NEMO_RL_REF:-v0.7.0}"   # NGC has no :latest tag; keep the container and the cloned code on the same release
NEMO_RL_IMAGE="${NEMO_RL_IMAGE:-nvcr.io/nvidia/nemo-rl:$NEMO_RL_REF}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
# NeMo RL containers run as root over a checkout owned by the login user. Git refuses such repos ("dubious ownership"),
# and the uv builds of the Gym/Megatron submodules ask git for their version, so trust every repo inside the container.
TRUST_REPO='git config --global --add safe.directory \*'
GPUS="${GPUS:-2}"
REPEATS="${REPEATS:-16}"           # rollouts per test material; base and GRPO must use the same value to be comparable
TEMPERATURE="${TEMPERATURE:-0.6}"  # test-time sampling (training rollouts keep the config's 1.0 for exploration)
EXP="${EXP:-v2}"                   # run version: new checkpoint/log/merged dirs so a run never resumes an older one
# Policy model. Lightning is the main experiment; small dense Qwen3 models are a comparison trained and evaluated with
# the same data, GRPO settings, LoRA rank and scorer. Lightning's rollouts have no model tag (candidates_base_test.jsonl ...).
MODEL="${MODEL:-lightning}"
case "$MODEL" in
  lightning)  BASE_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 ;;
  qwen3-1.7b) BASE_MODEL=Qwen/Qwen3-1.7B ;;
  qwen3-4b)   BASE_MODEL=Qwen/Qwen3-4B ;;
  qwen3-8b)   BASE_MODEL=Qwen/Qwen3-8B ;;
  *) echo "unknown MODEL=$MODEL (lightning, qwen3-1.7b, qwen3-4b, qwen3-8b)"; exit 1 ;;
esac
if [[ "$MODEL" == lightning ]]; then
  CONFIG_NAME=grpo_active_space_candidates_lightning35.yaml RUN_NAME=lightning35-$EXP BASE_TAG=base GRPO_TAG=grpo
  VLLM_PARSERS="--tool-call-parser qwen3_coder --reasoning-parser nemotron_v3"
else
  CONFIG_NAME=grpo_active_space_candidates_qwen3.yaml RUN_NAME=$MODEL-$EXP BASE_TAG=${MODEL}_base GRPO_TAG=${MODEL}_grpo
  VLLM_PARSERS="--tool-call-parser hermes --reasoning-parser deepseek_r1"   # same parsers as grpo_active_space_candidates_qwen3.yaml
fi
CKPT_DIR=results/grpo-active-space-candidates-$RUN_NAME   # relative to the NeMo RL checkout (checkpointing.checkpoint_dir)
LOG_DIR=logs/grpo-active-space-candidates-$RUN_NAME       # TensorBoard metrics, also relative to the checkout
MERGED=merged_$RUN_NAME                             # results/<MERGED>_step<N>: base + LoRA merged to HF for vLLM
# Passed to every GRPO launch; for Lightning they equal the config's own values.
TRAIN_OVERRIDES="policy.model_name=$BASE_MODEL checkpointing.checkpoint_dir=$CKPT_DIR logger.log_dir=$LOG_DIR"
# v1 rollouts (old prompt without the cost column, 8 repeats at T=1.0) are kept in results/v1/, out of the table.

rexec() { brev exec "$NAME" "$@"; }
# ON_BOX=1: this script is running on the GPU box itself (started there by `detach`), so "remote" commands run locally
# and nothing depends on the laptop staying awake or connected.
if [[ "${ON_BOX:-0}" == 1 ]]; then
  rexec() { bash -c "$*"; }
  REMOTE_HOME="$HOME"
fi
# brev exec mixes connection chatter into stdout, so tag the value we want and pick it back out.
rval() { rexec "echo \"__VAL=\$($1)\"" 2>/dev/null | tr -d '\r' | sed -n 's/^.*__VAL=//p' | tail -1; }
# The login user differs by provider (ubuntu, shadeform, ...), so ask the box for its home dir.
remote_home() { [[ -n "${REMOTE_HOME:-}" ]] && { echo "$REMOTE_HOME"; return; }; rval 'echo $HOME'; }
[[ "${1:-}" =~ ^(up|down|shell|)$ ]] || { REMOTE="$(remote_home)"; [[ -n "$REMOTE" ]] || { echo "could not reach $NAME (brev ls)"; exit 1; }; }
# Host-side Gym client: nemo-gym 0.7 needs Python >=3.13.14, newer than the box's system Python, so use a uv-managed venv.
# Capped below 3.14: an open bound lets uv pick the newest Python (3.15), which ray (a nemo-gym dep) has no wheels for.
# --clear rebuilds a half-made venv from an earlier failed run; `|| exit 1` stops here instead of running tests without deps.
GYM_ENV='export PATH=$HOME/.local/bin:$PATH; [ -x $HOME/gym-venv/bin/gym ] || { python3 -m pip install -q --user uv && uv venv -q --clear -p ">=3.13.14,<3.14" $HOME/gym-venv && uv pip install -q -p $HOME/gym-venv/bin/python "nemo-gym>=0.7" pytest; } || exit 1; source $HOME/gym-venv/bin/activate'

sync_project() {
  [[ "${ON_BOX:-0}" == 1 ]] && return   # already running from the box's copy of the project
  local tar=/tmp/active-space-project.tgz
  # Gym builds a ~1 GB .venv inside each server dir on first use; it is platform- and Gym-version-specific, never ship it.
  # COPYFILE_DISABLE: stop macOS tar from adding ._* AppleDouble files next to every file on the box.
  # Only what the candidates track runs on the box; the PySCF pipeline and the per-orbital environment stay local.
  (cd "$ROOT" && COPYFILE_DISABLE=1 tar czf "$tar" --exclude='__pycache__' --exclude='.venv*' \
      resources_servers/active_space_candidates scripts/candidates training serving env.yaml.example brev/run_on_brev.sh)
  brev copy "$tar" "$NAME:$REMOTE/project.tgz"
  rexec "mkdir -p $REMOTE/project && tar xzf $REMOTE/project.tgz -C $REMOTE/project"
}

# Run a command inside the NeMo RL container with the repo, project and HF cache mounted.
in_container() {
  rexec "docker run --rm --gpus all --shm-size=64g --ulimit memlock=-1 -u root --network host \
    -v $REMOTE/RL:/opt/nemo-rl -v $REMOTE/project:/workspace/project -v $REMOTE/hf:/hf_home -v $REMOTE/results:/workspace/results \
    -e HF_HOME=/hf_home -e NRL_FORCE_REBUILD_VENVS=\${NRL_FORCE_REBUILD_VENVS:-false} \
    -w /opt/nemo-rl $NEMO_RL_IMAGE bash -lc '$TRUST_REPO && $1'"
}

# Serve a model as `nemotron-lightning` on :8000 with identical flags for base and trained, so only the weights differ.
#   $1: extra docker args (mounts), $2: --model value. The tokenizer/chat template always comes from the base model.
stop_vllm() { rexec "docker rm -f vllm-serve >/dev/null 2>&1 || true" || true; }
serve_vllm() {
  trap stop_vllm EXIT   # free the GPUs even if the rollouts fail
  rexec "docker rm -f vllm-serve >/dev/null 2>&1 || true; docker run -d --name vllm-serve --gpus all --ipc=host --network host \
    -v $REMOTE/hf:/root/.cache/huggingface $1 $VLLM_IMAGE --model $2 --tokenizer $BASE_MODEL --served-model-name nemotron-lightning \
    --tensor-parallel-size $GPUS --max-model-len 16384 --enable-prefix-caching --enable-auto-tool-choice \
    $VLLM_PARSERS --trust-remote-code --port 8000"
  echo "waiting for vLLM"; rexec "until curl -sf localhost:8000/v1/models >/dev/null; do docker ps -q -f name=vllm-serve | grep -q . || { docker logs --tail 60 vllm-serve; exit 1; }; sleep 10; done; echo ready"
}

# Collect test rollouts from whatever vLLM is serving -> project/results/candidates_<tag>_test.jsonl
rollouts() {
  rexec "$GYM_ENV; cd $REMOTE/project && { [ -f env.yaml ] || cp env.yaml.example env.yaml; } \
    && TAG=$1 REPEATS=$REPEATS TEMPERATURE=$TEMPERATURE MAX_OUTPUT_TOKENS=6000 bash scripts/candidates/run_eval.sh"
}

# oracle + random rows always; base / GRPO rows for every model's rollout files that exist.
score() {
  rexec "$GYM_ENV; cd $REMOTE/project && python scripts/candidates/evaluate.py \$(ls results/candidates_*_test.jsonl 2>/dev/null)"
}

baseline() {
  sync_project
  fetch_model
  serve_vllm "" "$BASE_MODEL"
  rexec "$GYM_ENV; cd $REMOTE/project && python -m pytest -q resources_servers/active_space_candidates/tests"
  rollouts $BASE_TAG
  stop_vllm
}

# GRPO jobs (smoke, train) run in detached containers so a dropped SSH connection never kills or restarts them.
#   $1: container name, $2...: extra Hydra overrides for training/launch_grpo_candidates.sh
run_job() {
  local name="$1"; shift
  local busy; busy="$(rval "docker ps --format '{{.Names}}' | grep -xE 'candidates-train|candidates-smoke' | tr '\\n' ' '" || true)"
  if [[ "$busy" == *"$name"* ]]; then
    echo "$name is already running; not restarting it (stop it with: brev exec $NAME 'docker stop $name')"; return
  elif [[ -n "${busy// }" ]]; then
    echo "another GRPO job is using the GPUs: $busy(stop it with: brev exec $NAME 'docker stop $busy')"; exit 1
  fi
  sync_project
  stop_vllm   # GRPO needs both GPUs
  convert
  rexec "docker rm -f $name >/dev/null 2>&1 || true; docker run -d --name $name --gpus all --shm-size=64g --ulimit memlock=-1 -u root --network host \
    -v $REMOTE/RL:/opt/nemo-rl -v $REMOTE/project:/workspace/project -v $REMOTE/hf:/hf_home -e HF_HOME=/hf_home -w /opt/nemo-rl $NEMO_RL_IMAGE \
    bash -lc '$TRUST_REPO && CONFIG_NAME=$CONFIG_NAME NEMO_RL_DIR=/opt/nemo-rl bash /workspace/project/training/launch_grpo_candidates.sh cluster.gpus_per_node=$GPUS $TRAIN_OVERRIDES $*'"
  echo "started $name ($MODEL)"
}

# Convert the HF model to Megatron format once, with nothing else on the box. Left to NeMo RL, the conversion happens
# inside the training workers after the colocated vLLM has parked ~60 GB of weights in host RAM, and the node runs out
# of memory (Ray kills a worker at 95% of 147 GB). The result is cached in $REMOTE/hf/nemo_rl and reused by every run.
MCORE_DIR="nemo_rl/$BASE_MODEL"   # under the HF cache; the path NeMo RL checks before converting on its own
MEGATRON_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
fetch_model() {   # no-op when the weights are already in the HF cache
  rexec "docker run --rm -v $REMOTE/hf:/hf_home -e HF_HOME=/hf_home $NEMO_RL_IMAGE bash -lc 'uvx --from huggingface-hub hf download --quiet $BASE_MODEL'"
}
convert() {
  [[ "$(rval "[ -f $REMOTE/hf/$MCORE_DIR/.conversion_complete ] && echo yes")" == yes ]] && { echo "Megatron checkpoint cached"; return; }
  fetch_model
  echo "converting $BASE_MODEL to Megatron format (one-off, ~10-20 min for Lightning, a few for Qwen3)"
  # A killed conversion leaves iter_0000000 behind, and NeMo RL would treat that as done and fail to load it.
  # Its files are root-owned (written from a container), so the container deletes them before converting.
  rexec "docker rm -f candidates-convert >/dev/null 2>&1 || true; \
    docker run -d --name candidates-convert --gpus all --shm-size=64g --ulimit memlock=-1 -u root --network host \
    -v $REMOTE/RL:/opt/nemo-rl -v $REMOTE/project:/workspace/project -v $REMOTE/hf:/hf_home -e HF_HOME=/hf_home \
    -e PYTHONPATH=/opt/nemo-rl -w /opt/nemo-rl $NEMO_RL_IMAGE \
    bash -lc 'rm -rf /hf_home/$MCORE_DIR && $TRUST_REPO && $MEGATRON_PY /workspace/project/training/convert_hf_to_megatron.py $BASE_MODEL'"
  wait_job candidates-convert
  [[ "$(rval "[ -f $REMOTE/hf/$MCORE_DIR/.conversion_complete ] && echo yes")" == yes ]] || { echo "conversion did not finish"; exit 1; }
}

# Fail fast so a broken run doesn't sit on the GPUs. Polls with short brev calls (a dropped connection costs one poll)
# and stops the container when
#   * the log shows a fatal error (Ray actor/task death, CUDA OOM, NCCL) and the job hasn't exited 5 min later, or
#   * the log hasn't grown for STALL_MIN minutes (default 45; model conversion/loading is quiet for a while).
# FAILFAST=0 turns both off and only waits for the container to exit.
FATAL_RE='RayTaskError|RayActorError|ActorDiedError|ActorUnavailableError|OutOfMemoryError|CUDA out of memory|NCCL error|Segmentation fault'
STALL_MIN="${STALL_MIN:-45}"
job_status() {  # -> "<status> <exit code> <log lines> <fatal matches in the last 300 lines>"
  rval "{ docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' $1 2>/dev/null || echo missing -1; } | tr -d '\\n'; echo -n ' '; \
    docker logs $1 2>&1 | wc -l | tr -d ' \\n'; echo -n ' '; docker logs --tail 300 $1 2>&1 | grep -cE '$FATAL_RE'"
}
abort_job() {
  echo "stopping $1: $2"
  rexec "docker stop -t 30 $1 >/dev/null; docker logs --tail 60 $1 2>&1"
  exit 1
}
wait_job() {
  local name="$1" status code lines fatal last_lines=-1 changed now fatal_at=0 polls=0
  changed="$(date +%s)"
  while :; do
    read -r status code lines fatal <<< "$(job_status "$name" || true)"
    now="$(date +%s)"
    case "$status" in
      exited)
        [[ "$code" == 0 ]] && { echo "$name finished"; return; }
        echo "$name failed (exit $code); last log lines:"; rexec "docker logs --tail 60 $name 2>&1"; exit 1 ;;
      running)
        [[ "$lines" != "$last_lines" ]] && { last_lines="$lines"; changed="$now"; }
        (( polls++ % 5 == 0 )) && echo "$(date +%H:%M) $name: $(rval "docker logs --tail 1 $name 2>&1 | cut -c1-150" || true)"
        if [[ "${FAILFAST:-1}" == 1 ]]; then
          if (( ${fatal:-0} > 0 )); then
            (( fatal_at == 0 )) && { fatal_at="$now"; echo "$(date +%H:%M) fatal error in $name log; stopping it in 5 min unless it exits"; }
            (( now - fatal_at > 300 )) && abort_job "$name" "fatal error in the log and the job did not exit"
          fi
          (( now - changed > STALL_MIN * 60 )) && abort_job "$name" "no log output for $STALL_MIN min"
        fi
        sleep 60 ;;
      missing) echo "$name container is gone (removed by hand?)"; exit 1 ;;
      "") echo "$(date +%H:%M) could not read $name status, retrying"; sleep 60 ;;
      *) abort_job "$name" "unexpected container state '$status'" ;;
    esac
  done
}

evaltrained() {
  # Default to the newest checkpoint, so an early-stopped run still evaluates. STEP=N picks a specific one.
  local step="${STEP:-$(rval "ls -d $REMOTE/RL/$CKPT_DIR/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1")}"
  [[ -n "$step" ]] || { echo "no checkpoints in $REMOTE/RL/$CKPT_DIR; run train first"; exit 1; }
  echo "evaluating GRPO checkpoint step_$step"
  sync_project
  stop_vllm
  rexec "[ -f $REMOTE/results/${MERGED}_step$step/config.json ] && echo '${MERGED}_step$step exists, skipping LoRA merge' || \
    docker run --rm --gpus all --shm-size=64g -u root -v $REMOTE/RL:/opt/nemo-rl -v $REMOTE/hf:/hf_home -v $REMOTE/results:/workspace/results -e HF_HOME=/hf_home -w /opt/nemo-rl $NEMO_RL_IMAGE bash -lc \
      '$TRUST_REPO && uv run --extra mcore python examples/converters/convert_lora_to_hf.py --base-ckpt /hf_home/nemo_rl/$BASE_MODEL/iter_0000000 \
       --adapter-ckpt $CKPT_DIR/step_$step/policy/weights/iter_0000000 --hf-model-name $BASE_MODEL --hf-ckpt-path /workspace/results/${MERGED}_step$step'"
  serve_vllm "-v $REMOTE/results/${MERGED}_step$step:/model" /model
  rollouts $GRPO_TAG
  stop_vllm
  score
}

case "${1:-}" in
  up)
    echo "== matching instance (dry run)"; brev create "$NAME" --type "$TYPE" --dry-run
    if [[ "${ASSUME_YES:-0}" != 1 ]]; then read -r -p "Create '$NAME' ($TYPE)? This starts billing. [y/N] " ok; [[ "$ok" == y ]] || exit 1; fi
    brev create "$NAME" --type "$TYPE"
    echo "wait for RUNNING:"; brev ls
    ;;
  setup)
    sync_project
    rexec "nvidia-smi --query-gpu=name,memory.total --format=csv && df -h $REMOTE | tail -1 && free -g | head -2 && docker --version"
    rexec "mkdir -p $REMOTE/hf $REMOTE/results $REMOTE/logs && docker pull $NEMO_RL_IMAGE && docker pull $VLLM_IMAGE"
    rexec "[ -d $REMOTE/RL ] || git clone --recursive --depth 1 --branch $NEMO_RL_REF https://github.com/NVIDIA-NeMo/RL.git $REMOTE/RL"
    rexec "docker run --rm -v $REMOTE/hf:/hf_home -e HF_HOME=/hf_home $NEMO_RL_IMAGE bash -lc 'uvx --from huggingface-hub hf download $BASE_MODEL'"
    ;;
  baseline) baseline; score ;;
  prepare)
    sync_project
    in_container "nvidia-smi -L && PREPARE_ONLY=1 CONFIG_NAME=$CONFIG_NAME NEMO_RL_DIR=/opt/nemo-rl bash /workspace/project/training/launch_grpo_candidates.sh $TRAIN_OVERRIDES"
    ;;
  convert) sync_project; stop_vllm; convert ;;
  smoke)
    run_job candidates-smoke grpo.max_num_steps=2 grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 checkpointing.enabled=false
    wait_job candidates-smoke
    ;;
  train) run_job candidates-train; echo "follow with: bash brev/run_on_brev.sh logs" ;;
  logs) rexec "docker logs -f --tail 80 ${JOB:-candidates-train}" ;;   # JOB=candidates-smoke for the smoke test
  wait) wait_job "${JOB:-candidates-train}" ;;   # block until a running GRPO job ends (fail-fast rules apply)
  detach)
    # Queue steps to run ON the box under nohup, so the laptop can be closed. Each argument is one step:
    # "[VAR=value ...] command", run in order (a failed step does not stop the next one). Example:
    #   bash brev/run_on_brev.sh detach wait evaltrained "MODEL=qwen3-1.7b all"
    shift; (( $# )) || set -- all
    sync_project
    queue=""
    for step in "$@"; do
      vars="" cmd=""
      for w in $step; do if [[ "$w" == *=* ]]; then vars+="$w "; else cmd+="$w "; fi; done
      queue+="echo; echo \"=== \$(date +%F_%T) step: $step\"; env ON_BOX=1 ${vars}bash brev/run_on_brev.sh ${cmd}; "
    done
    queue+="echo; echo \"=== \$(date +%F_%T) queue finished\""
    # cd in the foreground and background only the fully redirected nohup, so brev exec isn't held open by it
    rexec "cd $REMOTE/project && { nohup setsid bash -c '$queue' >> $REMOTE/pipeline.log 2>&1 < /dev/null & } && echo queued on the box: $*"
    echo "follow with: bash brev/run_on_brev.sh pipeline"
    ;;
  pipeline) rexec "tail -n ${LINES:-40} $REMOTE/pipeline.log; pgrep -f 'ON_BOX=1' >/dev/null && echo '(queue still running)' || echo '(queue not running)'" ;;
  evaltrained) evaltrained ;;
  all)
    # Reuses a base run made after the current prompts/data were built (test.jsonl); BASELINE=1 forces one,
    # e.g. after changing REPEATS or TEMPERATURE.
    sync_project
    if [[ "${BASELINE:-0}" == 1 || "$(rval "cd $REMOTE/project && [ -s results/candidates_${BASE_TAG}_test.jsonl ] && \
          [ results/candidates_${BASE_TAG}_test.jsonl -nt resources_servers/active_space_candidates/data/test.jsonl ] && echo yes")" != yes ]]; then
      baseline
    else
      echo "reusing results/candidates_${BASE_TAG}_test.jsonl (BASELINE=1 to redo)"
    fi
    run_job candidates-train
    wait_job candidates-train
    evaltrained   # ends with the oracle / random / base / GRPO table (every model run so far)
    ;;
  score) score ;;
  pull)
    mkdir -p "$ROOT/results" "$ROOT/logs"
    brev copy "$NAME:$REMOTE/project/results/" "$ROOT/results/" || true
    brev copy "$NAME:$REMOTE/project/logs/" "$ROOT/logs/" || true
    # training metrics (TensorBoard event files) live in the NeMo RL checkout; checkpoints stay on the box (large)
    brev copy "$NAME:$REMOTE/RL/$LOG_DIR/" "$ROOT/$LOG_DIR/" || true   # this MODEL's run; MODEL=... pull for the others
    ;;
  shell) brev shell "$NAME" ;;
  down)
    read -r -p "Type the instance name to DELETE it ($NAME): " n; [[ "$n" == "$NAME" ]] || exit 1
    brev delete "$NAME"
    ;;
  *) sed -n 2,25p "${BASH_SOURCE[0]}"; exit 1 ;;
esac
