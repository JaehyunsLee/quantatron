# active_space_candidates: step-by-step fine-tuning guide

Fine-tune **Nemotron 3.5 Lightning (30B-A3B)** with **GRPO + LoRA** (NeMo RL) in the `active_space_candidates`
**NeMo Gym** environment, whose task is: *given a material, pick the active space with the best
accuracy-vs-quantum-cost trade-off from a fixed menu of candidates*. Everything GPU-side runs on one **NVIDIA Brev**
box (2× A100 80GB) driven from your laptop by [`brev/run_on_brev.sh`](../brev/run_on_brev.sh).

```text
laptop ── brev exec/copy ──▶ Brev box (2× A100 80GB)
                              ├─ vLLM container      : serves Lightning for baseline / trained eval
                              ├─ NeMo RL container   : GRPO training (Megatron LoRA + colocated vLLM)
                              └─ NeMo Gym (host venv): active_space_candidates resources server + agent
```

Results of the runs described here are in the [main README](../README.md#results) and
[`docs/results/`](results/).

Contents: [0 What gets trained](#0-what-gets-trained) · [1 Laptop prep](#1-one-time-laptop-prep-no-gpu-free) ·
[2 Create the box](#2-create-the-gpu-box) · [3 Setup](#3-set-up-the-box-25-min) ·
[4 Baseline](#4-baseline-the-untrained-model) · [5 Wiring check](#5-wiring-check-no-training) ·
[6 Smoke test](#6-grpo-smoke-test-2-steps) · [7 Train](#7-full-training-run-32-steps) ·
[8 Evaluate](#8-evaluate-the-fine-tuned-model) · [9 Pull & shut down](#9-pull-results-and-delete-the-box) ·
[v1 → v2](#v1--v2) · [Unattended runs](#running-unattended-laptop-closed) ·
[10 Plots & Qwen3](#10-plots-and-the-qwen3-comparison) · [Tuning](#tuning-knobs) ·
[Troubleshooting](#troubleshooting) · [Files](#files)

---

## 0. What gets trained

|            |                                                                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Base model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (ungated, ~60 GB)                                                           |
| Method     | GRPO, LoRA rank 32 / alpha 64 on Megatron (TP2, EP2), vLLM generation colocated on the same 2 GPUs                              |
| Data       | 16 **train** materials (rollouts) / 8 **held-out test** materials (whole chemical families held out), Phasecraft DZVP table     |
| Batch      | 4 materials × 12 rollouts = 48 episodes per step, 32 steps (~470 s each, ~4.2 h), checkpoint every 8, LR 3e-5, PPO dual-clip 3 |
| Reward     | 0 for an invalid / off-menu answer, else the min-max normalized score of the first valid `submit_active_space`                  |
| Score      | `corr_recovered − 0.2 · log10(toffoli_total)` (β = 0.2, read from the DZVP table)                                               |

Built from [`data/phasecraft_tables/dzvp.csv`](../data/phasecraft_tables/) by
[`scripts/candidates/build_env_data.py`](../scripts/candidates/build_env_data.py), keeping only rows with
`use_for_training` and all agent-visible fields: 24 of the table's 34 materials. Dropped: 7 whose windows hold ~0
correlation (CsPbF3, CsPbI3, LaFeO3, LaNiO3, LaTiO3, PbHfO3, SrTiO3), YFeO3 (wrong imported structure), and Al2O3 /
LiNiO2 (imported rows with no gap, window or orbital character). Test families tm_binary_oxide and ionic_salt never
appear in train; the SZV split's third test family, halide perovskites, has no usable DZVP rows.

| Split      | Materials                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------- |
| train (16) | AlAs, BN, BaNiO3, BaRuO3, C, CaO, CdSe, MgO, NiO2, PbTiO3, Si, SiC, Sr2RuO4, SrMoO3, SrZrO3, ZnSe |
| test (8)   | HfO2, PdO, NbO2, TaO2, Nb2O5, CuO (TM binary oxides); BaS, NaCl (ionic salts)                     |

An episode: the agent calls `get_structure` / `get_orbitals` to inspect the candidate menu, then
`submit_active_space(material_id, candidate_id)`. Scores live only in the verify-side `scores.json`; the agent
only ever sees `orbitals.json`.

**Cost.** The box is ~$3.24/h. Setup ≈ 25 min, baseline ≈ 15 min, smoke ≈ 20–30 min, 32-step training ≈ 4.2 h,
evaluation ≈ 20 min → roughly **$20–25** for the whole guide. Billing runs until you **delete** the box (step 9).

---

## 1. One-time laptop prep (no GPU, free)

From the repo root:

```bash
# 1a. Python env for the Gym client + tests (Python >= 3.12)
python3 -m venv .venv && .venv/bin/pip install -r requirements-gym.txt -r requirements-analysis.txt

# 1b. Environment logic tests: server tools, verifier, split leakage, GRPO config shape
.venv/bin/python -m pytest -q resources_servers/active_space_candidates/tests    # expect: 9 passed

# 1c. Brev CLI login (browser flow; never paste tokens into files or chats)
brev login && brev org ls                                    # see brev/BREV_TOKEN.md for safe token storage
```

Optional: a full NeMo Gym rollout loop with a **scripted fake model** (no GPU) to see what an episode and the
output file look like:

```bash
cp env.yaml.example env.yaml
.venv/bin/python scripts/candidates/mock_policy_server.py --port 8000 &   # pretends to be vLLM
PATH=$PWD/.venv/bin:$PATH TAG=mock REPEATS=2 bash scripts/candidates/run_eval.sh
.venv/bin/python scripts/candidates/evaluate.py results/candidates_mock_test.jsonl
kill %1
```

---

## 2. Create the GPU box

```bash
bash brev/run_on_brev.sh up        # shows the type + price, asks y/N, then creates 'active-space-gpu'
brev ls                            # wait until STATUS RUNNING, BUILD COMPLETED, SHELL READY (~5–10 min)
```

Defaults: name `active-space-gpu`, type `massedcompute_A100_80Gx2`. If that type is sold out, pick another
2× A100/H100 80GB type and pass it in:

```bash
brev search --gpu-name A100 --min-vram 80 --sort price
BREV_TYPE=<type-from-search> bash brev/run_on_brev.sh up
```

All later commands talk to the box by name (`BREV_INSTANCE=...` to override). They auto-detect the box's
home directory (it is `/home/shadeform` on Massed Compute, `/home/ubuntu` elsewhere).

---

## 3. Set up the box (~25 min)

```bash
bash brev/run_on_brev.sh setup
```

What it does, in order:

1. Copies `resources_servers/active_space_candidates/`, `scripts/candidates/`, `training/`, `serving/` and
   `env.yaml.example` to `~/project` on the box.
2. Prints GPUs / disk / RAM / Docker version — expect `2 × NVIDIA A100 80GB`, >500 GB free.
3. `docker pull nvcr.io/nvidia/nemo-rl:v0.7.0` (training) and `vllm/vllm-openai:v0.27.1` (serving).
4. `git clone --recursive --branch v0.7.0` NeMo RL into `~/RL` (includes the pinned NeMo Gym under `3rdparty/`).
5. Downloads the BF16 Lightning weights into `~/hf` (shared by both containers).

Re-running `setup` is safe; finished pieces are skipped/cached.

---

## 4. Baseline the untrained model

```bash
bash brev/run_on_brev.sh baseline
```

Starts vLLM (TP=2, `--reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice`),
installs the Gym client in `~/gym-venv`, runs the environment tests, then 8 test materials × 16 repeats
(temperature 0.6) through `scripts/candidates/run_eval.sh` and prints:

```text
| method                          | reward | <=5% of oracle | valid |
| oracle                          |  1.000 |          1.000 |  1.00 |
| random menu entry               |    ... |            ... |  1.00 |
| rule: largest candidate         |    ... |            ... |  1.00 |
| rule: tightest window/orbital   |    ... |            ... |  1.00 |
| candidates_base_test            |    ... |            ... |   ... |
| candidates_base_test (majority) |    ... |            ... |   ... |
```

`reward` (0–1) is the number training should raise; `valid` is the fraction of episodes that ended with an
on-menu submission. The vLLM container is removed at the end so the GPUs are free for training.
Fewer repeats for a quicker look: `REPEATS=2 bash brev/run_on_brev.sh baseline`.

---

## 5. Wiring check (no training)

```bash
bash brev/run_on_brev.sh prepare
```

Inside the NeMo RL container: lists the GPUs, copies the `active_space_candidates` server into NeMo RL's Gym
checkout, copies the GRPO configs next to the example they inherit from, and collates the train split. It stops
before training. Expect:

```text
GPU 0: NVIDIA A100 80GB PCIe ...
GPU 1: NVIDIA A100 80GB PCIe ...
collated 16 train rows -> /opt/nemo-rl/3rdparty/Gym-workspace/Gym/data/active_space_candidates
config OK: examples/nemo_gym/grpo_active_space_candidates_lightning35.yaml ...
PREPARE_ONLY=1: wiring OK, not training
```

The first run in a fresh container builds NeMo RL's uv venvs, so it can take several minutes.

---

## 6. GRPO smoke test (2 steps)

```bash
bash brev/run_on_brev.sh smoke
```

Runs `training/launch_grpo_candidates.sh` with `grpo.max_num_steps=2 grpo.num_prompts_per_step=2
grpo.num_generations_per_prompt=4 checkpointing.enabled=false`. This is the step that proves the **memory fit**
(Megatron LoRA + colocated vLLM on 2 GPUs) and the full Gym ⇄ NeMo RL loop. Success looks like two `Step 1/2`,
`Step 2/2` blocks with `Avg reward`, `Loss` and timing lines, then a clean exit.

If it runs out of memory, see [Troubleshooting](#troubleshooting) before starting step 7.

---

## 7. Full training run (32 steps)

```bash
bash brev/run_on_brev.sh train     # starts the 'candidates-train' container in the background, returns immediately
bash brev/run_on_brev.sh logs      # follow the log (Ctrl-C only stops following; training keeps running)
```

Things to watch in the log per step:

- `Avg reward` / `rewards` — should trend upward (noisy: only 48 episodes/step).
- Fraction of zero rewards — invalid or off-menu submissions; should fall early.
- `Mean generation length` — should not blow up toward `max_total_sequence_length` (8192).

Checkpoints are written to `~/RL/results/grpo-active-space-candidates-lightning35-v2/step_{8,16,24,32}` (all 4 kept);
TensorBoard events to `~/RL/logs/grpo-active-space-candidates-lightning35-v2`. To view TensorBoard live:

```bash
brev port-forward active-space-gpu -p 6006:6006 &
brev exec active-space-gpu 'pip install -q --user tensorboard && ~/.local/bin/tensorboard --logdir ~/RL/logs --port 6006' &
# open http://localhost:6006
```

Stop early: `brev exec active-space-gpu 'docker stop candidates-train'` (the last saved checkpoint stays usable).
Pass extra Hydra overrides by editing the `train)` line in `brev/run_on_brev.sh`, or run step 7 by hand:

```bash
bash brev/run_on_brev.sh shell
docker run -it --rm --gpus all --shm-size=64g --ulimit memlock=-1 -u root --network host \
  -v ~/RL:/opt/nemo-rl -v ~/project:/workspace/project -v ~/hf:/hf_home -e HF_HOME=/hf_home \
  -w /opt/nemo-rl nvcr.io/nvidia/nemo-rl:v0.7.0 \
  bash -lc 'NEMO_RL_DIR=/opt/nemo-rl bash /workspace/project/training/launch_grpo_candidates.sh grpo.max_num_steps=40'
```

---

## 8. Evaluate the fine-tuned model

```bash
bash brev/run_on_brev.sh evaltrained              # newest checkpoint; STEP=16 for a specific one
bash brev/run_on_brev.sh all                      # or steps 4 + 7 + 8 in one go (reuses an existing baseline)
bash brev/run_on_brev.sh score                    # reprint the table later
```

1. Merges the LoRA adapter at `step_$STEP` into the base model → HF checkpoint
   `~/results/merged_lightning35-v2_step$STEP` (NeMo RL's `examples/converters/convert_lora_to_hf.py`).
2. Serves it with vLLM under the same name (`nemotron-lightning`) and parsers as the baseline.
3. Runs the same 8 test materials × 16 repeats and prints the step-4 table with `candidates_grpo_test` rows added.

The test materials were never seen in training, so a gain here is generalization to new chemical families.

---

## 9. Pull results and delete the box

```bash
bash brev/run_on_brev.sh pull      # results/*.jsonl, logs/, TensorBoard events -> your laptop
bash brev/run_on_brev.sh down      # type the instance name to confirm; this stops billing
brev ls                            # confirm it is gone
```

Checkpoints (~60 GB merged) are deliberately not pulled. If you want to keep the LoRA adapter only:
`brev copy active-space-gpu:~/RL/results/grpo-active-space-candidates-lightning35-v2/step_32 ./checkpoints/`.

## v1 → v2

The first full run (v1: 20 steps × 5 × 8, LR 5e-6, old prompt; rollouts kept in `results/v1/`) reached test reward
0.421 against 0.389 base and 0.399 random, all within each other's 95% CIs, while the no-model rule "pick the largest
candidate" scored 0.741: at β = 0.2 the recovered-correlation spread within a menu (~0.56) outweighs the cost-term
spread (~0.31), but the old prompt told the model that bigger spaces are expensive. v2 changes:

- **Prompt / data**: each menu entry shows `log10_toffolis` (cheap cost estimate; accuracy stays hidden), and the
  system prompt states the exchange rate (one decade of Toffolis = β of score). Rebuild with
  `python scripts/candidates/build_env_data.py`.
- **Training**: LR 3e-5, 4 × 12 rollouts, 32 steps, `loss_fn.ratio_clip_c: 3.0`, new checkpoint/log dirs (`EXP=v2`).
- **Evaluation**: 16 repeats at temperature 0.6 (`REPEATS`, `TEMPERATURE`); the table adds the two rules
  ("largest candidate", "tightest window/orbital") and a majority vote per model.

v2 moved the test reward from 0.404 (base) to 0.411, still within the base model's 95% CI and well below the
"largest candidate" rule; see the [README](../README.md#results).

## Running unattended (laptop closed)

Every command above is driven from the laptop over `brev exec`; only the GRPO container itself is detached. To let a
whole sequence run while the laptop is closed, queue it on the box:

```bash
bash brev/run_on_brev.sh detach all "MODEL=qwen3-1.7b all"   # one step per argument, run in order on the box
bash brev/run_on_brev.sh pipeline                            # later, from any machine: tail ~/pipeline.log
```

`detach` syncs the project, then starts the steps under `nohup setsid` with `ON_BOX=1`, which makes the script run its
"remote" commands locally on the box. A failed step does not stop the next one. `wait` (block until `candidates-train`
ends) is useful as a first step when a training job is already running. The box keeps billing after the queue ends:
`pull`, then `down`.

## 10. Plots and the Qwen3 comparison

```bash
bash brev/run_on_brev.sh pull
uv run --with tensorboard --with matplotlib --with numpy python scripts/candidates/plot_results.py
uv run --with matplotlib --with numpy python scripts/candidates/compare_models.py --with-base
```

`plot_results.py` writes `results/plots/`: `training_curves.png` (GRPO loss and mean training reward per step, one
line per model), `test_results.png` (mean test reward and near-oracle rate per method, 95% bootstrap CI over the 8
test materials), `test_per_material.png` and `summary.csv`. The oracle (1.0 by definition) is left out.
`compare_models.py` adds `results/plots/compare_models.png`. The copies in `docs/figures/` are what the README shows.

A small dense Qwen3 model runs through the same pipeline (same data, GRPO settings, LoRA rank 32, scorer) with `MODEL=`:

```bash
MODEL=qwen3-4b caffeinate -i bash brev/run_on_brev.sh all     # also qwen3-1.7b, qwen3-8b
MODEL=qwen3-4b bash brev/run_on_brev.sh pull                  # its TensorBoard logs; results/ holds every model
```

`all` downloads the model, converts it to Megatron (a few minutes), runs the base rollouts, trains, merges the LoRA,
runs the GRPO rollouts and prints one table with every model run so far (`candidates_qwen3-4b_base_test`, ...). Qwen
uses [`training/configs/grpo_active_space_candidates_qwen3.yaml`](../training/configs/grpo_active_space_candidates_qwen3.yaml)
(inherits the Lightning config; EP1, `hermes` tool parser, `deepseek_r1` reasoning parser) and its own checkpoint/log
dirs, so it never overwrites the Lightning run.

---

## Tuning knobs

All in [`training/configs/grpo_active_space_candidates_lightning35.yaml`](../training/configs/grpo_active_space_candidates_lightning35.yaml),
or as Hydra overrides after `training/launch_grpo_candidates.sh`:

| Override                                                        | Default | When to change                                                                                          |
| --------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `grpo.max_num_steps`                                            | 32      | More steps once the smoke test and first curve look sane                                                |
| `grpo.num_prompts_per_step` / `grpo.num_generations_per_prompt` | 4 / 12  | More generations → lower-variance advantages, more time per step                                        |
| `policy.megatron_cfg.peft.dim` / `alpha`                        | 32 / 64 | Larger rank = more capacity, more memory                                                                |
| `policy.megatron_cfg.optimizer.lr`                              | 3e-5    | v1's 5e-6 left reward flat (grad norm ~0.1); lower if it collapses                                      |
| `policy.generation.vllm_cfg.gpu_memory_utilization`             | 0.5     | Lower on OOM during the training phase                                                                  |
| `policy.max_total_sequence_length`                              | 8192    | Lower to save memory if generations are short                                                           |
| `cluster.gpus_per_node`, TP/EP                                  | 2, 2, 2 | Match the box; e.g. 4 GPUs → `cluster.gpus_per_node=4 policy.megatron_cfg.expert_model_parallel_size=4` |

Changing the data (split, β): edit `TEST_MATERIALS` / filters in `scripts/candidates/build_env_data.py`, re-run
`python scripts/candidates/build_env_data.py` (reads `data/phasecraft_tables/dzvp.csv`), then
`pytest resources_servers/active_space_candidates/tests`. `all` re-runs the baseline automatically when the data is
newer than the saved base rollouts.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `brev create`: type "not recognized" / out of capacity | Pick another type from `brev search --gpu-name A100 --min-vram 80` and set `BREV_TYPE` |
| `could not reach active-space-gpu` | `brev ls` — box must be RUNNING / READY; `brev refresh`. A box created before the rename is called `active-space-mini`: set `BREV_INSTANCE=active-space-mini` |
| `docker pull nvcr.io/...` unauthorized | Log the box into NGC: `brev shell`, then `docker login nvcr.io` (user `$oauthtoken`, password = NGC API key) |
| CUDA OOM in smoke/train (e.g. vLLM `wake_up` → `cumem_allocator.cpp: out of memory`) | The frozen LoRA base (~31 GB/GPU) stays on the GPU while vLLM generates, so vLLM's share must fit beside it (default is 0.5). In order: `policy.generation.vllm_cfg.gpu_memory_utilization=0.45`, `policy.max_total_sequence_length=6144`, `grpo.num_generations_per_prompt=4`; else use a 4× 80GB box with `cluster.gpus_per_node=4` |
| `ray.exceptions.OutOfMemoryError ... MegatronPolicyWorker.__init__` (host RAM, ~97% of 147 GB) | The HF → Megatron conversion ran inside training, on top of colocated vLLM's ~60 GB host-RAM weight backup. `bash brev/run_on_brev.sh convert` does it once beforehand (smoke/train/all call it automatically) and caches it in `~/hf/nemo_rl` |
| `AssertionError: Expected mapping for adapter base 'mtp.layers.0.eh_proj.weight'` (in `prepare_refit_info`) | LoRA was put on Lightning's multi-token-prediction layer, which Megatron Bridge can't map back to HF for vLLM. The config sets `mtp_num_layers: 0` (plus the other `mtp_*` fields) to drop that layer |
| `AssertionError` at `assert len(self.per_param_grad_ready_counts) == len(self.params)` (in `train` → `zero_grad_buffer`) | Some LoRA adapters got no gradient in the first backward, which Megatron's overlapped grad reduce forbids. The config sets `distributed_data_parallel_config.overlap_grad_reduce: false` (no cost: data-parallel size is 1 on 2 GPUs) |
| vLLM fails on `--reasoning-parser nemotron_v3` | Image too old; use a newer `VLLM_IMAGE=vllm/vllm-openai:<tag>` |
| Baseline `valid` ≈ 0 | Check `logs/gym_candidates_servers_base.log`; often `MAX_OUTPUT_TOKENS` too small for the reasoning trace |
| `collation produced no train.jsonl` | Gym version mismatch; see the collate command in `training/launch_grpo_candidates.sh` and run it by hand in `~/RL/3rdparty/Gym-workspace/Gym` |
| Want a fresh NeMo RL venv build | `NRL_FORCE_REBUILD_VENVS=true bash brev/run_on_brev.sh smoke` |
| smoke/train stopped by the watcher | It stops a job ~5 min after a fatal error (Ray actor death, CUDA OOM, NCCL) or after `STALL_MIN` (45) silent minutes. Raise `STALL_MIN` if a slow model load trips it; `FAILFAST=0` disables both |

---

## Files

| Path | Role |
| --- | --- |
| `resources_servers/active_space_candidates/app.py` | NeMo Gym resources server: `get_structure`, `get_orbitals`, `submit_active_space`, `verify` |
| `resources_servers/active_space_candidates/configs/active_space_candidates.yaml` | Gym config: server + `active_space_candidates_simple_agent` (max 6 steps); train dataset only. `active_space_candidates_eval.yaml` adds the eval-only `environment_servers` wrapper (absent from NeMo RL's pinned Gym) |
| `resources_servers/active_space_candidates/data/` | `orbitals.json` (agent-visible), `scores.json` (verify-only), `train.jsonl`, `test.jsonl` |
| `resources_servers/active_space_candidates/tests/test_app.py` | Environment, routing, leakage and config-shape tests |
| `active_space/prompts_candidates.py`, `scripts/candidates/build_env_data.py` | Prompts and tool schemas; data builder from the DZVP CSV |
| `training/configs/grpo_active_space_candidates_lightning35.yaml` | GRPO + LoRA config, inherits NeMo RL's `grpo_workplace_assistant_nemotron_nano_v2_9b.yaml` |
| `training/configs/grpo_active_space_candidates_qwen3.yaml` | Qwen3 comparison config (inherits the Lightning one; model set by `MODEL=` in `run_on_brev.sh`) |
| `training/launch_grpo_candidates.sh` | Copies server/configs into NeMo RL, collates train data, launches `run_grpo_nemo_gym.py` (`PREPARE_ONLY=1` stops before training) |
| `training/convert_hf_to_megatron.py` | One-off HF → Megatron conversion before training (avoids a host-RAM OOM on 2 GPUs) |
| `scripts/candidates/run_eval.sh`, `scripts/candidates/evaluate.py` | Rollouts on the 8 test materials against any served model; scoring vs. random and two no-model rules |
| `scripts/candidates/plot_results.py`, `scripts/candidates/compare_models.py` | Training curves and test-result plots for every model, oracle excluded |
| `scripts/candidates/mock_policy_server.py` | Scripted OpenAI-compatible fake model for GPU-free end-to-end tests |
| `brev/run_on_brev.sh`, `brev/BREV_TOKEN.md` | Brev driver and safe credential handling |

## Caveats

- 32 GRPO steps over 16 train materials is a hackathon-scale run, not a converged result; expect noisy curves.
- The DZVP table has per-candidate summaries (size, energy window, aggregate atomic character), not per-orbital
  energies, which is why this environment has its own `app.py` rather than reusing `resources_servers/active_space`.
- `ref_cas` is shown to the agent for context only; it is a classical reference, not the oracle.
- The test split is intentionally not declared in the Gym config, so it can never be collated into training.
