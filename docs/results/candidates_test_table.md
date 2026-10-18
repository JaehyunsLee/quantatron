# active_space_candidates: held-out test results

Output of `python scripts/candidates/evaluate.py results/candidates_*_test.jsonl` on the 8 held-out test materials
(16 rollouts per material at temperature 0.6). `candidates_grpo_test` is the v2 Lightning GRPO run, evaluated from the
newest checkpoint at evaluation time (`bash brev/run_on_brev.sh evaltrained`); the rollout files do not record the step.
The Qwen3-1.7B GRPO run stopped after 2 steps and was not evaluated.

| method                                     | reward | <=5% of oracle | valid |
|--------------------------------------------|--------|----------------|-------|
| oracle                                     |  1.000 |          1.000 |  1.00 |
| random menu entry                          |  0.399 |          0.148 |  1.00 |
| rule: largest candidate                    |  0.741 |          0.500 |  1.00 |
| rule: tightest window/orbital              |  0.693 |          0.250 |  1.00 |
| candidates_base_test                       |  0.404 |          0.078 |  0.98 |
| candidates_base_test (majority)            |  0.534 |          0.125 |  1.00 |
| candidates_grpo_test                       |  0.411 |          0.078 |  0.95 |
| candidates_grpo_test (majority)            |  0.518 |          0.125 |  1.00 |
| candidates_qwen3-1.7b_base_test            |  0.359 |          0.164 |  0.87 |
| candidates_qwen3-1.7b_base_test (majority) |  0.467 |          0.250 |  1.00 |
