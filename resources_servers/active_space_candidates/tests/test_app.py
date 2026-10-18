"""Tests for the active_space_candidates server and its shipped data/configs, no GPU or served model needed.

    python -m pytest resources_servers/active_space_candidates/tests -q
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient

SERVER = Path(__file__).resolve().parent.parent
ROOT = SERVER.parent.parent

# Import app.py by path under a unique name: a top-level `resources_servers` package here would shadow the one nemo-gym
# installs, and a plain `import app` would collide with the other server's app.py when both test suites run together.
_spec = importlib.util.spec_from_file_location("active_space_candidates_app", SERVER / "app.py")
app = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)
ActiveSpaceCandidatesConfig = app.ActiveSpaceCandidatesConfig
ActiveSpaceCandidatesServer = app.ActiveSpaceCandidatesServer
CandidatesVerifyRequest = app.CandidatesVerifyRequest
GetOrbitalsRequest = app.GetOrbitalsRequest
SubmitRequest = app.SubmitRequest

MID = "mp-0001"
ORBITALS = {
    MID: {
        "material_id": MID, "name": "XO", "formula": "XO", "n_atoms": 2, "gap_eV": 1.0,
        "ref_cas": "(8e,8o)",
        "menu": {
            "c0:2": {"n_electrons": 2, "n_orbitals": 2, "mo_indices": [0, 1], "window_ev": [-1.0, 1.0], "character": {"X-3d": 0.9}},
            "c1:4": {"n_electrons": 4, "n_orbitals": 4, "mo_indices": [-1, 0, 1, 2], "window_ev": [-2.0, 2.0], "character": {"X-3d": 0.8}},
        },
    }
}
SCORES = {
    "settings": {"beta": 0.2},
    "materials": {
        MID: {
            "oracle": "c1:4",
            "best_score": -0.1,
            "worst_score": -0.5,
            "candidates": {
                "c0:2": {"score": -0.5, "reward": 0.0},
                "c1:4": {"score": -0.1, "reward": 1.0},
            },
        }
    },
}


@pytest.fixture
def server(tmp_path):
    (tmp_path / "orbitals.json").write_text(json.dumps(ORBITALS))
    (tmp_path / "scores.json").write_text(json.dumps(SCORES))
    config = ActiveSpaceCandidatesConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="active_space_candidates",
        orbitals_path=str(tmp_path / "orbitals.json"), scores_path=str(tmp_path / "scores.json"),
    )
    return ActiveSpaceCandidatesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(*calls):
    output = [
        {"id": f"c{i}", "call_id": f"c{i}", "name": name, "arguments": json.dumps(args), "type": "function_call",
         "status": "completed"}
        for i, (name, args) in enumerate(calls)
    ]
    return NeMoGymResponse(id="r", created_at=0, model="m", object="response", output=output,
                           parallel_tool_calls=False, tool_choice="auto", tools=[])


def _verify(server, *calls, material_id=MID):
    req = CandidatesVerifyRequest(responses_create_params={"input": "x"}, response=_response(*calls), material_id=material_id)
    return asyncio.run(server.verify(req))


def test_oracle_gets_full_reward(server):
    r = _verify(server, ("submit_active_space", {"material_id": MID, "candidate_id": "c1:4"}))
    assert r.reward == 1.0 and r.chosen == "c1:4"


def test_worst_gets_zero_but_counts_as_valid(server):
    r = _verify(server, ("submit_active_space", {"material_id": MID, "candidate_id": "c0:2"}))
    assert r.reward == 0.0 and r.valid_submission == 1.0


def test_off_menu_and_missing_answers_get_zero(server):
    assert _verify(server, ("submit_active_space", {"material_id": MID, "candidate_id": "c9:99"})).reward == 0.0
    assert _verify(server, ("get_orbitals", {"material_id": MID, "candidate_id": None})).reward == 0.0


def test_tools_never_reveal_scores(server):
    full = asyncio.run(server.get_orbitals(GetOrbitalsRequest(material_id=MID, candidate_id=None))).model_dump_json()
    sub = asyncio.run(server.submit_active_space(SubmitRequest(material_id=MID, candidate_id="c1:4")))
    assert "score" not in full and "reward" not in full and "oracle" not in full
    assert sub.ok and "score" not in sub.model_dump_json()


# ----------------------------------------------------------------------------- shipped-data regression tests
import yaml  # noqa: E402


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_rows_route_to_the_agent_the_config_defines():
    cfg = yaml.safe_load((SERVER / "configs" / "active_space_candidates.yaml").read_text())
    agents = {k for k, v in cfg.items() if "responses_api_agents" in v}
    for split in ("train", "test"):
        for row in _jsonl(SERVER / "data" / f"{split}.jsonl"):
            assert row["agent_ref"]["name"] in agents


def test_gym_config_uses_a_resolvable_server_layout():
    cfg = yaml.safe_load((SERVER / "configs" / "active_space_candidates.yaml").read_text())
    assert cfg["active_space_candidates"]["resources_servers"]["active_space_candidates"]["entrypoint"] == "app.py"
    assert (SERVER / "app.py").exists() and (SERVER / "requirements.txt").exists()
    train = cfg["active_space_candidates_simple_agent"]["responses_api_agents"]["simple_agent"]["datasets"]
    assert [d["name"] for d in train] == ["train"], "the test split must not be declared to Gym"
    assert (ROOT / train[0]["jsonl_fpath"]).exists()


def test_splits_are_disjoint_and_prompts_do_not_leak_scores():
    train, test = _jsonl(SERVER / "data" / "train.jsonl"), _jsonl(SERVER / "data" / "test.jsonl")
    assert not {r["material_id"] for r in train} & {r["material_id"] for r in test}
    scores = json.loads((SERVER / "data" / "scores.json").read_text())["materials"]
    assert {r["material_id"] for r in test} == {m for m, s in scores.items() if s["split"] == "test"}
    orbitals = json.dumps(json.loads((SERVER / "data" / "orbitals.json").read_text())).lower()
    for row in train + test:
        text = json.dumps(row["responses_create_params"]["input"]).lower()
        assert "toffoli_total" not in text and "is_oracle" not in text
    # The cost estimate (log10_toffolis) is deliberately visible; accuracy and everything derived from it is not.
    for forbidden in ('"score"', '"reward"', '"oracle"', '"accuracy"', "corr_recovered", "toffoli_total", '"toffolis"'):
        assert forbidden not in orbitals
    assert '"log10_toffolis"' in orbitals


def test_grpo_config_has_complete_peft_block_and_never_loads_test_data():
    cfg = yaml.safe_load((ROOT / "training" / "configs" / "grpo_active_space_candidates_lightning35.yaml").read_text())
    peft = cfg["policy"]["megatron_cfg"]["peft"]
    required = {"enabled", "target_modules", "exclude_modules", "dim", "alpha", "dropout", "dropout_position",
                "lora_A_init_method", "lora_B_init_method", "a2a_experimental", "lora_dtype"}
    assert required <= set(peft)
    g = cfg["grpo"]
    assert g["val_period"] == 0 and not g["val_at_start"]
    assert "test" not in json.dumps(cfg["data"])


def test_training_gym_config_only_uses_server_types_the_pinned_gym_has():
    # NeMo RL v0.7.0's Gym has resources_servers / responses_api_agents / responses_api_models but no
    # environment_servers; declaring one there crashes training at Gym spin-up. It belongs in the eval-only config.
    cfg = yaml.safe_load((SERVER / "configs" / "active_space_candidates.yaml").read_text())
    kinds = {kind for v in cfg.values() for kind in v}
    assert kinds <= {"resources_servers", "responses_api_agents", "responses_api_models"}, kinds
    eval_cfg = yaml.safe_load((SERVER / "configs" / "active_space_candidates_eval.yaml").read_text())
    assert "environment_servers" in eval_cfg["active_space_candidates_environment_server"]
