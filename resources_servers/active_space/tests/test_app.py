import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient

# Import app.py by path under a unique name: a top-level `resources_servers` package here would shadow the one nemo-gym
# installs, and a plain `import app` would collide with the other server's app.py when both test suites run together.
_spec = importlib.util.spec_from_file_location("active_space_app", Path(__file__).resolve().parent.parent / "app.py")
app = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)
ActiveSpaceResourcesServer = app.ActiveSpaceResourcesServer
ActiveSpaceResourcesServerConfig = app.ActiveSpaceResourcesServerConfig
ActiveSpaceVerifyRequest = app.ActiveSpaceVerifyRequest
GetOrbitalsRequest = app.GetOrbitalsRequest
SubmitRequest = app.SubmitRequest

MID = "mp-0001"

ORBITALS = {
    MID: {
        "material_id": MID,
        "name": "XO",
        "structure": {"spacegroup": "Fm-3m"},
        "calc": {},
        "orbitals": [
            {"index": -1, "energy_eV": 0.0, "occupation": 2, "mp2_occupation": 1.9, "character": {"X-3d": 0.9}},
            {"index": 0, "energy_eV": 1.0, "occupation": 0, "mp2_occupation": 0.1, "character": {"X-3d": 0.8}},
        ],
        "localized_counts": {},
        "menu": {"frontier": [2, 4]},
        "members": {"frontier:2": {"occ": [-1], "vir": [0]}, "frontier:4": {"occ": [-2, -1], "vir": [0, 1]}},
    }
}
SCORES = {
    "settings": {"metric": "mp2_pair", "beta": 0.05, "near_oracle_tol": 0.05},
    "materials": {
        MID: {
            "oracle": "frontier:4",
            "best_score": -0.1,
            "worst_score": -0.5,
            "candidates": {"frontier:2": {"score": -0.5}, "frontier:4": {"score": -0.1}},
        }
    },
}


@pytest.fixture
def server(tmp_path):
    (tmp_path / "orbitals.json").write_text(json.dumps(ORBITALS))
    (tmp_path / "scores.json").write_text(json.dumps(SCORES))
    config = ActiveSpaceResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="active_space",
        orbitals_path=str(tmp_path / "orbitals.json"), scores_path=str(tmp_path / "scores.json"),
    )
    return ActiveSpaceResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(*calls):
    output = [
        {"id": f"c{i}", "call_id": f"c{i}", "name": name, "arguments": json.dumps(args), "type": "function_call",
         "status": "completed"}
        for i, (name, args) in enumerate(calls)
    ]
    return NeMoGymResponse(id="r", created_at=0, model="m", object="response", output=output,
                           parallel_tool_calls=False, tool_choice="auto", tools=[])


def _verify(server, *calls, material_id=MID):
    req = ActiveSpaceVerifyRequest(responses_create_params={"input": "x"}, response=_response(*calls),
                                   material_id=material_id)
    return asyncio.run(server.verify(req))


def test_oracle_gets_full_reward(server):
    r = _verify(server, ("submit_active_space", {"material_id": MID, "selection": "frontier", "n_orbitals": 4}))
    assert r.reward == 1.0 and r.near_oracle == 1.0 and r.chosen == "frontier:4"


def test_worst_gets_zero_but_counts_as_valid(server):
    r = _verify(server, ("submit_active_space", {"material_id": MID, "selection": "frontier", "n_orbitals": 2}))
    assert r.reward == 0.0 and r.valid_submission == 1.0


def test_off_menu_and_missing_answers_get_zero(server):
    assert _verify(server, ("submit_active_space", {"material_id": MID, "selection": "frontier", "n_orbitals": 6})).reward == 0.0
    assert _verify(server, ("get_orbitals", {"material_id": MID, "candidate": None})).reward == 0.0
    assert _verify(server, ("submit_active_space", {"material_id": "mp-9", "selection": "frontier", "n_orbitals": 4})).reward == 0.0


def test_first_valid_submission_counts_after_a_corrected_error(server):
    r = _verify(
        server,
        ("submit_active_space", {"material_id": MID, "selection": "X-3d", "n_orbitals": 4}),
        ("submit_active_space", {"material_id": MID, "selection": "frontier", "n_orbitals": 4}),
        ("submit_active_space", {"material_id": MID, "selection": "frontier", "n_orbitals": 2}),
    )
    assert r.reward == 1.0 and r.invalid_submissions == 1


def test_tools_never_reveal_scores(server):
    full = asyncio.run(server.get_orbitals(GetOrbitalsRequest(material_id=MID, candidate=None))).model_dump_json()
    sub = asyncio.run(server.submit_active_space(SubmitRequest(material_id=MID, selection="frontier", n_orbitals=4)))
    assert "score" not in full and "reward" not in full and "oracle" not in full
    assert sub.ok and "score" not in sub.model_dump_json()


def test_submit_rejects_off_menu_with_menu_hint(server):
    sub = asyncio.run(server.submit_active_space(SubmitRequest(material_id=MID, selection="frontier", n_orbitals=3)))
    assert not sub.ok and "frontier" in sub.error
