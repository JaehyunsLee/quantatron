"""NeMo Gym resources server: active-space selection for quantum simulation of materials.

Tools (all read-only except the submission, which only validates):
    get_structure(material_id)
    get_orbitals(material_id, candidate)
    submit_active_space(material_id, selection, n_orbitals)

verify() reads scores.json, which no tool exposes. It returns 0 for a missing, invalid or
off-menu answer, else the min-max normalized score
    (score - worst) / (best - worst)   in [0, 1],
for the first valid submit_active_space call in the rollout.
"""

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

DATA_DIR = Path(__file__).parent / "data"
SUBMIT_TOOL = "submit_active_space"


class ActiveSpaceResourcesServerConfig(BaseResourcesServerConfig):
    orbitals_path: str = str(DATA_DIR / "orbitals.json")
    scores_path: str = str(DATA_DIR / "scores.json")


class GetStructureRequest(BaseModel):
    material_id: str


class GetOrbitalsRequest(BaseModel):
    material_id: str
    candidate: Optional[str] = None


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    material_id: str
    selection: str
    n_orbitals: Any = None  # validated by hand so a malformed value is reported, not a 422


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    ok: bool
    error: Optional[str] = None


class ActiveSpaceVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    material_id: str


class ActiveSpaceVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    chosen: Optional[str] = None
    oracle: Optional[str] = None
    valid_submission: float = 0.0
    near_oracle: float = 0.0
    invalid_submissions: int = 0
    n_orbitals_chosen: Optional[int] = None


def candidate_id(selection: Any, n_orbitals: Any) -> Optional[str]:
    try:
        n = int(n_orbitals)
    except (TypeError, ValueError):
        return None
    if isinstance(n_orbitals, float) and not float(n_orbitals).is_integer():
        return None
    return f"{str(selection).strip()}:{n}"


class ActiveSpaceVerifier:
    """Pure scoring logic, separated from the web server so it can be unit-tested and reused offline."""

    def __init__(self, scores: dict):
        self.scores = scores
        self.tol = scores["settings"]["near_oracle_tol"]

    def first_valid_submission(self, response_output, material_id: str) -> tuple[Optional[str], int]:
        invalid = 0
        menu = self.scores["materials"].get(material_id, {}).get("candidates", {})
        for item in response_output:
            if getattr(item, "type", None) != "function_call" or getattr(item, "name", None) != SUBMIT_TOOL:
                continue
            try:
                args = json.loads(item.arguments) if isinstance(item.arguments, str) else dict(item.arguments)
            except (json.JSONDecodeError, TypeError):
                invalid += 1
                continue
            cid = candidate_id(args.get("selection"), args.get("n_orbitals"))
            if args.get("material_id") == material_id and cid in menu:
                return cid, invalid
            invalid += 1
        return None, invalid

    def score(self, response_output, material_id: str) -> dict:
        entry = self.scores["materials"].get(material_id)
        cid, invalid = self.first_valid_submission(response_output, material_id)
        if entry is None or cid is None:
            return {"reward": 0.0, "chosen": None, "oracle": entry and entry["oracle"],
                    "valid_submission": 0.0, "near_oracle": 0.0, "invalid_submissions": invalid,
                    "n_orbitals_chosen": None}
        best, worst = entry["best_score"], entry["worst_score"]
        s = entry["candidates"][cid]["score"]
        reward = (s - worst) / (best - worst) if best > worst else 1.0
        reward = min(1.0, max(0.0, reward))
        return {"reward": reward, "chosen": cid, "oracle": entry["oracle"], "valid_submission": 1.0,
                "near_oracle": float(reward >= 1.0 - self.tol), "invalid_submissions": invalid,
                "n_orbitals_chosen": int(cid.split(":")[1])}


class ActiveSpaceResourcesServer(SimpleResourcesServer):
    config: ActiveSpaceResourcesServerConfig

    @cached_property
    def orbitals(self) -> dict:
        return json.loads(Path(self.config.orbitals_path).read_text())

    @cached_property
    def verifier(self) -> ActiveSpaceVerifier:
        return ActiveSpaceVerifier(json.loads(Path(self.config.scores_path).read_text()))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_structure")(self.get_structure)
        app.post("/get_orbitals")(self.get_orbitals)
        app.post(f"/{SUBMIT_TOOL}")(self.submit_active_space)
        return app

    def _entry(self, material_id: str) -> Optional[dict]:
        return self.orbitals.get(material_id.strip())

    def _unknown(self, material_id: str) -> ToolResponse:
        return ToolResponse(ok=False, error=f"Unknown material_id {material_id!r}. Use the id given in the task, e.g. 'mp-18717'.")

    async def get_structure(self, body: GetStructureRequest) -> ToolResponse:
        e = self._entry(body.material_id)
        if e is None:
            return self._unknown(body.material_id)
        return ToolResponse(ok=True, material_id=e["material_id"], name=e["name"], structure=e["structure"],
                            description=e.get("description"), calculation=e["calc"])

    async def get_orbitals(self, body: GetOrbitalsRequest) -> ToolResponse:
        e = self._entry(body.material_id)
        if e is None:
            return self._unknown(body.material_id)
        if body.candidate:
            members = e["members"].get(body.candidate.strip())
            if members is None:
                return ToolResponse(ok=False, error=f"{body.candidate!r} is not on the menu. Menu: {e['menu']}")
            idx = set(members["occ"]) | set(members["vir"])
            return ToolResponse(ok=True, candidate=body.candidate.strip(),
                                orbitals=[o for o in e["orbitals"] if o["index"] in idx])
        return ToolResponse(ok=True, material_id=e["material_id"], orbitals=e["orbitals"],
                            localized_counts=e["localized_counts"], menu=e["menu"])

    async def submit_active_space(self, body: SubmitRequest) -> ToolResponse:
        e = self._entry(body.material_id)
        if e is None:
            return self._unknown(body.material_id)
        cid = candidate_id(body.selection, body.n_orbitals)
        if cid is None or cid not in e["members"]:
            return ToolResponse(ok=False, error=f"({body.selection!r}, {body.n_orbitals!r}) is not on the menu for "
                                                f"{e['material_id']}. Menu: {e['menu']}")
        # Deliberately no score feedback: the reward is only computed in verify().
        return ToolResponse(ok=True, accepted=cid, message="Submission recorded. This is your final answer.")

    async def verify(self, body: ActiveSpaceVerifyRequest) -> ActiveSpaceVerifyResponse:
        result = self.verifier.score(body.response.output, body.material_id)
        return ActiveSpaceVerifyResponse(**body.model_dump(), **result)


if __name__ == "__main__":
    ActiveSpaceResourcesServer.run_webserver()
