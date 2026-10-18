"""NeMo Gym resources server: candidate-level active-space selection, built from data/phasecraft_tables/dzvp.csv
by scripts/candidates/build_env_data.py.

Same contract as resources_servers/active_space/app.py: read-only lookups, a submit tool that
only validates, and a verify() that is the only place scores live.

    get_structure(material_id)
    get_orbitals(material_id, candidate_id)      # candidate_id optional; null = full menu
    submit_active_space(material_id, candidate_id)

verify() returns 0 for a missing/invalid/off-menu answer, else the min-max normalized score
(score - worst) / (best - worst) in [0, 1], for the FIRST valid submit_active_space call.
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


class ActiveSpaceCandidatesConfig(BaseResourcesServerConfig):
    orbitals_path: str = str(DATA_DIR / "orbitals.json")
    scores_path: str = str(DATA_DIR / "scores.json")


class GetStructureRequest(BaseModel):
    material_id: str


class GetOrbitalsRequest(BaseModel):
    material_id: str
    candidate_id: Optional[str] = None


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    material_id: str
    candidate_id: Any = None


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    ok: bool
    error: Optional[str] = None


class CandidatesVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    material_id: str


class CandidatesVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    chosen: Optional[str] = None
    oracle: Optional[str] = None
    valid_submission: float = 0.0
    invalid_submissions: int = 0


class CandidatesVerifier:
    """Pure scoring logic, kept separate from the web server so it is unit-testable."""

    def __init__(self, scores: dict):
        self.scores = scores

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
            cid = str(args.get("candidate_id", "")).strip()
            if args.get("material_id") == material_id and cid in menu:
                return cid, invalid
            invalid += 1
        return None, invalid

    def score(self, response_output, material_id: str) -> dict:
        entry = self.scores["materials"].get(material_id)
        cid, invalid = self.first_valid_submission(response_output, material_id)
        if entry is None or cid is None:
            return {"reward": 0.0, "chosen": None, "oracle": entry and entry["oracle"],
                    "valid_submission": 0.0, "invalid_submissions": invalid}
        reward = min(1.0, max(0.0, entry["candidates"][cid]["reward"]))
        return {"reward": reward, "chosen": cid, "oracle": entry["oracle"],
                "valid_submission": 1.0, "invalid_submissions": invalid}


class ActiveSpaceCandidatesServer(SimpleResourcesServer):
    config: ActiveSpaceCandidatesConfig

    @cached_property
    def orbitals(self) -> dict:
        return json.loads(Path(self.config.orbitals_path).read_text())

    @cached_property
    def verifier(self) -> CandidatesVerifier:
        return CandidatesVerifier(json.loads(Path(self.config.scores_path).read_text()))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_structure")(self.get_structure)
        app.post("/get_orbitals")(self.get_orbitals)
        app.post(f"/{SUBMIT_TOOL}")(self.submit_active_space)
        return app

    def _entry(self, material_id: str) -> Optional[dict]:
        return self.orbitals.get(material_id.strip())

    def _unknown(self, material_id: str) -> ToolResponse:
        return ToolResponse(ok=False, error=f"Unknown material_id {material_id!r}.")

    async def get_structure(self, body: GetStructureRequest) -> ToolResponse:
        e = self._entry(body.material_id)
        if e is None:
            return self._unknown(body.material_id)
        return ToolResponse(ok=True, material_id=e["material_id"], name=e["name"], formula=e["formula"],
                            n_atoms=e["n_atoms"], gap_eV=e["gap_eV"], ref_cas=e["ref_cas"])

    async def get_orbitals(self, body: GetOrbitalsRequest) -> ToolResponse:
        e = self._entry(body.material_id)
        if e is None:
            return self._unknown(body.material_id)
        if body.candidate_id:
            cand = e["menu"].get(body.candidate_id.strip())
            if cand is None:
                return ToolResponse(ok=False, error=f"{body.candidate_id!r} is not on the menu. Menu: {list(e['menu'])}")
            return ToolResponse(ok=True, candidate_id=body.candidate_id.strip(), **cand)
        return ToolResponse(ok=True, material_id=e["material_id"], menu=e["menu"])

    async def submit_active_space(self, body: SubmitRequest) -> ToolResponse:
        e = self._entry(body.material_id)
        if e is None:
            return self._unknown(body.material_id)
        cid = str(body.candidate_id or "").strip()
        if cid not in e["menu"]:
            return ToolResponse(ok=False, error=f"{cid!r} is not on the menu for {e['material_id']}. Menu: {list(e['menu'])}")
        # Deliberately no score feedback: the reward is only computed in verify().
        return ToolResponse(ok=True, accepted=cid, message="Submission recorded. This is your final answer.")

    async def verify(self, body: CandidatesVerifyRequest) -> CandidatesVerifyResponse:
        result = self.verifier.score(body.response.output, body.material_id)
        return CandidatesVerifyResponse(**body.model_dump(), **result)


if __name__ == "__main__":
    ActiveSpaceCandidatesServer.run_webserver()
