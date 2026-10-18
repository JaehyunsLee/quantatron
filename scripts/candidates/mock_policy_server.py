"""A scripted stand-in for the vLLM endpoint, so scripts/candidates/run_eval.sh can be tested end to end with no GPU.

It speaks just enough of the OpenAI chat-completions API for NeMo Gym's vllm_model server:
turn 1 calls get_orbitals, turn 2 calls submit_active_space with a pseudo-random menu entry.

    python scripts/candidates/mock_policy_server.py --port 8000 [--always-oracle]
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request

DATA = Path(__file__).resolve().parents[2] / "resources_servers" / "active_space_candidates" / "data"
ORBITALS = json.loads((DATA / "orbitals.json").read_text())
SCORES = json.loads((DATA / "scores.json").read_text())["materials"]
app = FastAPI()
ALWAYS_ORACLE = False
MODEL = "mock-lightning"


def _text(content) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _tool_call(name: str, args: dict, i: int) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "mock"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    msgs = body["messages"]
    prompt = " ".join(_text(m.get("content")) for m in msgs if m["role"] in ("system", "user"))
    mid = next(m for m in re.findall(r"mp-\d+|mvc-\d+", prompt) if m in ORBITALS)
    already_looked = any(m["role"] == "tool" for m in msgs)
    if not already_looked:
        call = _tool_call("get_orbitals", {"material_id": mid, "candidate_id": None}, 0)
    else:
        menu = list(SCORES[mid]["candidates"])
        pick = SCORES[mid]["oracle"] if ALWAYS_ORACLE else menu[int(hashlib.md5(str(msgs).encode()).hexdigest(), 16) % len(menu)]
        call = _tool_call("submit_active_space", {"material_id": mid, "candidate_id": pick}, 1)
    return {
        "id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()), "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "tool_calls",
                     "message": {"role": "assistant", "content": None, "tool_calls": [call]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--always-oracle", action="store_true")
    a = p.parse_args()
    ALWAYS_ORACLE = a.always_oracle
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
