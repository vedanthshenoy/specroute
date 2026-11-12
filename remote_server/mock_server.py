"""Lightweight mock target server for local testing.

Returns plausible verification responses without loading a real model.
Useful for testing the full pipeline before connecting to Colab.
"""

import logging
import random
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("specroute.mock")

app = FastAPI(title="SpecRoute Mock Target Server")


class SessionInitRequest(BaseModel):
    session_id: str
    prompt_token_ids: list[int]
    tokenizer_name: str


class VerifyRequest(BaseModel):
    session_id: str
    draft_token_ids: list[int]
    draft_probs: list[float]
    cache_position: int


sessions: dict[str, int] = {}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True, "mock": True}


@app.post("/api/session/init")
async def session_init(req: SessionInitRequest):
    sessions[req.session_id] = len(req.prompt_token_ids)
    log.info("Mock session %s initialized with %d prompt tokens", req.session_id, len(req.prompt_token_ids))
    return {
        "session_id": req.session_id,
        "ready": True,
        "cache_position": len(req.prompt_token_ids),
    }


@app.post("/api/verify")
async def verify(req: VerifyRequest):
    # Simulate: accept most tokens with some random rejections
    n = len(req.draft_token_ids)
    accepted = 0
    for i in range(n):
        if random.random() < 0.75:  # 75% acceptance rate
            accepted += 1
        else:
            break

    # Use last draft token as corrective if all accepted, otherwise pick the draft token at rejection
    if accepted == n:
        corrective = req.draft_token_ids[-1]
    else:
        corrective = req.draft_token_ids[accepted]

    cache_pos = req.cache_position + accepted + 1
    sessions[req.session_id] = cache_pos

    log.info("[Mock Speculation] %d/%d accepted | corrective=%d", accepted, n, corrective)

    return {
        "accepted_count": accepted,
        "corrective_token_id": corrective,
        "accepted_token_ids": req.draft_token_ids[:accepted],
        "target_probs": [0.5] * n,
        "new_cache_position": cache_pos,
    }


if __name__ == "__main__":
    log.info("Starting mock target server on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
