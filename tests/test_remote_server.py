"""Tests for Remote Target Server Verification Endpoints."""

import pytest
import os
import asyncio
from httpx import AsyncClient, ASGITransport

# Set simulation mode for testing
os.environ["SPECROUTE_SIMULATE_TARGET"] = "true"
from server.target_server import app, startup_event


@pytest.mark.asyncio
async def test_server_endpoints():
    await startup_event()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"

        # 2. Tokenizer check
        r = await client.post("/v1/speculative/tokenizer_check", json={
            "tokenizer_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "vocab_size": 151646,
            "test_phrase": "The quick brown fox jumps over the lazy dog.",
            "test_token_ids": [],
        })
        assert r.status_code == 200

        # 3. Session Init
        init_r = await client.post("/v1/speculative/session/init", json={
            "session_id": "test_remote_sess",
            "prompt_token_ids": [100, 200, 300],
            "temperature": 0.0,
            "top_p": 0.95,
        })
        assert init_r.status_code == 200
        init_data = init_r.json()
        assert init_data["status"] == "ok"
        assert "first_token_id" in init_data

        # 4. Verify Draft
        verify_r = await client.post("/v1/speculative/verify", json={
            "session_id": "test_remote_sess",
            "step_index": 1,
            "draft_tokens": [10, 20, 30, 40],
            "temperature": 0.0,
        })
        assert verify_r.status_code == 200
        verify_data = verify_r.json()
        assert "accepted_tokens" in verify_data
        assert "corrective_token" in verify_data
        assert verify_data["acceptance_rate"] >= 0.0

        # 5. Session Close
        close_r = await client.post("/v1/speculative/session/close", json={
            "session_id": "test_remote_sess",
        })
        assert close_r.status_code == 200
        assert close_r.json()["freed"] is True


if __name__ == "__main__":
    asyncio.run(test_server_endpoints())
    print("All remote server endpoint tests passed!")

