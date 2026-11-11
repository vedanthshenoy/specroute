"""End-to-End integration tests for SpecRoute Local Proxy & LibreChat protocol."""

import os
import json
import pytest
import asyncio
from httpx import AsyncClient, ASGITransport

os.environ["SPECROUTE_SIMULATE_TARGET"] = "true"

from server.target_server import app as target_app, startup_event as server_startup
from proxy.main import app as proxy_app, startup_proxy
import proxy.main as proxy_module


@pytest.mark.asyncio
async def test_end_to_end_speculative_proxy():
    # 1. Initialize server and proxy
    await server_startup()
    await startup_proxy()

    # Route remote_client calls in proxy to the in-process target_app ASGI transport
    target_transport = ASGITransport(app=target_app)
    mock_http_client = AsyncClient(transport=target_transport, base_url="http://target-server")
    
    # Override proxy's remote client to route through in-process target server
    proxy_module.remote_client._http_client = mock_http_client
    proxy_module.remote_client._use_ws = False  # Use HTTP transport for in-process test

    proxy_transport = ASGITransport(app=proxy_app)
    async with AsyncClient(transport=proxy_transport, base_url="http://proxy") as client:
        # Check Proxy Health
        h = await client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "healthy"

        # Check Models Endpoint (as requested by LibreChat)
        m = await client.get("/v1/models")
        assert m.status_code == 200
        models_data = m.json()["data"]
        model_ids = [item["id"] for item in models_data]
        assert "specroute-hybrid-qwen" in model_ids

        # Test Non-streaming Chat Completion
        req_non_stream = {
            "model": "specroute-hybrid-qwen",
            "messages": [
                {"role": "user", "content": "What is speculative decoding?"}
            ],
            "stream": False,
            "max_tokens": 30,
        }
        res = await client.post("/v1/chat/completions", json=req_non_stream)
        assert res.status_code == 200
        data = res.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        assert data["choices"][0]["finish_reason"] == "stop"

        # Test Streaming Chat Completion (SSE Format for LibreChat)
        req_stream = {
            "model": "specroute-hybrid-qwen",
            "messages": [
                {"role": "user", "content": "Explain hybrid inference advantages."}
            ],
            "stream": True,
            "max_tokens": 30,
        }
        stream_res = await client.post("/v1/chat/completions", json=req_stream)
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers.get("content-type", "")

        lines = stream_res.text.strip().split("\n")
        sse_events = [line for line in lines if line.startswith("data: ")]
        assert len(sse_events) >= 2
        assert sse_events[-1] == "data: [DONE]"

        # Parse SSE data payloads
        streamed_text = ""
        for ev in sse_events[:-1]:
            payload = json.loads(ev[len("data: "):])
            delta = payload["choices"][0]["delta"]
            if "content" in delta:
                streamed_text += delta["content"]

        assert len(streamed_text) > 0
        print(f"\n[Test Output] Streamed text successfully received:\n'{streamed_text}'")


if __name__ == "__main__":
    asyncio.run(test_end_to_end_proxy())
    print("All End-to-End tests passed successfully!")

