"""OpenAI-compatible API server wrapping the speculative decoding orchestrator.

Exposes /v1/chat/completions with SSE streaming, suitable for LibreChat integration.
"""

import asyncio
import json
import logging
import time
import uuid

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from local_engine.orchestrator import SpeculativeOrchestrator

log = logging.getLogger("specroute.api")
app = FastAPI(title="SpecRoute API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator: SpeculativeOrchestrator | None = None


def init_orchestrator(config: dict):
    global orchestrator
    orchestrator = SpeculativeOrchestrator(config)


@app.get("/v1/models")
async def list_models():
    model_name = orchestrator.config["api_server"]["model_name"]
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "specroute",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    session_id = str(uuid.uuid4())
    model_name = orchestrator.config["api_server"]["model_name"]

    if stream:
        return StreamingResponse(
            _stream_response(messages, session_id, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: collect full response
    try:
        full_text = ""
        async for chunk in orchestrator.generate_stream(messages, session_id):
            full_text += chunk
    except Exception as e:
        log.exception("Generation failed")
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": f"chatcmpl-{session_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def _stream_response(messages: list[dict], session_id: str, model_name: str):
    """Yield SSE events in OpenAI streaming format."""
    # Initial chunk with role
    yield _sse(
        {
            "id": f"chatcmpl-{session_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
    )

    async for text_chunk in orchestrator.generate_stream(messages, session_id):
        yield _sse(
            {
                "id": f"chatcmpl-{session_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None,
                    }
                ],
            }
        )

    # Final chunk
    yield _sse(
        {
            "id": f"chatcmpl-{session_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }
    )
    yield "data: [DONE]\n\n"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
