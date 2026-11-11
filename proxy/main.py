"""OpenAI-Compatible Reverse Proxy API for LibreChat.

Exposes standard /v1/chat/completions and /v1/models endpoints, routing
chat completion requests into the local hybrid speculative decoding engine.
"""

import time
import json
import uuid
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from common.logger import get_logger
from common.tokenizer_sync import SynchronizedTokenizer
from engine.local_draft import create_draft_engine
from engine.remote_client import RemoteTargetClient
from engine.speculative_decoder import SpeculativeDecodingOrchestrator
from config.settings import config

logger = get_logger("LibreChatProxy")

app = FastAPI(
    title="SpecRoute OpenAI-Compatible Proxy for LibreChat",
    description="Hybrid Local-Cloud Speculative Decoding proxy server.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator components
orchestrator: Optional[SpeculativeDecodingOrchestrator] = None
tokenizer: Optional[SynchronizedTokenizer] = None
remote_client: Optional[RemoteTargetClient] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "specroute-hybrid-qwen"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


@app.on_event("startup")
async def startup_proxy():
    global orchestrator, tokenizer, remote_client
    logger.info("Starting SpecRoute Local Proxy...")
    tokenizer = SynchronizedTokenizer(config.models.tokenizer_id)
    draft_engine = create_draft_engine(config, tokenizer)
    remote_client = RemoteTargetClient(config.remote_server)

    orchestrator = SpeculativeDecodingOrchestrator(
        config=config,
        tokenizer=tokenizer,
        draft_engine=draft_engine,
        remote_client=remote_client,
    )
    logger.info("SpecRoute Proxy successfully initialized and listening for LibreChat requests.")


@app.on_event("shutdown")
async def shutdown_proxy():
    global remote_client
    if remote_client:
        await remote_client.close()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "SpecRoute Local Orchestrator Proxy",
        "model": config.local_proxy.model_name,
        "draft_model": config.models.draft_model_id,
        "target_model": config.models.target_model_id,
        "target_url": config.remote_server.base_url,
        "backend": config.models.local_engine_backend,
    }


@app.get("/v1/models")
async def list_models():
    """Returns OpenAI-compatible model listing consumed by LibreChat."""
    return {
        "object": "list",
        "data": [
            {
                "id": config.local_proxy.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "specroute",
                "permission": [],
                "root": config.local_proxy.model_name,
                "parent": None,
            },
            {
                "id": "specroute-hybrid-qwen",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "specroute",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with speculative streaming."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created_ts = int(time.time())
    messages_dicts = [m.model_dump() for m in req.messages]

    if req.stream:
        # Streaming response (SSE)
        async def event_generator():
            try:
                # 1. Role chunk
                role_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"

                # 2. Speculatively verified content deltas
                async for delta_text in orchestrator.generate_stream(
                    messages=messages_dicts,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_new_tokens=req.max_tokens,
                ):
                    delta_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(delta_chunk)}\n\n"

                # 3. Final stop chunk
                stop_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error streaming speculative tokens: {e}", exc_info=True)
                err_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n[Speculative Generation Error: {e}]"},
                            "finish_reason": "error",
                        }
                    ],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    text_deltas = []
    async for delta in orchestrator.generate_stream(
        messages=messages_dicts,
        temperature=req.temperature,
        top_p=req.top_p,
        max_new_tokens=req.max_tokens,
    ):
        text_deltas.append(delta)

    full_content = "".join(text_deltas)
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(messages_dicts),
            "completion_tokens": len(text_deltas),
            "total_tokens": len(messages_dicts) + len(text_deltas),
        },
    }


def start_proxy(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_proxy()

