"""Remote Target Model Verification Server.

Provides high-throughput speculative decoding verification endpoints over
both low-latency WebSockets and standard HTTP POST APIs.
"""

import os
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from common.protocol import (
    TokenizerCheckRequest,
    TokenizerCheckResponse,
    SessionInitRequest,
    SessionInitResponse,
    VerifyRequest,
    VerifyResponse,
    SessionCloseRequest,
    SessionCloseResponse,
)
from common.logger import get_logger
from server.verifier import SpeculativeTargetVerifier
from config.settings import config

logger = get_logger("TargetServer")

app = FastAPI(
    title="SpecRoute Target Verification Server",
    description="Remote Large Language Model host for hybrid speculative decoding verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global verifier instance
verifier: Optional[SpeculativeTargetVerifier] = None


@app.on_event("startup")
async def startup_event():
    global verifier
    use_sim = os.environ.get("SPECROUTE_SIMULATE_TARGET", "false").lower() in ("true", "1")
    # If explicitly requested or no GPU available, simulation can be chosen
    logger.info("Initializing Speculative Target Verifier on server startup...")
    verifier = SpeculativeTargetVerifier(
        model_id=config.models.target_model_id,
        tokenizer_id=config.models.tokenizer_id,
        device="auto",
        use_simulation=use_sim,
    )
    logger.info("Target Verifier ready for incoming client verification requests.")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SpecRoute Target Verification Host",
        "model_id": config.models.target_model_id,
        "device": verifier.device if verifier else "unknown",
        "simulation_mode": verifier.use_simulation if verifier else True,
        "active_sessions": len(verifier.kv_manager._sessions) if verifier else 0,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": config.models.target_model_id,
                "object": "model",
                "owned_by": "specroute-remote",
            }
        ],
    }


@app.post("/v1/speculative/tokenizer_check", response_model=TokenizerCheckResponse)
@app.get("/v1/speculative/tokenizer_check", response_model=TokenizerCheckResponse)
async def check_tokenizer(req: Optional[TokenizerCheckRequest] = None):
    if req is None:
        req = verifier.tokenizer.generate_check_request()
    return verifier.tokenizer.validate_check_request(req)


@app.post("/v1/speculative/session/init", response_model=SessionInitResponse)
async def init_session(req: SessionInitRequest):
    if not verifier:
        raise HTTPException(status_code=503, detail="Verifier not initialized")
    return verifier.init_session(req)


@app.post("/v1/speculative/verify", response_model=VerifyResponse)
async def verify_speculative_draft(req: VerifyRequest):
    if not verifier:
        raise HTTPException(status_code=503, detail="Verifier not initialized")
    return verifier.verify_draft(req)


@app.post("/v1/speculative/session/close", response_model=SessionCloseResponse)
async def close_session(req: SessionCloseRequest):
    if not verifier:
        raise HTTPException(status_code=503, detail="Verifier not initialized")
    freed = verifier.kv_manager.remove_session(req.session_id)
    return SessionCloseResponse(session_id=req.session_id, freed=freed)


@app.websocket("/v1/speculative/ws")
async def speculative_websocket_endpoint(websocket: WebSocket):
    """Ultra-low latency persistent bidirectional socket for speculative decoding."""
    await websocket.accept()
    logger.info("Client connected to speculative WebSocket channel.")

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            msg_type = data.get("type")
            payload = data.get("data", {})

            if msg_type == "init":
                req = SessionInitRequest(**payload)
                resp = verifier.init_session(req)
                await websocket.send_text(json.dumps({"type": "init_response", "data": resp.model_dump()}))

            elif msg_type == "verify":
                req = VerifyRequest(**payload)
                resp = verifier.verify_draft(req)
                await websocket.send_text(json.dumps({"type": "verify_response", "data": resp.model_dump()}))

            elif msg_type == "check":
                req = TokenizerCheckRequest(**payload)
                resp = verifier.tokenizer.validate_check_request(req)
                await websocket.send_text(json.dumps({"type": "check_response", "data": resp.model_dump()}))

            elif msg_type == "close":
                req = SessionCloseRequest(**payload)
                freed = verifier.kv_manager.remove_session(req.session_id)
                await websocket.send_text(json.dumps({"type": "close_response", "data": {"session_id": req.session_id, "freed": freed}}))

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            else:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Unknown message type '{msg_type}'"}))

    except WebSocketDisconnect:
        logger.info("Client disconnected from speculative WebSocket channel.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close()
        except Exception:
            pass


def start_server(host: str = "0.0.0.0", port: int = 8001):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()

