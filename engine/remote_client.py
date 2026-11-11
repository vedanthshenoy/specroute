"""Remote Target Server Client.

Manages persistent WebSocket and fallback HTTP keep-alive connections to the
remote target model server running on Google Colab, Kaggle, or cloud GPU.
"""

import json
import asyncio
import httpx
import websockets
from typing import Optional, Dict, Any
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
from common.tokenizer_sync import SynchronizedTokenizer
from config.settings import RemoteServerConfig

logger = get_logger("RemoteClient")


class RemoteTargetClient:
    """High-speed asynchronous client for communicating with the target verification server."""

    def __init__(self, config: RemoteServerConfig):
        self.config = config
        self._http_client: Optional[httpx.AsyncClient] = None
        self._ws = None
        self._ws_lock = asyncio.Lock()
        self._use_ws = config.use_websocket

    async def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.config.auth_token:
                headers["Authorization"] = f"Bearer {self.config.auth_token}"
            self._http_client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                headers=headers,
                timeout=self.config.timeout_seconds,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return self._http_client

    async def connect_websocket(self) -> bool:
        if not self._use_ws:
            return False
        async with self._ws_lock:
            if self._ws is not None and not self._ws.closed:
                return True
            try:
                ws_url = self.config.ws_url
                logger.info(f"Connecting persistent WebSocket to {ws_url}...")
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        ws_url,
                        ping_interval=20,
                        ping_timeout=10,
                        close_timeout=5,
                        max_size=10 * 1024 * 1024,
                    ),
                    timeout=5.0,
                )
                logger.info("WebSocket connection established with remote target server.")
                return True
            except Exception as e:
                logger.warning(f"Could not connect WebSocket ({e}). Using HTTP fallback.")
                self._ws = None
                return False

    async def close(self):
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def check_tokenizer_parity(self, tokenizer: SynchronizedTokenizer) -> bool:
        """Sends verification payload to confirm tokenizers match exactly."""
        req = tokenizer.generate_check_request()
        try:
            # Try via WebSocket if active
            if await self.connect_websocket():
                async with self._ws_lock:
                    await self._ws.send(json.dumps({"type": "check", "data": req.model_dump()}))
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                    resp_data = json.loads(raw)
                    resp = TokenizerCheckResponse(**resp_data.get("data", {}))
                    return tokenizer.verify_remote_parity(resp)
        except Exception as e:
            logger.debug(f"WS tokenizer check failed ({e}), trying HTTP...")

        # Fallback to HTTP
        try:
            client = await self.get_http_client()
            r = await client.post("/v1/speculative/tokenizer_check", json=req.model_dump())
            r.raise_for_status()
            resp = TokenizerCheckResponse(**r.json())
            return tokenizer.verify_remote_parity(resp)
        except Exception as e:
            logger.error(f"Failed to verify tokenizer parity via HTTP ({e})")
            return False

    async def init_session(self, req: SessionInitRequest) -> SessionInitResponse:
        """Initializes target KV cache with prompt prefill."""
        # Try WebSocket
        if await self.connect_websocket():
            try:
                async with self._ws_lock:
                    await self._ws.send(json.dumps({"type": "init", "data": req.model_dump()}))
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=self.config.timeout_seconds)
                    resp_data = json.loads(raw)
                    return SessionInitResponse(**resp_data.get("data", {}))
            except Exception as e:
                logger.warning(f"WS session init failed ({e}), falling back to HTTP")
                self._ws = None

        # Fallback to HTTP
        client = await self.get_http_client()
        r = await client.post("/v1/speculative/session/init", json=req.model_dump())
        r.raise_for_status()
        return SessionInitResponse(**r.json())

    async def verify_draft(self, req: VerifyRequest) -> VerifyResponse:
        """Dispatches draft tokens to target server for single-pass verification."""
        # Try WebSocket
        if await self.connect_websocket():
            try:
                async with self._ws_lock:
                    await self._ws.send(json.dumps({"type": "verify", "data": req.model_dump()}))
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=self.config.timeout_seconds)
                    resp_data = json.loads(raw)
                    return VerifyResponse(**resp_data.get("data", {}))
            except Exception as e:
                logger.warning(f"WS verify draft failed ({e}), falling back to HTTP")
                self._ws = None

        # Fallback to HTTP
        client = await self.get_http_client()
        r = await client.post("/v1/speculative/verify", json=req.model_dump())
        r.raise_for_status()
        return VerifyResponse(**r.json())

    async def close_session(self, session_id: str) -> bool:
        """Instructs server to release KV-cache memory allocated for session."""
        req = SessionCloseRequest(session_id=session_id)
        try:
            if self._ws is not None and not self._ws.closed:
                async with self._ws_lock:
                    await self._ws.send(json.dumps({"type": "close", "data": req.model_dump()}))
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=3.0)
                    return True
        except Exception:
            pass

        try:
            client = await self.get_http_client()
            r = await client.post("/v1/speculative/session/close", json=req.model_dump())
            return r.status_code == 200
        except Exception:
            return False

