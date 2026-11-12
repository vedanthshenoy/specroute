import asyncio
import json
import logging
import time
from typing import AsyncIterator

import aiohttp
import websockets

from local_engine.draft_model import DraftModel
from protocol.messages import (
    SessionInitRequest,
    SessionInitResponse,
    VerifyRequest,
    VerifyResponse,
)

log = logging.getLogger("specroute.orchestrator")


class SpeculativeOrchestrator:
    """Core speculative decoding loop coordinating local draft and remote target models."""

    def __init__(self, config: dict):
        self.config = config
        self.draft = DraftModel(
            model_name=config["draft_model"]["name"],
            tokenizer_name=config["tokenizer"],
            device=config["draft_model"].get("device", "cpu"),
        )
        self.tokenizer = self.draft.tokenizer
        self.remote_url = config["remote"]["url"]
        self.use_ws = config["remote"].get("websocket", True)
        self.lookahead = config["speculation"]["lookahead"]
        self.temperature = config["speculation"]["temperature"]
        self.top_p = config["speculation"]["top_p"]
        self.max_rounds = config["speculation"]["max_rounds"]
        self.timeout = config["remote"].get("timeout_ms", 10000) / 1000.0

        # Stats
        self._total_draft = 0
        self._total_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        if self._total_draft == 0:
            return 0.0
        return self._total_accepted / self._total_draft

    async def generate_stream(
        self, messages: list[dict], session_id: str
    ) -> AsyncIterator[str]:
        """Run the full speculative decoding loop, yielding accepted text chunks.

        This is the main entry point called by the API server.
        """
        # Tokenize the chat prompt
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        log.info("Prompt length: %d tokens", len(prompt_ids))

        # Prefill draft model
        self.draft.prefill_and_store(prompt_ids)

        # Initialize remote session
        await self._init_remote_session(session_id, prompt_ids)

        # Speculative decoding loop
        generated_ids: list[int] = []
        self._total_draft = 0
        self._total_accepted = 0

        for round_num in range(self.max_rounds):
            # 1. Generate draft tokens locally
            t0 = time.perf_counter()
            draft_output = self.draft.generate_draft_with_logits(
                n_tokens=self.lookahead,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            draft_time = time.perf_counter() - t0

            log.debug(
                "Round %d: drafted %d tokens in %.1fms: %s",
                round_num,
                len(draft_output.token_ids),
                draft_time * 1000,
                self.tokenizer.decode(draft_output.token_ids),
            )

            # 2. Send to remote for verification
            verify_req = VerifyRequest(
                session_id=session_id,
                draft_token_ids=draft_output.token_ids,
                draft_probs=draft_output.probs,
                cache_position=len(prompt_ids) + len(generated_ids),
            )

            t1 = time.perf_counter()
            verify_resp = await self._verify_remote(verify_req)
            verify_time = time.perf_counter() - t1

            accepted = verify_resp.accepted_count
            self._total_draft += len(draft_output.token_ids)
            self._total_accepted += accepted

            log.info(
                "[Speculation] %d/%d tokens accepted by remote target model "
                "(%.1fms draft, %.1fms verify, cumulative rate: %.1f%%)",
                accepted,
                len(draft_output.token_ids),
                draft_time * 1000,
                verify_time * 1000,
                self.acceptance_rate * 100,
            )

            # 3. Process accepted tokens + corrective token
            # Truncate draft cache to the point of divergence
            cache_rewind = len(prompt_ids) + len(generated_ids) + accepted
            self.draft.truncate_cache(cache_rewind)

            # Collect all tokens to emit: accepted draft tokens + corrective token
            new_tokens = verify_resp.accepted_token_ids + [verify_resp.corrective_token_id]
            generated_ids.extend(new_tokens)

            # Feed the corrective token (and any accepted tokens) into draft cache
            self.draft.feed_tokens(new_tokens)

            # Yield the text for these tokens
            text_chunk = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            if text_chunk:
                yield text_chunk

            # Check for EOS
            if verify_resp.corrective_token_id == self.tokenizer.eos_token_id:
                log.info(
                    "Generation complete. Total tokens: %d, Acceptance rate: %.1f%%",
                    len(generated_ids),
                    self.acceptance_rate * 100,
                )
                break

        log.info(
            "Final stats: %d tokens generated, %d/%d draft tokens accepted (%.1f%%)",
            len(generated_ids),
            self._total_accepted,
            self._total_draft,
            self.acceptance_rate * 100,
        )

    async def _init_remote_session(self, session_id: str, prompt_ids: list[int]):
        """Send the prompt to the remote server to initialize its KV-cache."""
        req = SessionInitRequest(
            session_id=session_id,
            prompt_token_ids=prompt_ids,
            tokenizer_name=self.config["tokenizer"],
        )
        headers = {
            "ngrok-skip-browser-warning": "true",
            "Content-Type": "application/json",
        }

        # First, verify remote is reachable
        health_url = f"{self.remote_url}/health"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    health_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    body = await resp.text()
                    log.info("Remote health check: status=%d body=%s", resp.status, body[:200])
                    if resp.status != 200:
                        raise RuntimeError(f"Remote health check failed ({resp.status}): {body[:500]}")
            except aiohttp.ClientError as e:
                raise RuntimeError(f"Cannot reach remote server at {self.remote_url}: {e}")

        url = f"{self.remote_url}/api/session/init"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=req.model_dump(), headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout * 3),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Session init failed ({resp.status}): {body[:500]}")
                data = await resp.json(content_type=None)
                init_resp = SessionInitResponse(**data)
                if not init_resp.ready:
                    raise RuntimeError("Remote server failed to initialize session")
                log.info(
                    "Remote session initialized, cache_position=%d",
                    init_resp.cache_position,
                )

    async def _verify_remote(self, req: VerifyRequest) -> VerifyResponse:
        """Send draft tokens to remote server for verification."""
        url = f"{self.remote_url}/api/verify"
        headers = {
            "ngrok-skip-browser-warning": "true",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=req.model_dump(), headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Verification failed ({resp.status}): {body[:500]}")
                data = await resp.json(content_type=None)
                return VerifyResponse(**data)
