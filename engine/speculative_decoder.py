"""Speculative Decoding Orchestrator.

Implements the speculative decoding loop coordinating the local lightweight
draft model with the remote high-capacity target model.
"""

import time
import uuid
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple
from common.protocol import (
    SessionInitRequest,
    SessionInitResponse,
    VerifyRequest,
    VerifyResponse,
)
from common.logger import get_logger, SpeculativeMetrics
from common.tokenizer_sync import SynchronizedTokenizer
from engine.base import BaseDraftEngine
from engine.remote_client import RemoteTargetClient
from engine.streamer import TokenStreamSmoother
from config.settings import AppConfig

logger = get_logger("SpeculativeDecoder")


class SpeculativeDecodingOrchestrator:
    """Manages the rolling speculative generation loop between local draft and cloud target."""

    def __init__(
        self,
        config: AppConfig,
        tokenizer: SynchronizedTokenizer,
        draft_engine: BaseDraftEngine,
        remote_client: RemoteTargetClient,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.draft_engine = draft_engine
        self.remote_client = remote_client

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Main entry point: streams verified text deltas to the caller / SSE proxy."""
        temp = temperature if temperature is not None else self.config.speculative.temperature
        p = top_p if top_p is not None else self.config.speculative.top_p
        max_tokens = max_new_tokens if max_new_tokens is not None else self.config.speculative.max_new_tokens
        k = self.config.speculative.lookahead_k

        session_id = f"spec_{uuid.uuid4().hex[:12]}"
        metrics = SpeculativeMetrics()

        # Step 1: Format prompt using chat template and encode to token IDs
        prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        logger.info(
            f"Starting Speculative Decoding session '{session_id}' | "
            f"Prompt tokens: {len(prompt_tokens)} | Lookahead K: {k} | Temp: {temp}"
        )

        # Step 2: Initialize remote target server session with prompt prefill
        t_prefill = time.perf_counter()
        init_req = SessionInitRequest(
            session_id=session_id,
            prompt_token_ids=prompt_tokens,
            temperature=temp,
            top_p=p,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )
        init_resp = await self.remote_client.init_session(init_req)
        if init_resp.status != "ok":
            logger.error(f"Remote prefill failed for session '{session_id}': {init_resp.error}")
            yield f"[Error initializing speculative session: {init_resp.error}]"
            return

        first_token = init_resp.first_token_id
        prefill_latency = (time.perf_counter() - t_prefill) * 1000.0
        logger.info(
            f"Remote prefill complete ({prefill_latency:.1f}ms). "
            f"Initial token ID: {first_token} ({repr(self.tokenizer.decode([first_token]))})"
        )

        # Step 3: Initialize local draft model with prompt + first token
        self.draft_engine.reset()
        self.draft_engine.prefill(prompt_tokens, initial_token=first_token)

        # Step 4: Setup token smoother / streamer
        smoother = TokenStreamSmoother(
            tokenizer=self.tokenizer,
            pacing_ms=self.config.speculative.token_pacing_ms,
            enabled=self.config.speculative.smooth_streaming,
        )

        # Background worker for speculative loop
        async def run_speculative_loop():
            nonlocal first_token
            step = 1
            tokens_generated = 1  # count first_token

            # Push initial token into smoother immediately
            await smoother.push_verified_tokens([first_token])

            if init_resp.is_eos or first_token == self.tokenizer.eos_token_id:
                logger.info("First token is EOS. Generation complete.")
                await smoother.finish()
                return

            try:
                while tokens_generated < max_tokens:
                    t_step = time.perf_counter()

                    # 1. Local draft model guesses N-token lookahead window
                    draft_tokens = self.draft_engine.generate_draft(
                        k=k,
                        temperature=temp,
                        top_p=p,
                    )

                    # 2. Network dispatch to remote target server
                    verify_req = VerifyRequest(
                        session_id=session_id,
                        step_index=step,
                        draft_tokens=draft_tokens,
                        temperature=temp,
                        top_p=p,
                    )
                    verify_resp = await self.remote_client.verify_draft(verify_req)

                    if verify_resp.error:
                        logger.error(f"Speculative verify error at step {step}: {verify_resp.error}")
                        break

                    # 3. Rollback unverified tokens and inject verified tokens into local draft engine
                    self.draft_engine.rollback_and_inject(
                        accepted_tokens=verify_resp.accepted_tokens,
                        corrective_token=verify_resp.corrective_token,
                    )

                    # 4. Record and log metrics
                    corr_text = self.tokenizer.decode([verify_resp.corrective_token], skip_special_tokens=False)
                    step_latency = (time.perf_counter() - t_step) * 1000.0
                    step_log = metrics.record_step(
                        step=step,
                        drafted=len(draft_tokens),
                        accepted=verify_resp.accepted_count,
                        corrective_token_text=corr_text,
                        latency_ms=step_latency,
                    )
                    logger.info(step_log)

                    # 5. Push verified slice + corrective token to UI streamer
                    verified_slice = list(verify_resp.accepted_tokens) + [verify_resp.corrective_token]
                    await smoother.push_verified_tokens(verified_slice)

                    tokens_generated += len(verified_slice)
                    step += 1

                    # 6. Check for termination condition
                    if verify_resp.is_eos:
                        logger.info(f"EOS reached at step {step}. Finishing generation.")
                        break

            finally:
                # Signal smoother we are done
                await smoother.finish()
                # Close server session to free KV cache memory
                await self.remote_client.close_session(session_id)
                # Print final summary statistics
                logger.info(metrics.summary())

        # Start the background speculative generation task
        loop_task = asyncio.create_task(run_speculative_loop())

        # Concurrently yield text deltas from the smoother to the HTTP/SSE client
        try:
            async for delta in smoother.stream_text_deltas():
                yield delta
        finally:
            await loop_task

