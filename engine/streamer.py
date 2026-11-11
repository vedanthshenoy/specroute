"""Streaming Token Pacer and UI Smoothing Engine.

Ensures that tokens arrive in the LibreChat UI with a smooth, continuous
typewriter rhythm, concealing remote validation network latency and
eliminating jerky burst artifacts native to speculative chunk verification.
"""

import asyncio
import time
from typing import AsyncGenerator, List, Optional
from common.tokenizer_sync import SynchronizedTokenizer


class TokenStreamSmoother:
    """Buffers and smoothly paces verified speculative tokens to the SSE stream."""

    def __init__(
        self,
        tokenizer: SynchronizedTokenizer,
        pacing_ms: int = 12,
        enabled: bool = True,
    ):
        self.tokenizer = tokenizer
        self.pacing_seconds = max(0.0, pacing_ms / 1000.0)
        self.enabled = enabled
        self._queue: asyncio.Queue[Optional[int]] = asyncio.Queue()
        self._done: bool = False

    async def push_verified_tokens(self, token_ids: List[int]) -> None:
        """Pushes a verified slice of tokens (accepted + corrective) into the pacing queue."""
        for tid in token_ids:
            await self._queue.put(tid)

    async def finish(self) -> None:
        """Signals completion of generation."""
        self._done = True
        await self._queue.put(None)

    async def stream_text_deltas(self) -> AsyncGenerator[str, None]:
        """Asynchronously yields decoded string deltas at a paced interval."""
        while True:
            token_id = await self._queue.get()
            if token_id is None:
                break

            # Decode individual token into text delta
            text_delta = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if text_delta:
                yield text_delta

            if self.enabled and self.pacing_seconds > 0:
                await asyncio.sleep(self.pacing_seconds)

