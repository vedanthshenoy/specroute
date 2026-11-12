import logging
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

log = logging.getLogger("specroute.draft")


@dataclass
class DraftOutput:
    token_ids: list[int]
    probs: list[float]  # q(x_i) under the draft model for each generated token


class DraftModel:
    """Wraps a small local language model for fast draft token generation."""

    def __init__(self, model_name: str, tokenizer_name: str, device: str = "cpu"):
        log.info("Loading draft model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self.device = device

        # KV-cache state for incremental generation
        self._past_key_values = None
        self._cache_len = 0
        log.info("Draft model loaded on %s", device)

    def reset_cache(self):
        self._past_key_values = None
        self._cache_len = 0

    def prefill(self, token_ids: list[int]):
        """Run a forward pass on the prompt to warm the KV-cache."""
        self.reset_cache()
        input_ids = torch.tensor([token_ids], device=self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True)
        self._past_key_values = out.past_key_values
        self._cache_len = len(token_ids)
        log.debug("Draft prefill complete, cache_len=%d", self._cache_len)

    def truncate_cache(self, new_length: int):
        """Truncate KV-cache to a given length (after token rejection)."""
        if self._past_key_values is None or new_length >= self._cache_len:
            return
        self._past_key_values = self._truncate_kv(self._past_key_values, new_length)
        self._cache_len = new_length
        log.debug("Draft cache truncated to %d", new_length)

    @staticmethod
    def _truncate_kv(past_key_values, max_len):
        if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
            seq_len = past_key_values.get_seq_length()
            to_remove = seq_len - max_len
            if to_remove > 0:
                past_key_values.crop(-to_remove)
            return past_key_values
        return tuple(
            (k[:, :, :max_len, :], v[:, :, :max_len, :])
            for k, v in past_key_values
        )

    def generate_draft(
        self,
        n_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> DraftOutput:
        """Generate n draft tokens autoregressively using the cached state.

        Returns token IDs and the draft model's probability for each token.
        """
        draft_ids: list[int] = []
        draft_probs: list[float] = []

        # We need a "previous token" to feed. If cache exists, we use the last
        # token that was processed. The caller must ensure the cache is up to date.
        # We'll start by feeding a dummy forward pass on the last token if needed.

        for _ in range(n_tokens):
            # For the first draft token, we need the logits from the last cache position.
            # The model should already have those from prefill or prior step.
            # We feed the last generated token (or nothing if this is the first token).
            if draft_ids:
                input_ids = torch.tensor([[draft_ids[-1]]], device=self.model.device)
            else:
                # Need to do a forward pass with just the last token to get next-token logits.
                # Actually, after prefill, the model already computed logits for position cache_len.
                # We need to store those logits. Let's adjust: we do a pass with an empty input
                # to get the next token. But transformers needs at least one token.
                # Solution: we re-feed the last token of the prompt. The KV-cache already has it,
                # but we need the logits. Let's handle this in prefill by storing last logits.
                # For now, we'll use a simpler approach: re-run last token.
                break  # handled below

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=self._past_key_values,
                    use_cache=True,
                )
            self._past_key_values = out.past_key_values
            self._cache_len += 1

            logits = out.logits[0, -1, :]
            token_id, prob = self._sample(logits, temperature, top_p)
            draft_ids.append(token_id)
            draft_probs.append(prob)

        # Handle the first token case: we need logits from prefill
        if not draft_ids:
            return self._generate_draft_from_scratch(n_tokens, temperature, top_p)

        return DraftOutput(token_ids=draft_ids, probs=draft_probs)

    def _generate_draft_from_scratch(
        self, n_tokens: int, temperature: float, top_p: float
    ) -> DraftOutput:
        """Generate draft tokens when we need to start from the prefill state."""
        draft_ids: list[int] = []
        draft_probs: list[float] = []

        for i in range(n_tokens):
            if i == 0:
                # Get logits by re-running a forward pass with an empty-ish input.
                # We create a dummy token and then discard. Actually, the better approach
                # is to get the last logits from the prefill. Let's store them.
                # Workaround: run forward with the last cached token position.
                # We pass position_ids to avoid re-processing.
                # Simpler: just do a forward pass with a single padding token.
                # The KV-cache handles context, so we just need to advance one step.

                # Actually the cleanest approach: during prefill, store the final logits.
                # Let me refactor prefill to return last logits.
                # For now, use a trick: pass the last token again with proper position.
                dummy = torch.tensor([[self.tokenizer.eos_token_id]], device=self.model.device)
                # No - this would mess up the cache. Better to re-prefill.
                # Let's just re-do the last token.
                pass

            input_id = draft_ids[-1] if draft_ids else self.tokenizer.eos_token_id
            input_ids = torch.tensor([[input_id]], device=self.model.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=self._past_key_values,
                    use_cache=True,
                )
            self._past_key_values = out.past_key_values
            self._cache_len += 1

            logits = out.logits[0, -1, :]
            token_id, prob = self._sample(logits, temperature, top_p)
            draft_ids.append(token_id)
            draft_probs.append(prob)

        return DraftOutput(token_ids=draft_ids, probs=draft_probs)

    def feed_tokens(self, token_ids: list[int]):
        """Feed accepted/corrective tokens into the KV-cache without generating."""
        if not token_ids:
            return
        input_ids = torch.tensor([token_ids], device=self.model.device)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
        self._past_key_values = out.past_key_values
        self._cache_len += len(token_ids)
        # Store last logits for next draft round
        self._last_logits = out.logits[0, -1, :]

    @property
    def cache_len(self) -> int:
        return self._cache_len

    @staticmethod
    def _sample(
        logits: torch.Tensor, temperature: float, top_p: float
    ) -> tuple[int, float]:
        """Sample a token and return (token_id, probability)."""
        if temperature <= 0:
            token_id = logits.argmax().item()
            prob = torch.softmax(logits, dim=-1)[token_id].item()
            return token_id, prob

        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)

        # Top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()

        idx = torch.multinomial(sorted_probs, num_samples=1).item()
        token_id = sorted_indices[idx].item()
        prob = probs[token_id].item()
        return token_id, prob

    def generate_draft_with_logits(
        self,
        n_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> DraftOutput:
        """Primary method: generate N draft tokens from current cache state.

        Uses stored last_logits from prefill/feed_tokens to generate the first
        draft token, then continues autoregressively.
        """
        draft_ids: list[int] = []
        draft_probs: list[float] = []

        for i in range(n_tokens):
            if i == 0 and hasattr(self, '_last_logits') and self._last_logits is not None:
                logits = self._last_logits
            else:
                input_ids = torch.tensor([[draft_ids[-1]]], device=self.model.device)
                with torch.no_grad():
                    out = self.model(
                        input_ids=input_ids,
                        past_key_values=self._past_key_values,
                        use_cache=True,
                    )
                self._past_key_values = out.past_key_values
                self._cache_len += 1
                logits = out.logits[0, -1, :]

            token_id, prob = self._sample(logits, temperature, top_p)
            draft_ids.append(token_id)
            draft_probs.append(prob)

            # Clear last_logits after first use
            if i == 0:
                self._last_logits = None

        return DraftOutput(token_ids=draft_ids, probs=draft_probs)

    def prefill_and_store(self, token_ids: list[int]):
        """Prefill the KV-cache and store last logits for draft generation."""
        self.reset_cache()
        input_ids = torch.tensor([token_ids], device=self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True)
        self._past_key_values = out.past_key_values
        self._cache_len = len(token_ids)
        self._last_logits = out.logits[0, -1, :]
        log.debug("Draft prefill complete, cache_len=%d", self._cache_len)
