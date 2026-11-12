"""Remote target model verification server.

Hosts the large target model and verifies draft tokens from the local client
using speculative decoding acceptance criteria. Designed to run on
Google Colab / Kaggle with GPU, exposed via ngrok.
"""

import logging
import random
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

from protocol.messages import (
    SessionInitRequest,
    SessionInitResponse,
    VerifyRequest,
    VerifyResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("specroute.target")

app = FastAPI(title="SpecRoute Target Server")


class TargetModelServer:
    """Manages the target model, KV-cache, and speculative verification."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sessions: dict[str, dict] = {}  # session_id -> {past_key_values, cache_len}

    def load_model(self, model_name: str, tokenizer_name: str, device: str = "cuda"):
        log.info("Loading target model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        self.model.eval()
        self.device = device
        log.info("Target model loaded on %s", device)

    def init_session(self, req: SessionInitRequest) -> SessionInitResponse:
        """Process the prompt and build initial KV-cache."""
        log.info("Initializing session %s with %d prompt tokens", req.session_id, len(req.prompt_token_ids))

        input_ids = torch.tensor([req.prompt_token_ids], device=self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True)

        self.sessions[req.session_id] = {
            "past_key_values": out.past_key_values,
            "cache_len": len(req.prompt_token_ids),
            "last_logits": out.logits[0, -1, :],  # logits for next token prediction
        }

        return SessionInitResponse(
            session_id=req.session_id,
            ready=True,
            cache_position=len(req.prompt_token_ids),
        )

    def verify(self, req: VerifyRequest) -> VerifyResponse:
        """Verify draft tokens against the target model.

        Runs a single forward pass on all draft tokens, then applies the
        speculative decoding acceptance criterion sequentially.
        """
        session = self.sessions.get(req.session_id)
        if session is None:
            raise ValueError(f"Unknown session: {req.session_id}")

        past_kv = session["past_key_values"]
        last_logits = session["last_logits"]

        draft_ids = req.draft_token_ids
        draft_probs = req.draft_probs
        n_draft = len(draft_ids)

        # Forward pass: feed all draft tokens at once to get logits at each position
        input_ids = torch.tensor([draft_ids], device=self.model.device)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        # out.logits shape: [1, n_draft, vocab_size]
        # Position i gives logits for predicting token at position (cache_len + i + 1)
        # But we need logits for verifying token at position (cache_len + i)
        # The logits for verifying draft_ids[0] come from last_logits (stored from previous step)
        # The logits for verifying draft_ids[i] come from out.logits[0, i-1, :] (i > 0)

        all_logits = torch.stack(
            [last_logits] + [out.logits[0, i, :] for i in range(n_draft - 1)]
        )  # shape: [n_draft, vocab_size]

        # Compute target probabilities with temperature
        temperature = 0.7  # Should match client config
        if temperature > 0:
            target_probs_full = torch.softmax(all_logits / temperature, dim=-1)
        else:
            target_probs_full = torch.softmax(all_logits, dim=-1)

        # Extract p(x_i) for each draft token
        target_probs_for_draft = []
        for i, tid in enumerate(draft_ids):
            target_probs_for_draft.append(target_probs_full[i, tid].item())

        # Speculative decoding acceptance loop
        accepted_count = 0
        for i in range(n_draft):
            p = target_probs_for_draft[i]  # target model prob
            q = draft_probs[i]             # draft model prob

            if q <= 0:
                # Draft assigned zero probability, reject
                break

            # Accept if p >= q, otherwise accept with probability p/q
            if p >= q:
                accepted_count += 1
            elif random.random() < (p / q):
                accepted_count += 1
            else:
                break

        # Determine corrective token
        if accepted_count == n_draft:
            # All accepted: sample next token from target model at position after last draft token
            next_logits = out.logits[0, -1, :]
            corrective_id = self._sample_from_logits(next_logits, temperature)
        else:
            # Rejected at position `accepted_count`: sample from adjusted distribution
            # Simplified: sample from target distribution at the rejection point
            reject_logits = all_logits[accepted_count]
            corrective_id = self._sample_from_logits(reject_logits, temperature)

        # Truncate KV-cache to prompt + accepted tokens, then feed corrective token
        trunc_len = session["cache_len"] + accepted_count
        trunc_kv = self._truncate_kv(out.past_key_values, trunc_len)

        corr_input = torch.tensor([[corrective_id]], device=self.model.device)
        with torch.no_grad():
            corr_out = self.model(
                input_ids=corr_input,
                past_key_values=trunc_kv,
                use_cache=True,
            )

        session["past_key_values"] = corr_out.past_key_values
        session["cache_len"] = trunc_len + 1
        session["last_logits"] = corr_out.logits[0, -1, :]

        accepted_ids = draft_ids[:accepted_count]

        log.info(
            "[Speculation] %d/%d tokens accepted | corrective=%d (%s)",
            accepted_count,
            n_draft,
            corrective_id,
            self.tokenizer.decode([corrective_id]),
        )

        return VerifyResponse(
            accepted_count=accepted_count,
            corrective_token_id=corrective_id,
            accepted_token_ids=accepted_ids,
            target_probs=target_probs_for_draft,
            new_cache_position=session["cache_len"],
        )

    @staticmethod
    def _truncate_kv(past_key_values, max_len):
        """Truncate a KV-cache (DynamicCache or tuple-of-tuples) to max_len."""
        if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
            seq_len = past_key_values.get_seq_length()
            to_remove = seq_len - max_len
            if to_remove > 0:
                past_key_values.crop(-to_remove)
            return past_key_values
        # Legacy tuple-of-tuples format
        return tuple(
            (k[:, :, :max_len, :], v[:, :, :max_len, :])
            for k, v in past_key_values
        )

    @staticmethod
    def _sample_from_logits(logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0:
            return logits.argmax().item()
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


# Global server instance
target_server = TargetModelServer()


@app.post("/api/session/init")
async def session_init(req: SessionInitRequest):
    try:
        resp = target_server.init_session(req)
        return resp.model_dump()
    except Exception as e:
        log.exception("Session init failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify")
async def verify(req: VerifyRequest):
    try:
        resp = target_server.verify(req)
        return resp.model_dump()
    except Exception as e:
        log.exception("Verification failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": target_server.model is not None}


def start_server(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    host: str = "0.0.0.0",
    port: int = 8001,
    ngrok_token: str | None = None,
):
    target_server.load_model(model_name, tokenizer_name, device)

    if ngrok_token:
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        tunnel = ngrok.connect(port)
        log.info("ngrok tunnel: %s", tunnel.public_url)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SpecRoute Target Server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--ngrok-token", default=None)
    args = parser.parse_args()

    start_server(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        device=args.device,
        host=args.host,
        port=args.port,
        ngrok_token=args.ngrok_token,
    )
