"""Speculative Verification Protocol Models.

Defines the serialized payload structures exchanged between the local
orchestrator client and the remote target model verification server.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TokenizerCheckRequest(BaseModel):
    """Payload to verify tokenizer parity between client and server."""
    tokenizer_id: str
    vocab_size: int
    test_phrase: str = "The quick brown fox jumps over the lazy dog."
    test_token_ids: List[int] = Field(default_factory=list)


class TokenizerCheckResponse(BaseModel):
    """Result of tokenizer parity check."""
    status: str = "ok"
    synchronized: bool
    server_tokenizer_id: str
    server_vocab_size: int
    matches_vocab_size: bool
    matches_test_ids: bool
    message: str


class SessionInitRequest(BaseModel):
    """Initializes a generation session on the target server with prompt prefill."""
    session_id: str
    prompt_token_ids: List[int]
    temperature: float = 0.0
    top_p: float = 0.95
    eos_token_id: Optional[int] = None
    max_new_tokens: int = 512


class SessionInitResponse(BaseModel):
    """Confirms prompt prefill and returns the first target model token."""
    session_id: str
    status: str = "ok"
    prefill_length: int
    first_token_id: int
    is_eos: bool = False
    prefill_latency_ms: float = 0.0
    error: Optional[str] = None


class VerifyRequest(BaseModel):
    """Client request to verify K speculative draft tokens."""
    session_id: str
    step_index: int
    draft_tokens: List[int]
    draft_logprobs: Optional[List[float]] = None
    temperature: float = 0.0
    top_p: float = 0.95


class VerifyResponse(BaseModel):
    """Server response specifying accepted draft tokens and the corrective token."""
    session_id: str
    step_index: int
    accepted_tokens: List[int] = Field(default_factory=list)
    accepted_count: int
    rejection_index: Optional[int] = None
    corrective_token: int
    acceptance_rate: float
    target_kv_len: int
    is_eos: bool = False
    server_latency_ms: float = 0.0
    error: Optional[str] = None


class SessionCloseRequest(BaseModel):
    """Request to release KV-cache memory allocated for a session."""
    session_id: str


class SessionCloseResponse(BaseModel):
    """Confirmation of session KV cache cleanup."""
    session_id: str
    freed: bool = True

