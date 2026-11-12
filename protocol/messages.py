from pydantic import BaseModel


class SessionInitRequest(BaseModel):
    """Sent once at the start of a chat turn to sync state."""
    session_id: str
    prompt_token_ids: list[int]
    tokenizer_name: str


class SessionInitResponse(BaseModel):
    session_id: str
    ready: bool
    cache_position: int  # Server's KV-cache length after processing prompt


class VerifyRequest(BaseModel):
    """Client sends draft tokens for the server to verify."""
    session_id: str
    draft_token_ids: list[int]
    draft_probs: list[float]  # q(x_i) for each draft token under draft model
    cache_position: int       # Expected server KV-cache position before these tokens


class VerifyResponse(BaseModel):
    """Server returns verification results."""
    accepted_count: int          # How many consecutive draft tokens were accepted
    corrective_token_id: int     # Next correct token (sampled from target model)
    accepted_token_ids: list[int]  # The accepted tokens for confirmation
    target_probs: list[float]    # p(x_i) from target model (for logging)
    new_cache_position: int      # Server KV-cache position after this verification
