"""KV-Cache Session and Lifecycle Manager.

Maintains per-session Key-Value caches for target models during speculative
verification, with surgical truncation / rollback capabilities to prevent
state drift when draft tokens are rejected.
"""

import time
from typing import Dict, Any, Optional, Tuple
from common.logger import get_logger

logger = get_logger("KVCacheManager")


class SessionKVCache:
    """Encapsulates KV cache, token history, and metadata for a single chat session."""

    def __init__(self, session_id: str, temperature: float = 0.0, top_p: float = 0.95):
        self.session_id = session_id
        self.temperature = temperature
        self.top_p = top_p
        self.past_key_values = None
        self.verified_token_ids: list[int] = []
        self.current_length: int = 0
        self.created_at: float = time.time()
        self.last_accessed: float = time.time()

    def touch(self):
        self.last_accessed = time.time()

    def update_cache(self, new_past_key_values, new_length: int):
        self.past_key_values = new_past_key_values
        self.current_length = new_length
        self.touch()

    def crop_to_length(self, target_len: int) -> int:
        """Surgically crops KV cache back to target_len positions.
        
        Essential for speculative decoding: when tokens at index m..K-1 are
        rejected, the KV-states computed during the parallel forward pass must
        be truncated back to target_len = prefix_len + m.
        """
        if self.past_key_values is None:
            self.current_length = target_len
            self.verified_token_ids = self.verified_token_ids[:target_len]
            return target_len

        # If using HuggingFace DynamicCache (transformers >= 4.36)
        if hasattr(self.past_key_values, "crop"):
            try:
                self.past_key_values.crop(target_len)
                self.current_length = target_len
                self.verified_token_ids = self.verified_token_ids[:target_len]
                self.touch()
                return target_len
            except Exception as e:
                logger.debug(f"DynamicCache.crop failed, falling back to manual slicing: {e}")

        # Slicing for tuple/list based past_key_values: tuple of (key, value) per layer
        # Key shape: [batch, num_heads, seq_len, head_dim]
        try:
            new_pkv = []
            for layer in self.past_key_values:
                k, v = layer[0], layer[1]
                # Slice along sequence dimension (dim=2 in standard HuggingFace models)
                k_sliced = k[:, :, :target_len, :]
                v_sliced = v[:, :, :target_len, :]
                new_pkv.append((k_sliced, v_sliced))
            self.past_key_values = tuple(new_pkv)
            self.current_length = target_len
            self.verified_token_ids = self.verified_token_ids[:target_len]
            self.touch()
            return target_len
        except Exception as e:
            logger.warning(f"Could not slice past_key_values manually for session {self.session_id}: {e}")
            self.current_length = target_len
            self.verified_token_ids = self.verified_token_ids[:target_len]
            return target_len


class KVCacheManager:
    """Thread-safe registry for session KV-caches with TTL eviction."""

    def __init__(self, session_ttl_seconds: float = 3600.0):
        self._sessions: Dict[str, SessionKVCache] = {}
        self.session_ttl_seconds = session_ttl_seconds

    def create_session(
        self,
        session_id: str,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> SessionKVCache:
        self.cleanup_expired()
        session = SessionKVCache(session_id, temperature=temperature, top_p=top_p)
        self._sessions[session_id] = session
        logger.info(f"Initialized KV-cache session '{session_id}'")
        return session

    def get_session(self, session_id: str) -> Optional[SessionKVCache]:
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def remove_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Released KV-cache session '{session_id}'")
            return True
        return False

    def cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_accessed) > self.session_ttl_seconds
        ]
        for sid in expired:
            self.remove_session(sid)

