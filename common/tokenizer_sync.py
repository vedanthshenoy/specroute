"""Tokenizer Synchronization and Parity Validation Module.

Ensures that the local client and the remote server use identical tokenizers,
preventing token ID divergence during speculative generation.
"""

from typing import List, Dict, Any, Optional, Union
import hashlib
from common.protocol import TokenizerCheckRequest, TokenizerCheckResponse
from common.logger import get_logger

logger = get_logger("TokenizerSync")


class SynchronizedTokenizer:
    """Wraps HuggingFace AutoTokenizer or provides a high-fidelity standalone fallback."""

    CANONICAL_TEST_PHRASES = [
        "The quick brown fox jumps over the lazy dog.",
        "System prompt: You are a helpful assistant.",
        "Speculative decoding: 12345 + 67890 = 80235.",
    ]

    def __init__(self, tokenizer_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer_id = tokenizer_id
        self._hf_tokenizer = None
        self._is_loaded = False
        self._init_tokenizer()

    def _init_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            logger.info(f"Loading HuggingFace tokenizer '{self.tokenizer_id}'...")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                trust_remote_code=True,
            )
            self._is_loaded = True
            logger.info(f"Tokenizer loaded successfully. Vocab size: {len(self._hf_tokenizer)}")
        except Exception as e:
            logger.warning(
                f"Could not load HuggingFace tokenizer '{self.tokenizer_id}' ({e}). "
                f"Falling back to built-in SynchronizedTokenizer emulation."
            )
            self._hf_tokenizer = None
            self._is_loaded = False

    @property
    def is_hf_loaded(self) -> bool:
        return self._is_loaded and self._hf_tokenizer is not None

    @property
    def vocab_size(self) -> int:
        if self._hf_tokenizer:
            return len(self._hf_tokenizer)
        return 151646  # Default Qwen2.5 vocab size

    @property
    def eos_token_id(self) -> int:
        if self._hf_tokenizer and self._hf_tokenizer.eos_token_id is not None:
            return self._hf_tokenizer.eos_token_id
        return 151643  # <|im_end|> or <|endoftext|> in Qwen

    @property
    def pad_token_id(self) -> int:
        if self._hf_tokenizer and self._hf_tokenizer.pad_token_id is not None:
            return self._hf_tokenizer.pad_token_id
        return self.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if self._hf_tokenizer:
            return self._hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        
        # Emulation mode for offline / test environments
        # Generates deterministic token IDs based on word / char hashes
        return self._emulated_encode(text)

    def decode(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if self._hf_tokenizer:
            return self._hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return self._emulated_decode(token_ids)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Applies chat template (ChatML / Qwen format) to message list."""
        if self._hf_tokenizer and hasattr(self._hf_tokenizer, "apply_chat_template"):
            try:
                return self._hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception as e:
                logger.debug(f"apply_chat_template fallback due to: {e}")

        # Standard ChatML fallback (Qwen / Llama format)
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    def generate_check_request(self) -> TokenizerCheckRequest:
        """Constructs a parity verification request for the server."""
        test_phrase = self.CANONICAL_TEST_PHRASES[0]
        test_ids = self.encode(test_phrase, add_special_tokens=False)
        return TokenizerCheckRequest(
            tokenizer_id=self.tokenizer_id,
            vocab_size=self.vocab_size,
            test_phrase=test_phrase,
            test_token_ids=test_ids,
        )

    def verify_remote_parity(self, response: TokenizerCheckResponse) -> bool:
        """Evaluates whether the remote server's tokenizer matches exactly."""
        if not response.matches_vocab_size or not response.matches_test_ids:
            logger.error(
                f"[Tokenizer Parity FAILED] Local ({self.tokenizer_id}, vocab={self.vocab_size}) "
                f"differs from Remote ({response.server_tokenizer_id}, vocab={response.server_vocab_size}). "
                f"Message: {response.message}"
            )
            return False
        logger.info(
            f"[Tokenizer Parity OK] Synchronized with remote ({response.server_tokenizer_id}, "
            f"vocab={response.server_vocab_size})."
        )
        return True

    def validate_check_request(self, req: TokenizerCheckRequest) -> TokenizerCheckResponse:
        """Executed on the server to validate incoming client request."""
        server_test_ids = self.encode(req.test_phrase, add_special_tokens=False)
        matches_vocab = (self.vocab_size == req.vocab_size)
        matches_ids = (server_test_ids == req.test_token_ids)
        synchronized = matches_vocab and matches_ids

        msg = "Tokenizer synchronized successfully" if synchronized else (
            f"Mismatch detected! Server vocab={self.vocab_size}, Client vocab={req.vocab_size}; "
            f"Test IDs match: {matches_ids}"
        )

        return TokenizerCheckResponse(
            status="ok" if synchronized else "mismatch",
            synchronized=synchronized,
            server_tokenizer_id=self.tokenizer_id,
            server_vocab_size=self.vocab_size,
            matches_vocab_size=matches_vocab,
            matches_test_ids=matches_ids,
            message=msg,
        )

    # --------------------------------------------------------------------------
    # Deterministic Emulation for testing environments without HuggingFace weights
    # --------------------------------------------------------------------------
    _emulated_id_to_str: Dict[int, str] = {
        151643: "<|im_end|>",
        151644: "<|im_start|>",
        151645: "<|endoftext|>",
    }
    _emulated_str_to_id: Dict[str, int] = {
        "<|im_end|>": 151643,
        "<|im_start|>": 151644,
        "<|endoftext|>": 151645,
    }

    def _emulated_encode(self, text: str) -> List[int]:
        # Quick deterministic subword / word-piece encoder
        import re
        tokens = re.findall(r"<\|[^>]+\|>|\w+|[^\w\s]|\s+", text)
        result = []
        for t in tokens:
            if t in self._emulated_str_to_id:
                result.append(self._emulated_str_to_id[t])
            else:
                h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:6], 16) % 100000 + 1000
                self._emulated_str_to_id[t] = h
                self._emulated_id_to_str[h] = t
                result.append(h)
        return result

    def _emulated_decode(self, token_ids: List[int]) -> str:
        parts = []
        for tid in token_ids:
            parts.append(self._emulated_id_to_str.get(tid, f"[tok_{tid}]"))
        return "".join(parts)

