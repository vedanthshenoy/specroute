"""Abstract Interface for Local Draft Inference Engines."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseDraftEngine(ABC):
    """Base class for lightweight local models generating speculative candidate tokens."""

    @abstractmethod
    def prefill(self, prompt_tokens: List[int], initial_token: Optional[int] = None) -> None:
        """Initializes the draft model state with the prompt tokens and the first verified token."""
        pass

    @abstractmethod
    def generate_draft(self, k: int, temperature: float = 0.0, top_p: float = 0.95) -> List[int]:
        """Generates an N-token lookahead candidate window autoregressively."""
        pass

    @abstractmethod
    def rollback_and_inject(self, accepted_tokens: List[int], corrective_token: int) -> None:
        """Truncates rejected draft tokens and updates draft state with verified tokens."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Cleans up internal state and KV-cache."""
        pass

