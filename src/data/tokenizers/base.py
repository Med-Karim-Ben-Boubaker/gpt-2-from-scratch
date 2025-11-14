from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseTokenizer(ABC):
    """Minimal tokenizer interface used by the LLM application."""

    @abstractmethod
    def encode(
        self,
        text: str
    ) -> List[int]:
        """Encode a string into a sequence of token IDs."""

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode a sequence of token IDs back into a string."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the number of entries in the vocabulary."""

    @property
    @abstractmethod
    def vocab(self) -> Dict[str, int]:
        """Return the token-to-id mapping used by the tokenizer."""

