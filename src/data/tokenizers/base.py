from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union


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

    @abstractmethod
    def verify_token_frequency_distribution(
        self,
        corpus: Union[str, Path, List[Union[str, Path]]],
    ) -> Dict[str, Any]:
        """
        Analyze token frequency distribution in training data.
        
        Computes statistics about how frequently tokens appear in the corpus
        to evaluate the quality of the BPE tokenizer.
        
        Args:
            corpus: Training corpus (file path(s) or text string)
        
        Returns:
            Dictionary with frequency statistics:
                - total_unique_tokens: Total number of unique tokens in corpus
                - min_frequency: Minimum token frequency
                - max_frequency: Maximum token frequency
                - mean_frequency: Average token frequency
                - median_frequency: Median token frequency
                - percentiles: Dictionary with p25, p50, p75, p95, p99 percentiles
                - frequencies: List of all token frequencies (sorted)
        """

