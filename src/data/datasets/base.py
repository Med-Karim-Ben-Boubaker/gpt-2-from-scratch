from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.data.tokenizers.base import BaseTokenizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract dataset that tokenizes a corpus for downstream LLM training."""

    def __init__(self, tokenizer: BaseTokenizer, max_sequence_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        logger.info(
            f"Initializing {self.__class__.__name__} with tokenizer vocab size "
            f"{tokenizer.vocab_size} and max_sequence_length {max_sequence_length}"
        )

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of sequences the dataset can provide."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Return the tensors needed for a single training step."""


