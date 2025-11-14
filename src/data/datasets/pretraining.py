from __future__ import annotations

from typing import List

import torch

from src.data.datasets.base import BaseDataset
from src.data.tokenizers.base import BaseTokenizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PretrainingDataset(BaseDataset):
    def __init__(
        self,
        text: str,
        tokenizer: BaseTokenizer,
        max_sequence_length: int,
        stride: int = 1,
    ) -> None:
        super().__init__(tokenizer, max_sequence_length)
        self.stride = stride
        self.input_sequences: List[torch.Tensor] = []
        self.target_sequences: List[torch.Tensor] = []

        tokenized_text = tokenizer.encode(text)

        logger.info(
            f"{self.__class__.__name__}: text length {len(text)}, tokenized length "
            f"{len(tokenized_text)}, max_sequence_length {max_sequence_length}, stride {stride}"
        )

        if len(tokenized_text) <= max_sequence_length:
            logger.warning(
                "Tokenized text length (%d) <= max_sequence_length (%d); no examples created",
                len(tokenized_text),
                max_sequence_length,
            )
            return

        for sequence_start_index in range(0, len(tokenized_text) - max_sequence_length, max(1, stride)):
            input_sequence = tokenized_text[
                sequence_start_index : sequence_start_index + max_sequence_length
            ]
            target_sequence = tokenized_text[
                sequence_start_index + 1 : sequence_start_index + max_sequence_length + 1
            ]
            self.input_sequences.append(torch.tensor(input_sequence))
            self.target_sequences.append(torch.tensor(target_sequence))

    def __len__(self) -> int:
        return len(self.input_sequences)

    def __getitem__(self, sequence_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.input_sequences[sequence_index],
            self.target_sequences[sequence_index],
        )
        
# Small Test
if __name__ == "__main__":
    from src.data.tokenizers.bpe_tokenizer import BPETokenizer
    print("Pretraining Dataset Test")
    print("=" * 50)
    tokenizer = BPETokenizer()
    print("Training tokenizer...")
    tokenizer.train(corpus="data/synthetic-data/3.txt")
    print("Tokenizer trained successfully")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    text = open("data/synthetic-data/3.txt", "r").read()
    print("Creating pretraining dataset...")
    dataset = PretrainingDataset(text=text, tokenizer=tokenizer, max_sequence_length=10)
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample input: {dataset[0][0]}")
    print(f"Sample target: {dataset[0][1]}")


