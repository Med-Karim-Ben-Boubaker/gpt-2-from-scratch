from __future__ import annotations

from typing import Any, Dict, List

import torch

from src.data.datasets.base import BaseDataset
from src.data.tokenizers.base import BaseTokenizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InstructFineTuningDataset(BaseDataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: BaseTokenizer,
        max_sequence_length: int,
        stride: int = 0,
    ) -> None:
        super().__init__(tokenizer, max_sequence_length)
        self.stride = stride
        self.input_sequences: List[torch.Tensor] = []
        self.target_sequences: List[torch.Tensor] = []
        self.loss_masks: List[torch.Tensor] = []

        logger.info("Preparing %d instruction examples", len(examples))

        for example in examples:
            prompt = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n### Response:\n"
            )
            full_sequence = prompt + example["output"] + "<eot>"
            prompt_tokens = tokenizer.encode(prompt)
            full_tokens = tokenizer.encode(full_sequence)

            if len(full_tokens) > max_sequence_length:
                logger.debug(
                    "Skipping example (%d tokens) because it exceeds max_sequence_length %d",
                    len(full_tokens),
                    max_sequence_length,
                )
                continue

            loss_mask = (
                [0] * len(prompt_tokens)
                + [1] * (len(full_tokens) - len(prompt_tokens))
            )

            self.input_sequences.append(
                torch.tensor(prompt_tokens, dtype=torch.long)
            )
            self.target_sequences.append(
                torch.tensor(full_tokens, dtype=torch.long)
            )
            self.loss_masks.append(torch.tensor(loss_mask, dtype=torch.bool))

        logger.info(
            "Created %d filtered instruction examples", len(self.input_sequences)
        )

    def __len__(self) -> int:
        return len(self.input_sequences)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_sequences[idx],
            self.target_sequences[idx],
            self.loss_masks[idx],
        )

# Simple Test:
if __name__ == "__main__":
    from src.data.tokenizers.bpe_tokenizer import BPETokenizer
    import json
    print("Instruction Fine-Tuning Dataset Test")
    print("=" * 50)
    tokenizer = BPETokenizer()
    tokenizer.train(corpus="data/synthetic-data/3.txt")
    print("Tokenizer trained successfully")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("Creating instruction fine-tuning dataset...")
    with open("data/alpaca_data.json", "r") as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")
    dataset = InstructFineTuningDataset(examples=examples, tokenizer=tokenizer, max_sequence_length=512)
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample input: {dataset[0][0]}")
    print(f"Sample target: {dataset[0][1]}")
    print(f"Sample loss mask: {dataset[0][2]}")