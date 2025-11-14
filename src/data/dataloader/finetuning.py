from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

from src.data.datasets import InstructFineTuningDataset
from src.utils.logging import get_logger

logger = get_logger(__name__)


def instruction_collate_fn(batch: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Custom collate function for instruction fine-tuning dataset."""
    input_seqs, target_seqs, loss_masks = zip(*batch)
    return list(input_seqs), list(target_seqs), list(loss_masks)


def create_instruction_dataloader(
    examples: List[Dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    max_sequence_length: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    logger.info(
        f"Creating instruction dataloader with {len(examples)} examples, "
        f"batch_size: {batch_size}, max_sequence_length: {max_sequence_length}"
    )

    dataset = InstructFineTuningDataset(examples, tokenizer, max_sequence_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=instruction_collate_fn,
    )
