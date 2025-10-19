from torch.utils.data import DataLoader
from src.data.datasets import GPTDatasetV1, InstructFineTuningDataset
from src.utils.logging import get_logger
from typing import List, Dict, Any

logger = get_logger(__name__)

def instruction_collate_fn(batch):
    """
    Custom collate function for instruction fine-tuning dataset.
    """
    input_seqs, target_seqs, loss_masks = zip(*batch)
    
    return list(input_seqs), list(target_seqs), list(loss_masks)

def create_dataloader_v1(
    text, tokenizer, batch_size, max_length, stride, shuffle, drop_last, num_workers
):
    logger.info(f"Creating dataloader with text length: {len(text)}, Max length: {max_length}, Stride: {stride}")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

def create_instruction_dataloader(
    examples: List[Dict[str, Any]], 
    tokenizer,
    batch_size: int, 
    max_length: int, 
    shuffle: bool = True, 
    drop_last: bool = True, 
    num_workers: int = 0
) -> DataLoader:
    logger.info(f"Creating instruction dataloader with {len(examples)} examples, "
                f"batch_size: {batch_size}, max_length: {max_length}")
    
    dataset = InstructFineTuningDataset(examples, tokenizer, max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=instruction_collate_fn,
    )
