from torch.utils.data import Dataset
import torch
from typing import List, Dict, Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_sequences, self.target_sequences = [], []

        tokenized_text = tokenizer.encode(text)

        logger.info(
            f"Initializing GPTDatasetV1 with text length: {len(text)}, "
            f"Tokenized length: {len(tokenized_text)}, Max length: {max_length}, Stride: {stride}"
        )

        # Check if we can create any sequences
        if len(tokenized_text) <= max_length:
            logger.warning(
                f"Tokenized text length ({len(tokenized_text)}) is <= max_length ({max_length}). "
                f"No sequences can be created. Consider using a smaller max_length or longer text."
            )
            return

        for sequence_start_index in range(0, len(tokenized_text) - max_length, stride):
            input_sequence = tokenized_text[
                sequence_start_index : sequence_start_index + max_length
            ]
            target_sequence = tokenized_text[
                sequence_start_index + 1 : sequence_start_index + max_length + 1
            ]
            self.input_sequences.append(torch.tensor(input_sequence))
            self.target_sequences.append(torch.tensor(target_sequence))

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, sequence_index):
        return self.input_sequences[sequence_index], self.target_sequences[
            sequence_index
        ]
        
class InstructFineTuningDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int, stride: int = 0):
        self.input_sequences: List[torch.Tensor] = []
        self.target_sequences: List[torch.Tensor] = []
        self.loss_masks: List[torch.Tensor] = []
        
        logger.info(f"Initializing InstructFineTuningDataset with {len(examples)} examples")
        
        for example in examples:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
            full_sequence = prompt + example["output"] + "<eot>"
            prompt_tokens = tokenizer.encode(prompt)
            full_tokens = tokenizer.encode(full_sequence)
            
            if len(full_tokens) > max_length:
                continue
            
            loss_mask = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))
            
            self.input_sequences.append(torch.tensor(prompt_tokens, dtype=torch.long))
            self.target_sequences.append(torch.tensor(full_tokens, dtype=torch.long))
            self.loss_masks.append(torch.tensor(loss_mask, dtype=torch.bool))
        
        logger.info(f"Created {len(self.input_sequences)} training examples")
    
    def __len__(self) -> int:
        return len(self.input_sequences)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_sequences[idx],
            self.target_sequences[idx], 
            self.loss_masks[idx]
        )
        
