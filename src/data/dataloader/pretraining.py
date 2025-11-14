from torch.utils.data import DataLoader

from src.data.datasets import PretrainingDataset
from src.utils.logging import get_logger
from typing import Any

logger = get_logger(__name__)


def create_pretraining_dataloader(
    text: str,
    tokenizer: Any,
    batch_size: int,
    max_sequence_length: int,
    stride: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
) -> DataLoader:
    logger.info(
        f"Creating dataloader with text length: {len(text)}, Max sequence length: {max_sequence_length}, Stride: {stride}"
    )
    dataset = PretrainingDataset(text, tokenizer, max_sequence_length, stride)
    
    if len(dataset) == 0:
        tokenized_length = len(tokenizer.encode(text))
        raise ValueError(
            f"Cannot create dataloader: dataset is empty. "
            f"Tokenized text length ({tokenized_length}) must be greater than "
            f"max_sequence_length ({max_sequence_length}) to create at least one sequence."
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    
# Small Test:
if __name__ == "__main__":
    from src.data.tokenizers.bpe_tokenizer import BPETokenizer
    text = open("data/synthetic-data/3.txt", "r").read()
    print("Pretraining Dataloader Test")
    print("=" * 50)
    tokenizer = BPETokenizer()
    tokenizer.train(corpus="data/synthetic-data/3.txt")
    print("Tokenizer trained successfully")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("Creating pretraining dataloader...")
    dataloader = create_pretraining_dataloader(
        text=text,
        tokenizer=tokenizer,
        batch_size=32,
        max_sequence_length=10,
        stride=1,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    print(f"Dataloader length: {len(dataloader)}")
    print(f"Dataset length: {len(dataloader.dataset)}")
