import torch
import tiktoken
from src.data.english_tokenizer import EnglishTokenizer
from src.data.bpe_tokenizer import BPETokenizer
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_tokenizer(name: str = "gpt2") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def get_english_tokenizer() -> EnglishTokenizer:
    """Get English-aware tokenizer using A2-level vocabulary."""
    return EnglishTokenizer()

def get_bpe_tokenizer(
    tokenizer_path: Path = Path("artifacts/tokenizers/bpe_tokenizer.json"),
    corpus: str = None,
    vocab_size: int = 8000,
    force_retrain: bool = False,
) -> BPETokenizer:

    if tokenizer_path.exists() and not force_retrain:
        logger.info(f"Loading BPE tokenizer from {tokenizer_path}")
        return BPETokenizer.from_file(str(tokenizer_path))

    if corpus is None:
        raise ValueError("corpus parameter is required when training a new tokenizer")

    logger.info("Training BPE tokenizer from provided corpus text")
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=vocab_size, is_text=True)
    logger.info(f"Saving BPE tokenizer to {tokenizer_path}")
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    encoded_text = tokenizer.encode(text)
    return torch.tensor(encoded_text).unsqueeze(0)  # add a dimension to the tensor for the batch dimension

def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())  # remove the dimension from the tensor.
