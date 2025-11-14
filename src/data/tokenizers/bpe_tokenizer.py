from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from src.data.tokenizers.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding tokenizer powered by HuggingFace tokenizers."""

    DEFAULT_SPECIAL_TOKENS = ["<unk>", "<sys>", "<user>", "<eot>", "<ctx>"]

    def __init__(self, tokenizer: Optional[Tokenizer] = None) -> None:
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            self._tokenizer.normalizer = normalizers.Lowercase()
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            self._tokenizer.decoder = decoders.ByteLevel()

    def train(
        self,
        corpus: Union[str, Path, List[Union[str, Path]]],
        vocab_size: int = 8000,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """Train the tokenizer from raw text or corpus files."""

        special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

        if isinstance(corpus, (str, Path)) and Path(corpus).exists():
            files = [str(corpus)]
        elif isinstance(corpus, list):
            files = [str(item) for item in corpus]
            if not all(Path(item).exists() for item in files):
                raise ValueError("All paths provided in corpus must exist.")
        else:
            files = None

        if files:
            self._tokenizer.train(files=files, trainer=trainer)
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
                temp_file.write(corpus if isinstance(corpus, str) else str(corpus))
                temp_file_path = temp_file.name
            try:
                self._tokenizer.train(files=[temp_file_path], trainer=trainer)
            finally:
                os.unlink(temp_file_path)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save the trained tokenizer to disk."""

        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(target_path))

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "BPETokenizer":
        """Load a tokenizer previously saved with `save()`."""

        tokenizer = Tokenizer.from_file(str(file_path))
        return cls(tokenizer=tokenizer)

    def encode(
        self,
        text: str,
    ) -> List[int]:
        """Encode text into token IDs."""

        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""

        return self._tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size exposed through the HuggingFace tokenizer."""

        return self._tokenizer.get_vocab_size()

    @property
    def vocab(self) -> Dict[str, int]:
        """Token to ID mapping used internally."""

        return self._tokenizer.get_vocab()
    
    
# Small Test
if __name__ == "__main__":
    tokenizer = BPETokenizer()
    
    print("BPE Tokenizer Test with Corpus file path")
    print("=" * 50)
    corpus_file = Path("data/synthetic-data/3.txt")
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_file}")
    tokenizer.train(corpus=corpus_file)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.vocab}")
    tokenizer_text = tokenizer.encode('This text will be tokenized!!!')
    print(f"Sample encoded text: {tokenizer_text}")
    print(f"Sample decoded text: {tokenizer.decode(tokenizer_text)}")
    

