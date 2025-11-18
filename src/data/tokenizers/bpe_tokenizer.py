from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )

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
                - top_tokens: Top 10 most frequent tokens with metadata
        """
        token_counter: Counter[int] = Counter()

        # Read corpus and tokenize
        if isinstance(corpus, (str, Path)) and Path(corpus).exists():
            files = [Path(corpus)]
        elif isinstance(corpus, list):
            files = [Path(item) for item in corpus]
            if not all(item.exists() for item in files):
                raise ValueError("All paths provided in corpus must exist.")
        else:
            files = None

        if files:
            for file_path in files:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        token_ids = self.encode(line.strip())
                        token_counter.update(token_ids)
        else:
            text = corpus if isinstance(corpus, str) else str(corpus)
            token_ids = self.encode(text)
            token_counter.update(token_ids)

        # Calculate statistics
        frequencies = list(token_counter.values())
        total_unique_tokens = len(frequencies)

        if total_unique_tokens == 0:
            return {
                "total_unique_tokens": 0,
                "min_frequency": 0,
                "max_frequency": 0,
                "mean_frequency": 0.0,
                "median_frequency": 0.0,
                "percentiles": {"p25": 0, "p50": 0, "p75": 0, "p95": 0, "p99": 0},
                "top_tokens": []
            }

        frequencies_sorted = sorted(frequencies)
        min_freq = frequencies_sorted[0]
        max_freq = frequencies_sorted[-1]
        mean_freq = sum(frequencies) / total_unique_tokens
        median_freq = (
            frequencies_sorted[total_unique_tokens // 2]
            if total_unique_tokens % 2 == 1
            else (
                frequencies_sorted[total_unique_tokens // 2 - 1]
                + frequencies_sorted[total_unique_tokens // 2]
            ) / 2
        )

        # Calculate percentiles
        def get_percentile(percentile: float) -> int:
            index = int((percentile / 100) * (total_unique_tokens - 1))
            return frequencies_sorted[index]

        percentiles = {
            "p25": get_percentile(25),
            "p50": get_percentile(50),
            "p75": get_percentile(75),
            "p95": get_percentile(95),
            "p99": get_percentile(99),
        }

        vocab = self.vocab
        id_to_token = {token_id: token for token, token_id in vocab.items()}
        most_common_tokens = token_counter.most_common(10)
        top_tokens = [
            {
                "token": id_to_token.get(token_id, f"<unk:{token_id}>"),
                "token_id": token_id,
                "frequency": frequency,
            }
            for token_id, frequency in most_common_tokens
        ]

        return {
            "total_unique_tokens": total_unique_tokens,
            "min_frequency": min_freq,
            "max_frequency": max_freq,
            "mean_frequency": mean_freq,
            "median_frequency": median_freq,
            "percentiles": percentiles,
            "top_tokens": top_tokens,
        }
    
    
# Small Test
if __name__ == "__main__":
    tokenizer = BPETokenizer()
    
    print("BPE Tokenizer Test with Corpus file path")
    print("=" * 50)
    corpus_file = Path("data/tinystories.txt")
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_file}")
    tokenizer.train(corpus=corpus_file)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.vocab}")
    tokenizer_text = tokenizer.encode('This text will be tokenized!!!')
    print(f"Sample encoded text: {tokenizer_text}")
    print(f"Sample decoded text: {tokenizer.decode(tokenizer_text)}")
    
    print("=" * 50)
    print("Verifying token frequency distribution...")
    print(tokenizer.verify_token_frequency_distribution(corpus=corpus_file))

