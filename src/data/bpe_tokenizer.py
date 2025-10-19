import tempfile
import os
from pathlib import Path
from typing import List, Union, Dict

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders


class BPETokenizer:

    def __init__(self, tokenizer: Tokenizer = None) -> None:
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            self._tokenizer.normalizer = normalizers.Lowercase()
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            self._tokenizer.decoder = decoders.ByteLevel()

    def train(
        self,
        corpus: Union[str, List[str]],
        vocab_size: int = 8000,
        special_tokens: List[str] = None,
        is_text: bool = False,
    ) -> None:

        if special_tokens is None:
            special_tokens = ["<unk>", "<sys>", "<user>", "<eot>", "<ctx>"]

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens
        )

        if is_text:
            # corpus is text content, create temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(corpus if isinstance(corpus, str) else str(corpus))
                temp_file_path = temp_file.name

            try:
                self._tokenizer.train(files=[temp_file_path], trainer=trainer)
            finally:
                os.unlink(temp_file_path)
        else:
            # corpus is file paths
            files = (
                [corpus] if isinstance(corpus, str) else corpus
            )
            self._tokenizer.train(files=files, trainer=trainer)

    def save(self, file_path: Union[str, Path]) -> None:

        directory = Path(file_path).parent
        directory.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(file_path))

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "BPETokenizer":

        tokenizer = Tokenizer.from_file(str(file_path))
        return cls(tokenizer)

    def encode(self, text: str, allowed_special: dict = None) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def vocab(self) -> Dict[str, int]:
        """Return token to ID mapping."""
        return self._tokenizer.get_vocab()
