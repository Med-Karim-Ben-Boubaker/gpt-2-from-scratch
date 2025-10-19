import json
import logging
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class EnglishTokenizer:
    """English-aware tokenizer using A2-level vocabulary."""
    
    def __init__(self, vocab_path: str = "data/english-words.json") -> None:
        """Initialize tokenizer with vocabulary from JSON file.
        
        Args:
            vocab_path: Path to vocabulary JSON file.
        """
        self.vocab_path = Path(vocab_path)
        self.vocab = self._load_vocabulary()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
    def _load_vocabulary(self) -> List[str]:
        """Load vocabulary from JSON file."""
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def encode(self, text: str, allowed_special: Set[str] = None) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to tokenize.
            allowed_special: Set of allowed special tokens (unused).
            
        Returns:
            List of token IDs.
        """
        if allowed_special is None:
            allowed_special = set()
            
        text = text.lower()
        text = text.replace("<|endoftext|>", "<eot>")
        
        words = text.split()
        token_ids = []
        
        for word in words:
            token_ids.extend(self._tokenize_word(word))
            
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs.
            
        Returns:
            Decoded text.
        """
        tokens = [self.id_to_token.get(token_id, "<unk>") for token_id in token_ids]
        return " ".join(tokens)
    
    def _tokenize_word(self, word: str) -> List[int]:
        """Tokenize a single word using greedy longest-match."""
        if word in self.token_to_id:
            return [self.token_to_id[word]]
        
        tokens = []
        remaining = word
        
        while remaining:
            matched = False
            
            for length in range(len(remaining), 0, -1):
                candidate = remaining[:length]
                if candidate in self.token_to_id:
                    tokens.append(self.token_to_id[candidate])
                    remaining = remaining[length:]
                    matched = True
                    break
            
            if not matched:
                logger.warning(f"Unknown token: '{remaining[0]}'")
                tokens.append(self.token_to_id["<unk>"])
                remaining = remaining[1:]
        
        return tokens
