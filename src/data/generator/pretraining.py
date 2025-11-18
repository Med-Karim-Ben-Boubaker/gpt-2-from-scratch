import json
import os
from typing import Dict, List, Optional
import ollama
from tqdm import tqdm

from pathlib import Path

from src.utils.logging import get_logger
from src.data.tokenizers.bpe_tokenizer import BPETokenizer

MODEL_NAME = "gemma3:1b-it-qat"
OUTPUT_PATH = "data/synthetic-data/pretraining-data.txt"
TOPICS_CONFIG_PATH = "configs/data_gen_pretraining_topics.json"
TOKENIZER_PATH = "artifacts/tokenizers/bpe_tokenizer.json"

logger = get_logger(__name__)

# Load tokenizer from saved file
tokenizer_path = Path(TOKENIZER_PATH)
if tokenizer_path.exists():
    tokenizer = BPETokenizer.from_file(tokenizer_path)
    logger.info(f"Loaded tokenizer from {tokenizer_path} with vocab size: {tokenizer.vocab_size}")
else:
    raise FileNotFoundError(
        f"Tokenizer not found at {tokenizer_path}. "
        "Please train and save a tokenizer first."
    )

def load_topics() -> List[Dict[str, str]]:
    with open(TOPICS_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_existing_passages() -> int:
    if not os.path.exists(OUTPUT_PATH):
        return 0
    
    count = 0
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if '<eot>' in line:
                count += 1
    return count


def generate_passage(category: str, title: str, notes: str) -> Optional[str]:
    prompt = f"""# Task: Write a Simple English Passage

## Topic Information
- **Subject**: {title}
- **Category**: {category}
- **Context**: {notes}

## Writing Requirements
Write a passage of 200-1000 words that:

### Content Guidelines
- Uses simple, clear English sentences
- Naturally mixes present, past, and future tenses
- Includes at least one question, one negative statement, and one command
- Stays focused on the topic: {title}
- Incorporates the context: {notes}

### Style Requirements
- Write in a natural, conversational tone
- Use varied sentence structures
- Make the content engaging and informative
- Ensure the passage flows logically from start to finish

### Format Requirements
- Write ONLY the passage content
- Do NOT include titles, headings, or markdown formatting
- Do NOT add meta-commentary or notes to yourself
- Do NOT include word counts or instructions
- Start writing the passage immediately after this prompt

## Output
Begin your passage now:"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"].strip() + ' <eot>'
        return content
    
    except Exception:
        return None

def main():
    topics = load_topics()
    existing_passages = count_existing_passages()
    
    logger.info(f"Starting to generate passages from {existing_passages} to {len(topics)}...")
    
    with tqdm(total=len(topics), initial=existing_passages, desc="Generating passages") as pbar:
        total_tokens = 0
        for i in range(existing_passages, len(topics)):
            topic = topics[i]
            title = topic['title']
            category = topic['category']
            notes = topic['notes']
            
            pbar.set_description(f"Generating: {title[:40]}")
            
            for _ in range(3):
                passage = generate_passage(category, title, notes)
                if passage:
                    token_ids = tokenizer.encode(passage)
                    total_tokens += len(token_ids)
                    pbar.set_postfix({"Total tokens": total_tokens})
                    
                    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                        f.write(passage + '\n')
                    break
            
            pbar.update(1)


if __name__ == "__main__":
    main()