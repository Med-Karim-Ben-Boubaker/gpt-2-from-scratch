from datasets import load_dataset
from src.utils.logging import get_logger
import os

logger = get_logger(__name__)

DATASET_NAME = "roneneldan/TinyStories"
NUM_SAMPLES = 200000
OUTPUT_FILE = "data/tinystories.txt"

logger.info(f"Loading {NUM_SAMPLES} samples from {DATASET_NAME}...")

dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

os.makedirs("data", exist_ok=True)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for i, sample in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break
        f.write(sample['text'] + ' <eot>' +'\n\n')
        if (i + 1) % 100000 == 0:
            logger.info(f"Processed {i + 1}/{NUM_SAMPLES} samples...")

logger.info(f"✅ Saved {NUM_SAMPLES} samples to {OUTPUT_FILE}")
logger.info(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")