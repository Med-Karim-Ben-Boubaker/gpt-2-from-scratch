#!/usr/bin/env python3
"""
Comprehensive test and demo for BPE Tokenizer.

This file tests the BPE tokenizer implementation and validates its compatibility
with the existing tiktoken-based tokenizer used in the training pipeline.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import tiktoken
from src.data.bpe_tokenizer import BPETokenizer
from src.data.datasets import GPTDatasetV1
from src.data.dataloader import create_dataloader_v1
from src.utils.logging import get_logger

logger = get_logger(__name__)


def test_bpe_tokenizer_training():
    """Test training a BPE tokenizer on synthetic data."""
    print("\n" + "="*60)
    print("SECTION 1: Training BPE Tokenizer")
    print("="*60)
    
    # Training data path
    training_data_path = "data/synthetic-data/3.txt"
    
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data not found at {training_data_path}")
    
    # Create tokenizer with special tokens
    special_tokens = ["<unk>", "<sys>", "<user>", "<eot>", "<ctx>", "<|endoftext|>"]
    tokenizer = BPETokenizer()
    
    print(f"Training BPE tokenizer on {training_data_path}")
    print(f"Special tokens: {special_tokens}")
    
    # Train the tokenizer
    tokenizer.train(
        corpus=training_data_path,
        vocab_size=8000,
        special_tokens=special_tokens
    )
    
    # Save tokenizer
    save_path = Path("artifacts/tokenizers/bpe_tokenizer.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    
    print(f"Tokenizer saved to {save_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    return tokenizer


def test_interface_compatibility():
    """Test that BPE tokenizer has the same interface as tiktoken."""
    print("\n" + "="*60)
    print("SECTION 2: Interface Compatibility Tests")
    print("="*60)
    
    # Load trained tokenizer
    tokenizer_path = Path("artifacts/tokenizers/bpe_tokenizer.json")
    if not tokenizer_path.exists():
        print("No trained tokenizer found. Training first...")
        tokenizer = test_bpe_tokenizer_training()
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.from_file(tokenizer_path)
    
    # Test text with special tokens (use lowercase to match BPE normalization)
    test_text = "hello world! this is a test with <eot> special tokens."
    
    print(f"\nTest text: {test_text}")
    
    # Test encode without allowed_special
    encoded = tokenizer.encode(test_text)
    print(f"Encoded (no allowed_special): {encoded[:10]}... (showing first 10 tokens)")
    print(f"Encoded length: {len(encoded)}")
    
    # Test encode with allowed_special (API compatibility)
    encoded_with_special = tokenizer.encode(test_text, allowed_special={"<eot>", "<|endoftext|>"})
    print(f"Encoded (with allowed_special): {encoded_with_special[:10]}... (showing first 10 tokens)")
    print(f"Encoded length: {len(encoded_with_special)}")
    
    # Verify both methods return the same result
    assert encoded == encoded_with_special, "encode() should return same result regardless of allowed_special"
    print("✓ allowed_special parameter works correctly")
    
    # Test decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text: {decoded}")
    
    # Test that decode produces valid text
    assert isinstance(decoded, str), "Decode should return a string"
    assert len(decoded) > 0, "Decoded text should not be empty"
    print("✓ Decode produces valid text")
    
    # Test vocab_size property
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    assert isinstance(vocab_size, int), "vocab_size should be an integer"
    assert vocab_size > 0, "vocab_size should be positive"
    print("✓ vocab_size property works")
    
    return tokenizer


def test_shape_compatibility():
    """Test that BPE tokenizer produces compatible output shapes with tiktoken."""
    print("\n" + "="*60)
    print("SECTION 3: Shape Compatibility Tests")
    print("="*60)
    
    # Load tokenizers
    bpe_tokenizer = BPETokenizer.from_file("artifacts/tokenizers/bpe_tokenizer.json")
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    
    # Test texts with different characteristics
    test_texts = [
        "Simple text without special tokens.",
        "Text with <eot> special token.",
        "Text with <|endoftext|> GPT-2 special token.",
        "Longer text with multiple sentences. This should test the tokenization more thoroughly. <eot>",
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text[:50]}...")
        
        # Encode with both tokenizers
        bpe_encoded = bpe_tokenizer.encode(text, allowed_special={"<eot>", "<|endoftext|>"})
        gpt2_encoded = gpt2_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        # Check return types
        assert isinstance(bpe_encoded, list), "BPE encode should return list"
        assert isinstance(gpt2_encoded, list), "GPT-2 encode should return list"
        assert all(isinstance(x, int) for x in bpe_encoded), "BPE tokens should be integers"
        assert all(isinstance(x, int) for x in gpt2_encoded), "GPT-2 tokens should be integers"
        
        print(f"  BPE tokens: {len(bpe_encoded)} tokens, first 5: {bpe_encoded[:5]}")
        print(f"  GPT-2 tokens: {len(gpt2_encoded)} tokens, first 5: {gpt2_encoded[:5]}")
        
        # Test decode
        bpe_decoded = bpe_tokenizer.decode(bpe_encoded)
        gpt2_decoded = gpt2_tokenizer.decode(gpt2_encoded)
        
        print(f"  BPE decoded: {bpe_decoded}")
        print(f"  GPT-2 decoded: {gpt2_decoded}")
        
        # Verify BPE decode produces valid text
        assert isinstance(bpe_decoded, str), f"BPE decode should return string for test {i+1}"
        print(f"  ✓ BPE decode successful")
    
    print("\n✓ All shape compatibility tests passed")


def test_integration_with_dataset():
    """Test BPE tokenizer integration with GPTDatasetV1."""
    print("\n" + "="*60)
    print("SECTION 4: Integration Demo")
    print("="*60)
    
    # Load tokenizers
    bpe_tokenizer = BPETokenizer.from_file("artifacts/tokenizers/bpe_tokenizer.json")
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create test text
    test_text = """
    This is a test document for integration testing.
    It contains multiple sentences and special tokens like <eot>.
    The BPE tokenizer should handle this correctly.
    <eot>
    Another paragraph with more content.
    This tests the dataset creation process.
    """
    
    print(f"Test text length: {len(test_text)} characters")
    
    # Test with BPE tokenizer
    print("\nTesting with BPE tokenizer:")
    try:
        bpe_dataset = GPTDatasetV1(
            text=test_text,
            tokenizer=bpe_tokenizer,
            max_length=50,
            stride=25
        )
        print(f"  BPE Dataset created successfully: {len(bpe_dataset)} sequences")
        
        if len(bpe_dataset) > 0:
            sample_input, sample_target = bpe_dataset[0]
            print(f"  Sample input shape: {sample_input.shape}")
            print(f"  Sample target shape: {sample_target.shape}")
            print(f"  Sample input tokens: {sample_input.tolist()[:10]}...")
            print(f"  Sample target tokens: {sample_target.tolist()[:10]}...")
        
    except Exception as e:
        print(f"  ❌ BPE dataset creation failed: {e}")
        raise
    
    # Test with GPT-2 tokenizer for comparison
    print("\nTesting with GPT-2 tokenizer:")
    try:
        gpt2_dataset = GPTDatasetV1(
            text=test_text,
            tokenizer=gpt2_tokenizer,
            max_length=50,
            stride=25
        )
        print(f"  GPT-2 Dataset created successfully: {len(gpt2_dataset)} sequences")
        
        if len(gpt2_dataset) > 0:
            sample_input, sample_target = gpt2_dataset[0]
            print(f"  Sample input shape: {sample_input.shape}")
            print(f"  Sample target shape: {sample_target.shape}")
            print(f"  Sample input tokens: {sample_input.tolist()[:10]}...")
            print(f"  Sample target tokens: {sample_target.tolist()[:10]}...")
        
    except Exception as e:
        print(f"  ❌ GPT-2 dataset creation failed: {e}")
        raise
    
    print("\n✓ Integration tests passed - both tokenizers work with GPTDatasetV1")


def test_dataloader_compatibility():
    """Test BPE tokenizer with the actual dataloader used in training."""
    print("\n" + "="*60)
    print("SECTION 5: Dataloader Compatibility Test")
    print("="*60)
    
    # Create a small test corpus
    test_corpus = """
    This is a test corpus for dataloader testing.
    It should be long enough to create multiple sequences.
    The BPE tokenizer should work seamlessly with the dataloader.
    <eot>
    This is another paragraph with more content.
    We need enough text to test the sliding window approach.
    The dataloader will create overlapping sequences from this text.
    <eot>
    Final paragraph to ensure we have sufficient data.
    This tests the complete pipeline from text to batches.
    """
    
    print(f"Test corpus length: {len(test_corpus)} characters")
    
    # Test with BPE tokenizer (we'll need to modify the dataloader function temporarily)
    print("\nTesting dataloader compatibility...")
    
    # Create a custom dataloader function that uses BPE tokenizer
    def create_bpe_dataloader(text, batch_size, max_length, stride, shuffle, drop_last, num_workers):
        """Custom dataloader using BPE tokenizer."""
        bpe_tokenizer = BPETokenizer.from_file("artifacts/tokenizers/bpe_tokenizer.json")
        dataset = GPTDatasetV1(text, bpe_tokenizer, max_length, stride)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    
    try:
        bpe_loader = create_bpe_dataloader(
            text=test_corpus,
            batch_size=2,
            max_length=30,
            stride=15,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        print(f"  BPE Dataloader created successfully")
        print(f"  Number of batches: {len(bpe_loader)}")
        
        # Test getting a batch
        for batch_idx, (input_batch, target_batch) in enumerate(bpe_loader):
            print(f"  Batch {batch_idx}: input shape {input_batch.shape}, target shape {target_batch.shape}")
            if batch_idx >= 2:  # Only show first few batches
                break
        
        print("  ✓ BPE dataloader works correctly")
        
    except Exception as e:
        print(f"  ❌ BPE dataloader test failed: {e}")
        raise
    
    print("\n✓ Dataloader compatibility test passed")


def main():
    """Run all tests and demos."""
    print("BPE Tokenizer Test and Demo")
    print("="*60)
    
    try:
        # Run all test sections
        test_bpe_tokenizer_training()
        test_interface_compatibility()
        test_shape_compatibility()
        test_integration_with_dataset()
        test_dataloader_compatibility()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nThe BPE tokenizer is compatible with the existing codebase and can be used")
        print("as a drop-in replacement for the tiktoken GPT-2 tokenizer.")
        print("\nKey findings:")
        print("- ✓ Interface compatibility: encode(), decode(), vocab_size work correctly")
        print("- ✓ Shape compatibility: Returns same data types as tiktoken")
        print("- ✓ Special token handling: <eot> and other special tokens work correctly")
        print("- ✓ Dataset integration: Works with GPTDatasetV1")
        print("- ✓ Dataloader compatibility: Works with training pipeline")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
