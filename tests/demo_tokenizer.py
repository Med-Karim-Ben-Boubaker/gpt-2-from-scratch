from src.data.tokenizer import get_english_tokenizer, text_to_token_ids, token_ids_to_text


def main() -> None:
    """Run tokenizer demo."""
    tokenizer = get_english_tokenizer()
    
    # load the txt file
    corpus = open("data/synthetic-data/3.txt", "r").read()
    
    print("English Tokenizer Demo")
    print("=" * 50)
    
    token_ids = text_to_token_ids(corpus, tokenizer)
    decoded = token_ids_to_text(token_ids, tokenizer)
    
    # print(f"Tokens: {token_ids.squeeze().tolist()}")
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()
