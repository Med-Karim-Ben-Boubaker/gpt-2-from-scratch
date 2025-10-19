import json
import torch
import yaml
from src.config import GPTConfig
from src.data.tokenizer import text_to_token_ids, token_ids_to_text, get_bpe_tokenizer
from src.training.trainer import get_eos_token_id
from src.models.gpt import GPTModel
from src.utils.logging import get_logger
from src.training.generate import generate

logger = get_logger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    config_path = "configs/gpt2_59m_10heads_10layers.yaml"
    model_checkpoint_path = "artifacts/gpt2_59m_10heads_10layers.pth"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    gpt_config = GPTConfig(**raw["model"])

    model = GPTModel(gpt_config)
    model = model.to(device)

    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint["model"])
    model.eval()

    logger.info("Model loaded successfully")

    prompt = """Computer science is the study of algorithms and computational processes."""

    tokenizer = get_bpe_tokenizer()
    input_token_ids = text_to_token_ids(prompt, tokenizer).to(device)

    eos_token_id = get_eos_token_id(tokenizer)

    max_new_tokens = 256
    logger.info("Starting generation...")
    output_token_ids, trace_data = generate(
        model, 
        input_token_ids, 
        max_new_tokens, 
        gpt_config.context_length,
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=eos_token_id,
        min_new_tokens=1,
        trace=True,
        topk_log=10,
        verbose=True,
    )
    
    generated_text = token_ids_to_text(output_token_ids, tokenizer)
    logger.info(f"Generated text: {generated_text}")
    
    # Save trace data if available
    if trace_data:
        with open("artifacts/trace.json", "w") as f:
            json.dump(trace_data, f, indent=2)
        logger.info("Trace data saved to artifacts/trace.json")

if __name__ == "__main__":
    main()