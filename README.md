# LLM from Scratch

Educational implementation of a GPT-style language model in PyTorch, with training, instruction fine-tuning, and basic inference utilities.

## Highlights

- GPT-style Transformer model and layers (`src/models/`)
- Tokenization and dataloaders (`src/data/`)
- Training loop, evaluation, and text generation (`src/training/`)
- YAML-based configuration (`configs/`)
- TensorBoard logging (`runs/`) and checkpoint outputs (`artifacts/`)

## Requirements

- Python 3.11+
- `uv` for environment and dependency management

Install dependencies:

```bash
uv sync
```

## Data

The training scripts expect text data under the top-level `data/` directory.

- **Pretraining data**: `scripts/train.py` reads `data/fineweb_samples.txt`.
- **Instruction data**: `scripts/train_instruct.py` reads `data/alpaca_data.json`.

To generate a FineWeb text file via Hugging Face Datasets:

```bash
uv run python scripts/data/generate_data.py
```

This generates `data/fineweb.txt`. If you want to use it with `scripts/train.py` without editing code, create `data/fineweb_samples.txt` (for example, by copying or sampling from `data/fineweb.txt`).

## Training (pretraining)

Run the default training configuration:

```bash
uv run python -m scripts.train
```

Select a config file via the `CFG` environment variable:

```bash
CFG=configs/gpt_124m.yaml uv run python -m scripts.train
```

Outputs:

- Checkpoints: `artifacts/`
- TensorBoard logs: `runs/`

View TensorBoard:

```bash
uv run tensorboard --logdir runs
```

## Instruction fine-tuning

The instruction fine-tuning entrypoint is `scripts/train_instruct.py`. It expects:

- Instruction dataset at `data/alpaca_data.json`
- A base model checkpoint in `artifacts/` (see the path defined in `scripts/train_instruct.py`)

Run:

```bash
CFG=configs/gpt2_35m_4heads_12layers_finetuning.yaml uv run python -m scripts.train_instruct
```

## Inference

The inference entrypoint is `scripts/infere.py` and loads a checkpoint from `artifacts/` (see the path defined in the script).

Run:

```bash
uv run python -m scripts.infere
```

## Project layout

- `configs/`: model and training YAML configurations
- `scripts/`: CLI entrypoints (training, fine-tuning, inference)
- `src/`: library code (model, data, training, utilities)
- `notebooks/`: experiments and analysis notebooks
- `tests/`: ad hoc test and utility scripts

## Code quality

Format and lint with Ruff:

```bash
uv run ruff format .
uv run ruff check .
```
