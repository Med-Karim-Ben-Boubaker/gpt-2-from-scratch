from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
_DEFAULT_CONFIG_NAME = "gpt2_59m_10heads_10layers.yaml"


def _resolve_config_path(config_name: Union[str, Path]) -> Path:
  path = Path(config_name)
  if not path.is_absolute():
    path = _CONFIG_ROOT / path
  path = path.expanduser().resolve()
  if not path.exists():
    raise FileNotFoundError(f"Configuration not found at {path}")
  return path


def _load_model_config(path: Path) -> Dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    raw = yaml.safe_load(fh)
  model_data = raw.get("model") if isinstance(raw, dict) else None
  if not isinstance(model_data, dict):
    raise ValueError(f"Model section missing or invalid in {path}")
  return model_data


def _load_train_config(path: Path) -> Dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    raw = yaml.safe_load(fh)
  train_data = raw.get("train") if isinstance(raw, dict) else None
  if not isinstance(train_data, dict):
    raise ValueError(f"Train section missing or invalid in {path}")
  return train_data


@dataclass
class GPTConfig:
  vocab_size: int = 50257
  context_length: int = 128
  emb_dim: int = 256
  n_heads: int = 8
  n_layers: int = 6
  drop_rate: float = 0.1
  qkv_bias: bool = False

  @classmethod
  def from_yaml(cls, config_name: Union[str, Path] = _DEFAULT_CONFIG_NAME) -> "GPTConfig":
    path = _resolve_config_path(config_name)
    model_data = _load_model_config(path)
    return cls(**model_data)
  
@dataclass
class TrainConfig:
  batch_size: int = 4
  lr: float = 4e-4
  weight_decay: float = 0.01
  num_epochs: int = 10
  eval_freq: int = 100
  eval_iter: int = 50
  grad_accum_steps: int = 1
  amp: bool = True
  device: str = "cuda"
  seed: int = 123
  num_workers: int = 0
  warmup_steps: int = 1450
  min_lr: float = 1e-5
  betas: tuple = (0.9, 0.999)
  eps: float = 1e-8
  fused: bool = True
  grad_clip_norm: float = 1.0

  @classmethod
  def from_yaml(cls, config_name: Union[str, Path] = _DEFAULT_CONFIG_NAME) -> "TrainConfig":
    path = _resolve_config_path(config_name)
    train_data = _load_train_config(path)
    # Convert betas list to tuple if it's a list in YAML
    if "betas" in train_data and isinstance(train_data["betas"], list):
      train_data["betas"] = tuple(train_data["betas"])
    return cls(**train_data)