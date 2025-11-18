import torch
import torch.nn as nn
from src.models.gpt.layers import TransformerBlock, LayerNorm

class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
    self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
    self.drop_emb = nn.Dropout(cfg.drop_rate)
    self.trf_blocks = nn.Sequential(*[
      TransformerBlock(cfg.emb_dim, cfg.context_length, cfg.n_heads, cfg.drop_rate, cfg.qkv_bias)
      for _ in range(cfg.n_layers)
    ])
    self.final_norm = LayerNorm(cfg.emb_dim)
    self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
    # Share the weights of the output head with the token embedding layer
    self.out_head.weight = self.tok_emb.weight

  def forward(self, in_idx):
    b, t = in_idx.shape
    x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(t, device=in_idx.device))
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    return self.out_head(x)
  
# Small Test
if __name__ == "__main__":
  from src.config import GPTConfig
  cfg = GPTConfig(
    vocab_size=8000,
    context_length=512,
    emb_dim=384,
    n_heads=6,
    n_layers=18,
    drop_rate=0.1,
    qkv_bias=False,
  )
  model = GPTModel(cfg)
  print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")