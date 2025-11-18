import torch
import torch.nn as nn

class LayerNorm(nn.Module):
  def __init__(self, emb_dim: int):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=True)
    x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * x + self.shift

class FeedForward(nn.Module):
  def __init__(self, emb_dim: int):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(emb_dim, 4 * emb_dim),
      nn.GELU(),
      nn.Linear(4 * emb_dim, emb_dim),
    )
  def forward(self, x): return self.layers(x)

class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert d_out % num_heads == 0
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    b, t, _ = x.shape
    k = self.W_key(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
    q = self.W_query(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.W_value(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
    att = (q @ k.transpose(2, 3)) / (k.shape[-1] ** 0.5)
    m = self.mask.bool()[:t, :t]
    att = att.masked_fill(m, float("-inf"))
    w = torch.softmax(att, dim=-1)
    ctx = w @ v
    ctx = ctx.transpose(1, 2).contiguous().view(b, t, self.d_out)
    return self.out_proj(ctx)

class TransformerBlock(nn.Module):
  def __init__(self, emb_dim, context_length, n_heads, drop_rate, qkv_bias):
    super().__init__()
    self.att = MultiHeadAttention(emb_dim, emb_dim, context_length, drop_rate, n_heads, qkv_bias)
    self.ff = FeedForward(emb_dim)
    self.norm1 = LayerNorm(emb_dim)
    self.norm2 = LayerNorm(emb_dim)
    self.drop = nn.Dropout(drop_rate)
  def forward(self, x):
    x = x + self.drop(self.att(self.norm1(x)))
    x = x + self.drop(self.ff(self.norm2(x)))
    return x