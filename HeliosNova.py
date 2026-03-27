"""
Helios Nova  —  A 306M-parameter dense language model
======================================================
Successor to Helios-Tiny.  Designed for maximum coherence at small scale,
trained on FineWeb-Edu 10BT with a 16K BPE tokenizer.

Architecture
------------
• Dense transformer:  24 layers, no weight sharing.  Every layer has its own
  parameters — maximises capacity per FLOP at this scale.
• SwiGLU FFN:         gated activation (Shazeer 2020) for better parameter
                      efficiency vs standard MLP.
• Grouped-Query Attention (GQA):  4 KV heads serve 16 query heads, cutting
                      KV-cache by 4× with negligible quality loss.
• QK-Norm:            RMSNorm on queries and keys before the dot product.
                      Stabilises training and prevents attention entropy
                      collapse, especially at higher learning rates.
• RoPE:               rotary position embeddings (Su et al. 2021).
                      No learned positional embedding needed.
• RMSNorm:            pre-norm architecture, no bias anywhere.
• Tied embeddings:    input/output weight sharing saves ~16M params.

HuggingFace compatibility
-------------------------
• `HeliosNova.save_pretrained(path)` writes config.json + model.safetensors.
• `HeliosNova.from_pretrained(path)` loads them back.
• Works with the tokenizer at `respinosamena/Helios-Nova` on the Hub.

Parameter count:  ~306M (d=1024, L=24, FFN=3072, GQA 16q/4kv, head=64)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class HeliosNovaConfig:
    """All architectural hyper-parameters for Helios Nova."""

    vocab_size:      int   = 16384       # BPE vocabulary (power of 2 for GPU efficiency)
    d_model:         int   = 1024        # residual stream / embedding width
    n_heads:         int   = 16          # query heads in grouped-query attention
    n_kv_heads:      int   = 4           # key-value heads (GQA ratio = n_heads / n_kv_heads)
    head_dim:        int   = 64          # dimension per attention head
    ffn_dim:         int   = 3072        # SwiGLU intermediate size (~3× d_model)
    n_layers:        int   = 24          # transformer blocks (all unique — no weight sharing)
    max_seq_len:     int   = 2048        # maximum context length in BPE tokens
    dropout:         float = 0.0         # dropout rate (0 = off; plenty of data)
    rope_theta:      float = 10_000.0    # RoPE base frequency
    norm_eps:        float = 1e-6        # RMSNorm epsilon
    tie_embeddings:  bool  = True        # share input embed ↔ output projection weights
    qk_norm:         bool  = True        # RMSNorm on Q and K before attention dot product

    # ── Derived ──────────────────────────────────────────────────────────
    @property
    def gqa_groups(self) -> int:
        """Number of query heads per KV head."""
        return self.n_heads // self.n_kv_heads

    # ── Serialisation ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["model_type"] = "helios_nova"
        return d

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "HeliosNovaConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def load(cls, path: str | Path) -> "HeliosNovaConfig":
        with open(Path(path) / "config.json") as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════════
#  RMSNorm  (Zhang & Sennrich 2019)
# ═══════════════════════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation — no bias, no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


# ═══════════════════════════════════════════════════════════════════════════════
#  Rotary Position Embedding  (Su et al. 2021)
# ═══════════════════════════════════════════════════════════════════════════════
def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables for RoPE up to `seq_len` positions."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)                         # (seq_len, head_dim/2)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a (B, n_heads, T, head_dim) tensor."""
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)                # (1, 1, T, head_dim/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Grouped-Query Attention  with QK-Norm
# ═══════════════════════════════════════════════════════════════════════════════
class GQAttention(nn.Module):
    """
    Multi-head attention with grouped key-value heads (Ainslie et al. 2023).

    When `qk_norm=True`, queries and keys are independently normalised with
    RMSNorm *before* the dot product.  This prevents the attention logits from
    growing unboundedly, stabilising training especially at higher LRs and
    longer context lengths  (Dehghani et al. 2023, "Scaling ViTs").
    """

    def __init__(self, cfg: HeliosNovaConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.head_dim
        self.groups     = cfg.gqa_groups
        self.scale      = cfg.head_dim ** -0.5

        # Linear projections — no bias (standard for modern LLMs)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads    * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim,  cfg.d_model,   bias=False)

        # QK-Norm: per-head RMSNorm on Q and K (applied before RoPE)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps)

        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to multi-head Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm before RoPE for best stability
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Rotary position embeddings
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match query heads for GQA
        if self.groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.groups, -1, -1) \
                 .reshape(B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1) \
                 .reshape(B, self.n_heads, T, self.head_dim)

        # Scaled dot-product attention (uses FlashAttention-2 on A100)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )

        # Merge heads → project back to d_model
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.resid_drop(self.o_proj(out))


# ═══════════════════════════════════════════════════════════════════════════════
#  SwiGLU Feed-Forward  (Shazeer 2020)
# ═══════════════════════════════════════════════════════════════════════════════
class SwiGLUFFN(nn.Module):
    """
    Gated feed-forward with SiLU activation.
    Effective hidden size is `ffn_dim` (gate and up are separate projections).
    Total FFN params = 3 × d_model × ffn_dim.
    """

    def __init__(self, cfg: HeliosNovaConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up   = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Transformer Block  (pre-norm)
# ═══════════════════════════════════════════════════════════════════════════════
class TransformerBlock(nn.Module):
    """Pre-norm block: RMSNorm → Attention → + residual → RMSNorm → FFN → + residual."""

    def __init__(self, cfg: HeliosNovaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn      = GQAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn       = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Helios Nova  (full model)
# ═══════════════════════════════════════════════════════════════════════════════
class HeliosNova(nn.Module):
    """
    Causal language model — 306M dense parameters.

    Forward pass returns (logits, loss) where loss is computed if targets
    are provided.  Compatible with HuggingFace-style save/load via
    `save_pretrained` and `from_pretrained`.
    """

    def __init__(self, cfg: HeliosNovaConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding (shared with output projection when tie_embeddings=True)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Transformer body — all layers are unique (no weight sharing)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # Final layer norm before the LM head
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        # Language modelling head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Pre-computed RoPE cos/sin buffers (not saved in state_dict)
        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Weight initialisation
        self.apply(self._init_weights)

        # Scale residual-path projections by 1/√(2·depth) to keep
        # variance stable through the deep residual stream
        depth_scale = (2 * cfg.n_layers) ** -0.5
        for layer in self.layers:
            layer.attn.o_proj.weight.data *= depth_scale
            layer.ffn.down.weight.data   *= depth_scale

    # ── Initialisation ───────────────────────────────────────────────────
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ── Forward pass ─────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids:  (B, T)  token indices
            targets:    (B, T)  next-token labels, or None for inference

        Returns:
            logits:  (B, T, vocab_size)
            loss:    scalar cross-entropy, or None
        """
        x = self.tok_emb(input_ids)

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)

        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ── Generation ───────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Auto-regressive sampling with top-k filtering."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        self.train()
        return input_ids

    # ── Utilities ────────────────────────────────────────────────────────
    def param_count(self, only_trainable: bool = True) -> int:
        """Total (trainable) parameters, accounting for tied weights."""
        return sum(p.numel() for p in self.parameters() if not only_trainable or p.requires_grad)

    # ── HuggingFace-style persistence ────────────────────────────────────
    def save_pretrained(self, path: str | Path) -> None:
        """Save config.json + model.safetensors to `path`."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Config
        self.cfg.save(path)

        # Weights in safetensors format (fast, safe, HF-standard)
        state = {k: v.contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state, path / "model.safetensors")

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path, device: str = "cpu") -> "HeliosNova":
        """
        Load a Helios Nova model from a local directory or a HuggingFace repo ID.

        If `path_or_repo` is a local directory containing config.json, loads
        directly.  Otherwise treats it as a HF Hub repo ID and downloads the
        files (cached after first download).
        """
        from safetensors.torch import load_file

        local = Path(path_or_repo)
        if local.is_dir() and (local / "config.json").exists():
            # Local directory
            cfg = HeliosNovaConfig.load(local)
            model = cls(cfg)
            state = load_file(local / "model.safetensors", device=device)
        else:
            # HuggingFace Hub repo ID → download individual files
            from huggingface_hub import hf_hub_download

            repo_id = str(path_or_repo)
            cfg_path = hf_hub_download(repo_id, "config.json")
            cfg = HeliosNovaConfig.load(Path(cfg_path).parent)
            model = cls(cfg)
            weights_path = hf_hub_download(repo_id, "model.safetensors")
            state = load_file(weights_path, device=device)

        model.load_state_dict(state, strict=True)
        return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = HeliosNovaConfig()
    m = HeliosNova(cfg)
    n = m.param_count()
    print(f"Helios Nova  |  {n:,} params ({n / 1e6:.2f}M)")
    print(f"  {cfg.n_layers} layers  |  d={cfg.d_model}  |  GQA {cfg.n_heads}q/{cfg.n_kv_heads}kv")
    print(f"  FFN={cfg.ffn_dim}  |  head_dim={cfg.head_dim}  |  QK-Norm={cfg.qk_norm}")
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss = m(x, targets=x)
    print(f"  logits: {logits.shape}  loss: {loss.item():.4f}")
