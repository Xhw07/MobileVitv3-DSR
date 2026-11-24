"""MobileViTv3-DSR.

MobileViTv3 backbone with two attention modules plugged into the Global branch:

* Spatial-Reduction Attention (SRA)  -- PVT-style; queries come from the full
  token sequence while keys/values are spatially reduced by a strided
  convolution, which cuts the attention cost and suppresses background noise.
* Differential Attention (DA) -- two attention distributions are computed from
  the original query and a learnable offset query; their difference suppresses
  background responses and amplifies traffic-sign regions.

Variant switch (see ``build_model``):
    baseline : plain multi-head self-attention, no SRA, no DA
    sra      : SRA only
    da       : DA only
    dsr      : SRA + DA   (the proposed model)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MobileViTv3DSR",
    "build_model",
    "SpatialReductionAttention",
    "DifferentialAttention",
    "MultiHeadSelfAttention",
]

VARIANTS = {
    "baseline": (False, False),
    "sra": (True, False),
    "da": (False, True),
    "dsr": (True, True),
}


def conv_nxn_bn(inp: int, oup: int, kernel_size: int = 3, stride: int = 1):
    """Conv-BN-GELU block. GELU replaces the SiLU used by the vanilla MobileViT."""
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU(),
    )


def conv_1x1_bn(inp: int, oup: int):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU(),
    )


class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA, used by the baseline and the ``sra``/``da`` ablations."""

    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class SpatialReductionAttention(nn.Module):
    """Spatial-Reduction Attention (PVT-style).

    Queries keep the full token sequence ``N = H * W``; keys and values are
    downsampled by a ``sr_ratio`` strided convolution, so attention runs over
    ``N x N'`` instead of ``N x N``. Reduced sizes are read from the actual
    convolution output, never recomputed by hand, so non-divisible feature maps
    are handled correctly.
    """

    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 4,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            y = x.transpose(1, 2).reshape(b, c, h, w)
            y = self.sr(y).flatten(2).transpose(1, 2)  # (B, N', C), N' from real output
            y = self.norm(y)
        else:
            y = x

        kv = self.kv(y).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class DifferentialAttention(nn.Module):
    """Differential Attention (DA).

    ``A = softmax(Q K^T / sqrt(d)) - lambda * softmax((Q + Q_offset) K^T / sqrt(d))``

    ``Q_offset`` is produced by a zero-initialised linear layer, i.e. a learnable
    offset query branch that starts from the identity mapping; ``lambda`` is a
    per-head learnable scalar in (0, 1). Negative responses are clipped and the
    weights are renormalised so the aggregation stays numerically stable.
    """

    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, lambda_init: float = 0.8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.offset = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.offset.weight)

        init = torch.tensor(lambda_init).clamp(1e-3, 1 - 1e-3)
        self.lambda_logit = nn.Parameter(torch.full((num_heads,), torch.logit(init)))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_off = q + self.offset(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_off = (q_off @ k.transpose(-2, -1)) * self.scale
        attn_off = attn_off.softmax(dim=-1)

        lam = torch.sigmoid(self.lambda_logit).view(1, self.num_heads, 1, 1)
        diff = attn - lam * attn_off
        diff = diff.clamp(min=0.0)
        diff = diff / (diff.sum(dim=-1, keepdim=True) + 1e-6)
        diff = self.attn_drop(diff)

        out = (diff @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block; ``attn_type`` selects MHA or DA."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.0, attn_type: str = "da"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attn_type == "da":
            self.attn = DifferentialAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        elif attn_type == "mha":
            self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        else:
            raise ValueError(f"unknown attn_type: {attn_type}")

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MobileViTBlock(nn.Module):
    """Local (conv) + Global (unfold -> SRA -> Transformer -> fold) joint block.

    ``patch_size`` follows the vanilla MobileViT design: the global branch runs
    on non-overlapping ``patch_size x patch_size`` patches, so the token count
    drops by ``patch_size**2`` before attention and is restored by ``F.fold``.
    """

    def __init__(self, dim: int, depth: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.0, sr_ratio: int = 4, patch_size: int = 2,
                 use_sra: bool = True, use_da: bool = True):
        super().__init__()
        self.patch_size = patch_size
        token_dim = dim * patch_size * patch_size

        self.conv1 = conv_nxn_bn(dim, dim, 3)
        self.conv2 = conv_nxn_bn(dim, dim, 1)
        self.sra = SpatialReductionAttention(token_dim, num_heads, sr_ratio) if use_sra else None
        attn_type = "da" if use_da else "mha"
        self.transformer = nn.Sequential(
            *[TransformerBlock(token_dim, num_heads, mlp_ratio, dropout, attn_type)
              for _ in range(depth)]
        )
        # F.fold is the inverse of F.unfold, so it restores ``dim`` channels again
        self.conv3 = conv_1x1_bn(dim, dim)
        self.conv4 = conv_nxn_bn(dim * 2, dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError(f"feature map {h}x{w} is not divisible by patch_size={p}")

        identity = x
        x = self.conv2(self.conv1(x))

        if p > 1:
            tokens = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)  # (B, L, C*p*p)
            gh, gw = h // p, w // p
        else:
            tokens = x.flatten(2).transpose(1, 2)
            gh, gw = h, w

        if self.sra is not None:
            # residual injection keeps the sub-sampled detail in the token stream
            tokens = tokens + self.sra(tokens, gh, gw)

        tokens = self.transformer(tokens)

        if p > 1:
            x = F.fold(tokens.transpose(1, 2), output_size=(h, w), kernel_size=p, stride=p)
        else:
            x = tokens.transpose(1, 2).reshape(b, -1, h, w)

        x = self.conv3(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.conv4(torch.cat([x, identity], dim=1))


class MobileViTv3DSR(nn.Module):
    def __init__(self, num_classes: int = 4, dims: Sequence[int] = (16, 32, 64),
                 depths: Sequence[int] = (2, 3, 3), num_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.0,
                 sr_ratios: Sequence[int] = (8, 4, 2), patch_size: int = 2,
                 use_sra: bool = True, use_da: bool = True):
        super().__init__()
        if isinstance(sr_ratios, int):
            sr_ratios = tuple([sr_ratios] * len(dims))

        self.use_sra = use_sra
        self.use_da = use_da
        self.stem = conv_nxn_bn(3, dims[0], 3, stride=2)

        stages = []
        for i, dim in enumerate(dims):
            layers = []
            if i > 0:
                layers.append(conv_nxn_bn(dims[i - 1], dim, 3, stride=2))
            layers.append(
                MobileViTBlock(dim, depths[i], num_heads, mlp_ratio, dropout,
                              sr_ratios[i], patch_size, use_sra, use_da)
            )
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.flatten(2).mean(-1)
        x = self.norm(x)
        return self.head(x)


def build_model(variant: str = "dsr", num_classes: int = 4, **kwargs) -> MobileViTv3DSR:
    """Build one of the four ablation variants."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}, got {variant!r}")
    use_sra, use_da = VARIANTS[variant]
    return MobileViTv3DSR(num_classes=num_classes, use_sra=use_sra, use_da=use_da, **kwargs)
