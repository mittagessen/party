# coding=utf-8
# Copyright 2024 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal prompt encoder"""
import math
import torch
import logging

from torch import nn
from typing import Optional

from party.modules.attention import MultiHeadAttention
from party.modules.norms import RMSNorm
from party.modules.tanh_gate import TanhGate
from party.modules.feed_forward import FeedForward
from party.modules.transformer import TransformerCrossAttentionLayer, TransformerSelfAttentionLayer


logger = logging.getLogger(__name__)

__all__ = ['PromptEncoder', 'PromptCrossAttention']


@torch.compiler.disable()
class PromptEncoder(nn.Module):
    """
    Encodes prompts for input to party's decoder.

    Args:
        embed_dim: The prompts' embedding dimension. Needs to be divisible
        by 8.
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # 4 curve points + 2 bbox corners, box center, box extents
        self.point_embeddings = nn.Embedding(8, embed_dim // 4)
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, embed_dim // 8)))

    def _positional_embed(self, coords):
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def _embed_curves(self, curves: torch.FloatTensor):
        point_embedding = self._positional_embed(curves)
        point_embedding += self.point_embeddings.weight[:4]
        return point_embedding.view(curves.shape[0], -1)

    def _embed_boxes(self, boxes: torch.FloatTensor):
        box_embedding = self._positional_embed(boxes)
        box_embedding += self.point_embeddings.weight[4:]
        return box_embedding.view(boxes.shape[0], -1)

    def forward(self,
                curves: Optional[torch.FloatTensor] = None,
                boxes: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """
        Embeds different types of prompts, either cubic Bezier curves or
        bounding boxes.

        Args:
          curves: Normalized point coordinates of shape (B_1, 4, 2)
          boxes: Normalized bounding box corner coordinates of shape (B_2, 4, 2)

        Returns:
          Embeddings for the points with shape (B_1+B_2, E)
        """
        embeddings = torch.empty((0, self.embed_dim),
                                 device=self.point_embeddings.weight.device)
        if curves is not None:
            curve_embeddings = self._embed_curves(curves)
            embeddings = torch.cat([embeddings, curve_embeddings])
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            embeddings = torch.cat([embeddings, box_embeddings])

        return embeddings


class PromptCrossAttention(nn.Module):
    """
    Cross-attention prompt encoder that produces line-focused features by
    conditioning learned queries with Fourier-embedded geometry and
    cross-attending into encoder features.

    Args:
        embed_dim: Embedding dimension matching the decoder.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention + self-attention blocks.
        num_samples: Number of learned query tokens.
        num_freqs: Number of log-spaced Fourier frequencies for geometry encoding.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 num_samples: int = 32,
                 num_freqs: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_samples = num_samples
        head_dim = embed_dim // num_heads
        hidden_dim = 4 * embed_dim

        # Log-spaced Fourier frequencies for geometry encoding
        freqs = 2 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs, persistent=False)
        fourier_dim = 8 * (1 + 2 * num_freqs)  # 8 coords × (raw + sin + cos per freq)
        self.geom_proj = nn.Linear(fourier_dim, embed_dim)

        # Learned base queries
        self.base_queries = nn.Parameter(torch.randn(num_samples, embed_dim) * 0.02)

        # Learned type embeddings: 0 = curve, 1 = box
        self.type_embeddings = nn.Embedding(2, embed_dim)

        # Cross-attention + self-attention blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            x_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                q_norm=RMSNorm(dim=head_dim, eps=1e-05),
                k_norm=RMSNorm(dim=head_dim, eps=1e-05),
                pos_embeddings=None,
                attn_dropout=0.0,
                is_causal=False,
            )
            x_attn_layer = TransformerCrossAttentionLayer(
                attn=x_attn,
                mlp=FeedForward(
                    gate_proj=nn.Linear(embed_dim, hidden_dim),
                    down_proj=nn.Linear(hidden_dim, embed_dim),
                    up_proj=None,
                ),
                ca_norm=RMSNorm(embed_dim, eps=1e-5),
                mlp_norm=RMSNorm(embed_dim, eps=1e-5),
                ca_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )

            self_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=None,
                attn_dropout=0.0,
                is_causal=False,
            )
            self_attn_layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=FeedForward(
                    gate_proj=nn.Linear(embed_dim, hidden_dim),
                    down_proj=nn.Linear(hidden_dim, embed_dim),
                    up_proj=None,
                ),
                sa_norm=RMSNorm(embed_dim, eps=1e-5),
                mlp_norm=RMSNorm(embed_dim, eps=1e-5),
                sa_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )

            self.layers.append(nn.ModuleDict({
                'x_attn': x_attn_layer,
                'self_attn': self_attn_layer,
            }))

    def forward(self,
                encoder_features: torch.Tensor,
                curves: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            encoder_features: Encoder output of shape ``[1, S_enc, D]`` or
                              ``[b, S_enc, D]``.
            curves: Cubic Bézier control points of shape ``[b, 4, 2]``, or None.
            boxes: Bounding box coordinates of shape ``[b, 4, 2]``, or None.

        Returns:
            Line-focused features of shape ``[b, num_samples, embed_dim]``.
        """
        if curves is not None and boxes is not None:
            raise ValueError('curves and boxes are mutually exclusive')
        if curves is None and boxes is None:
            raise ValueError('One of curves or boxes must be provided')

        if curves is not None:
            geom = curves.flatten(-2)
            type_idx = 0
        else:
            geom = boxes.flatten(-2)
            type_idx = 1

        b = geom.size(0)

        freqs = self.freqs.to(dtype=geom.dtype)
        scaled = geom.unsqueeze(-1) * (2.0 * math.pi * freqs)
        geom_feat = torch.cat([geom,
                               torch.sin(scaled).flatten(-2),
                               torch.cos(scaled).flatten(-2)], dim=-1)

        # Project geometry and condition learned queries
        cond = self.geom_proj(geom_feat)
        h = self.base_queries.unsqueeze(0) + cond.unsqueeze(1)
        h = h + self.type_embeddings.weight[type_idx]

        encoder_features = encoder_features.expand(b, -1, -1)

        # Cross-attend into encoder, then self-attend among prompt tokens
        for layer in self.layers:
            h = layer['x_attn'](h, encoder_input=encoder_features)
            h = layer['self_attn'](h)

        return h
