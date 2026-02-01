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

__all__ = ['PromptCrossAttention']


def sample_bezier(control_points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Samples points along cubic Bézier curves.

    Args:
        control_points: Control points of shape ``[b, 4, 2]`` (P0, P1, P2, P3)
                        with coordinates in [0, 1].
        num_samples: Number of points to sample along each curve.

    Returns:
        Sampled points of shape ``[b, num_samples, 2]``.
    """
    t = torch.linspace(0, 1, num_samples,
                       device=control_points.device,
                       dtype=control_points.dtype)
    t = t.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
    p0 = control_points[:, 0:1, :]  # [b, 1, 2]
    p1 = control_points[:, 1:2, :]
    p2 = control_points[:, 2:3, :]
    p3 = control_points[:, 3:4, :]
    return ((1 - t)**3 * p0 +
            3 * (1 - t)**2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3)


def sample_box(boxes: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Samples points along the diagonal of bounding boxes.

    Samples uniformly from the top-left corner (xmin, ymin) to the
    bottom-right corner (xmax, ymax). This handles horizontal, vertical,
    and tilted text lines as the bbox diagonal naturally follows the reading
    direction.

    Args:
        boxes: Box coordinates of shape ``[b, 4, 2]`` where the points are
               ``[[xmin, ymin], [xmax, ymax], [cx, cy], [w, h]]`` with
               coordinates in [0, 1].
        num_samples: Number of points to sample along the diagonal.

    Returns:
        Sampled points of shape ``[b, num_samples, 2]``.
    """
    tl = boxes[:, 0:1, :]  # [b, 1, 2] (xmin, ymin)
    br = boxes[:, 1:2, :]  # [b, 1, 2] (xmax, ymax)
    t = torch.linspace(0, 1, num_samples,
                       device=boxes.device,
                       dtype=boxes.dtype)
    t = t.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
    return tl + t * (br - tl)


class PromptCrossAttention(nn.Module):
    """
    Cross-attention prompt encoder that produces line-focused features by
    sampling points along curves/boxes and cross-attending into encoder
    features.

    Args:
        embed_dim: Embedding dimension matching the decoder.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention + self-attention blocks.
        num_samples: Number of points sampled along each curve/box.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 num_samples: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_samples = num_samples
        head_dim = embed_dim // num_heads
        hidden_dim = 4 * embed_dim

        # Gaussian Fourier feature matrix
        self.register_buffer("positional_encoding_gaussian_matrix",
                             torch.randn((2, embed_dim // 2)))

        # Learned type embeddings: 0 = curve point, 1 = box point
        self.type_embeddings = nn.Embedding(2, embed_dim)

        # Project Fourier features to embed_dim
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Cross-attention + self-attention blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            cross_attn = MultiHeadAttention(
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
            cross_attn_layer = TransformerCrossAttentionLayer(
                attn=cross_attn,
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
                'cross_attn': cross_attn_layer,
                'self_attn': self_attn_layer,
            }))

    def _positional_embed(self, coords: torch.Tensor) -> torch.Tensor:
        """Gaussian Fourier features. Input [b, N, 2] -> output [b, N, embed_dim]"""
        coords = 2 * coords - 1
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

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
            points = sample_bezier(curves, self.num_samples)
            type_idx = 0
        else:
            points = sample_box(boxes, self.num_samples)
            type_idx = 1

        b = points.size(0)

        # Encode points with Fourier features + type embedding + projection
        h = self._positional_embed(points)
        h = self.input_proj(h)
        h = h + self.type_embeddings.weight[type_idx]

        encoder_features = encoder_features.expand(b, -1, -1)

        # Cross-attend into encoder, then self-attend among prompt tokens
        for layer in self.layers:
            h = layer['cross_attn'](h, encoder_input=encoder_features)
            h = layer['self_attn'](h)

        return h
