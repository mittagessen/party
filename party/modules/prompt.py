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
"""Prompt-conditioned cross-attention modules."""
import math
import logging
import torch

from torch import nn
from typing import Optional

from party.modules.attention import MultiHeadAttention
from party.modules.norms import RMSNorm
from party.modules.tanh_gate import TanhGate
from party.modules.feed_forward import FeedForward
from party.modules.transformer import TransformerCrossAttentionLayer, TransformerSelfAttentionLayer


logger = logging.getLogger(__name__)

__all__ = ['PromptCrossAttention']


class PromptCrossAttention(nn.Module):
    """
    Learned-query prompt encoder producing compact line-focused features by
    conditioning query slots with prompt geometry and ordered prompt samples
    before cross-attending into encoder features.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 num_samples: int = 32,
                 num_freqs: int = 8,
                 gate_init: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_samples = num_samples
        self.gate_init = float(gate_init)
        head_dim = embed_dim // num_heads
        hidden_dim = 4 * embed_dim

        freqs = 2 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs, persistent=False)
        self.register_buffer('slot_progress',
                             torch.linspace(0.0, 1.0, num_samples, dtype=torch.float32),
                             persistent=False)
        global_fourier_dim = 8 * (1 + 2 * num_freqs)
        slot_fourier_dim = 5 * (1 + 2 * num_freqs)
        self.geom_proj = nn.Linear(global_fourier_dim, embed_dim)
        self.slot_geom_proj = nn.Linear(slot_fourier_dim, embed_dim)

        self.base_queries = nn.Parameter(torch.randn(num_samples, embed_dim) * 0.02)
        self.type_embeddings = nn.Embedding(2, embed_dim)

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
            with torch.no_grad():
                x_attn_layer.ca_scale.scale.fill_(self.gate_init)
                x_attn_layer.mlp_scale.scale.fill_(self.gate_init)

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
            with torch.no_grad():
                self_attn_layer.sa_scale.scale.fill_(self.gate_init)
                self_attn_layer.mlp_scale.scale.fill_(self.gate_init)

            self.layers.append(nn.ModuleDict({
                'x_attn': x_attn_layer,
                'self_attn': self_attn_layer,
            }))

    def _fourier_features(self, values: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=values.device, dtype=values.dtype)
        scaled = values.unsqueeze(-1) * (2.0 * math.pi * freqs)
        return torch.cat([values,
                          torch.sin(scaled).flatten(-2),
                          torch.cos(scaled).flatten(-2)], dim=-1)

    def _sample_curve_slots(self, curves: torch.Tensor) -> torch.Tensor:
        progress = self.slot_progress.to(device=curves.device, dtype=curves.dtype)
        progress = progress.view(1, self.num_samples, 1)

        p0, p1, p2, p3 = curves.unbind(dim=1)
        one_minus_t = 1.0 - progress

        points = (
            (one_minus_t ** 3) * p0[:, None, :]
            + 3.0 * (one_minus_t ** 2) * progress * p1[:, None, :]
            + 3.0 * one_minus_t * (progress ** 2) * p2[:, None, :]
            + (progress ** 3) * p3[:, None, :]
        )
        tangents = (
            3.0 * (one_minus_t ** 2) * (p1 - p0)[:, None, :]
            + 6.0 * one_minus_t * progress * (p2 - p1)[:, None, :]
            + 3.0 * (progress ** 2) * (p3 - p2)[:, None, :]
        )
        tangents = tangents / tangents.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat([points,
                          tangents,
                          progress.expand(curves.size(0), -1, -1)], dim=-1)

    def _sample_box_slots(self, boxes: torch.Tensor) -> torch.Tensor:
        progress = self.slot_progress.to(device=boxes.device, dtype=boxes.dtype)
        progress = progress.view(1, self.num_samples, 1)

        xy_min, xy_max, center, _ = boxes.unbind(dim=1)
        start = torch.stack([xy_min[:, 0], center[:, 1]], dim=-1)
        end = torch.stack([xy_max[:, 0], center[:, 1]], dim=-1)
        direction = end - start
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        points = start[:, None, :] + progress * (end - start)[:, None, :]
        return torch.cat([points,
                          direction[:, None, :].expand(-1, self.num_samples, -1),
                          progress.expand(boxes.size(0), -1, -1)], dim=-1)

    def forward(self,
                encoder_features: torch.Tensor,
                curves: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if curves is not None and boxes is not None:
            raise ValueError('curves and boxes are mutually exclusive')
        if curves is None and boxes is None:
            raise ValueError('One of curves or boxes must be provided')

        if curves is not None:
            geom = curves.flatten(-2)
            slot_geom = self._sample_curve_slots(curves)
            type_idx = 0
        else:
            geom = boxes.flatten(-2)
            slot_geom = self._sample_box_slots(boxes)
            type_idx = 1

        batch_size = geom.size(0)

        geom_feat = self._fourier_features(geom)
        slot_geom_feat = self._fourier_features(slot_geom)

        cond = self.geom_proj(geom_feat)
        # Each prompt token is anchored to a fixed progress value along the
        # prompt geometry so the CTC auxiliary sees an explicitly ordered
        # sequence instead of a bank of exchangeable learned slots.
        h = self.base_queries.unsqueeze(0) + self.slot_geom_proj(slot_geom_feat)
        h = h + cond.unsqueeze(1)
        h = h + self.type_embeddings.weight[type_idx]

        if encoder_features.size(0) == 1:
            encoder_features = encoder_features.expand(batch_size, -1, -1)
        elif encoder_features.size(0) != batch_size:
            raise ValueError(
                f'encoder_features batch size ({encoder_features.size(0)}) '
                f'does not match prompt batch size ({batch_size})'
            )

        for layer in self.layers:
            h = layer['x_attn'](h, encoder_input=encoder_features)
            h = layer['self_attn'](h)

        return h
