# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#           2024 Benjamin Kiessling
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
"""Llama vision fusion model"""
import json
import torch
import logging
import torch.nn.functional as F

from torch import nn
from typing import Optional

from party.tokenizer import TOKEN_NUM
from party.modules import (MultiHeadAttention, RMSNorm, TanhGate,
                           TransformerCrossAttentionLayer, TransformerDecoder,
                           FeedForward, TransformerSelfAttentionLayer,
                           FusionLayer, scale_hidden_dim_for_mlp,
                           Llama3ScaledRoPE, llama3_mlp,
                           PositionEmbeddingRandom)


logger = logging.getLogger(__name__)

__all__ = ['bytellama_vision_decoder', 'LinePromptedMultiScaleResampler']


def bytellama_vision_decoder(vocab_size: int = TOKEN_NUM,
                             num_layers: int = 30,
                             num_heads: int = 9,
                             num_kv_heads: int = 3,
                             embed_dim: int = 576,
                             max_seq_len: int = 384,
                             intermediate_dim: int = 1536,
                             attn_dropout: float = 0.0,
                             norm_eps: int = 1e-5,
                             rope_base: int = 10000,
                             encoder_max_seq_len: int = 56700,
                             fusion_interval: int = 3,
                             pretrained: Optional[str] = None,
                             **kwargs) -> TransformerDecoder:
    """
    Builds a vision decoder from a ByteLlama model with additional fused cross
    attention layers. This includes:
    - Token embeddings
    - num_layers number of CausalSelfAttention blocks
    - Fused cross attention layers every fusion_interval number of layers
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value.
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~party.modules.KVCache`.
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~party.modules.scale_hidden_dim_for_mlp`.
        fusion_interval (int): interval number of layers between fusion layers.
        pretrained (str): huggingface hub identifier of pretrained bytellama
                          weights. All hyperparameters will except
                          encoder_max_seq_len will be ignored.

    Returns:
        TransformerDecoder: Instantiation of Llama 3.2 vision decoder.
    """
    config = {'vocab_size': vocab_size,
              'num_layers': num_layers,
              'num_heads': num_heads,
              'num_kv_heads': num_kv_heads,
              'embed_dim': embed_dim,
              'max_seq_len': max_seq_len,
              'intermediate_dim': intermediate_dim,
              'attn_dropout': attn_dropout,
              'norm_eps': norm_eps,
              'rope_base': rope_base,
              'encoder_max_seq_len': encoder_max_seq_len,
              'fusion_interval': fusion_interval}

    if pretrained:
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo_id=pretrained, filename='config.json'), 'r') as fp:
            config.update(json.load(fp))
            config['vocab_size'] = TOKEN_NUM

    head_dim = config['embed_dim'] // config['num_heads']
    num_kv_heads = config['num_kv_heads'] if config['num_kv_heads'] else config['num_heads']
    hidden_dim = config['intermediate_dim'] or scale_hidden_dim_for_mlp(config['embed_dim'])
    layers = []

    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=config['max_seq_len'], base=config['rope_base'])

    for idx in range(1, config['num_layers'] + 1):

        # Self attention layers for text decoder (GQA)
        self_attn = MultiHeadAttention(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(config['embed_dim'], config['num_heads'] * head_dim, bias=False),
            k_proj=nn.Linear(config['embed_dim'], num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(config['embed_dim'], num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(config['embed_dim'], config['embed_dim'], bias=False),
            pos_embeddings=rope,
            max_seq_len=config['max_seq_len'],
            attn_dropout=0.0,
        )
        mlp = llama3_mlp(dim=config['embed_dim'], hidden_dim=hidden_dim)
        decoder_layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=config['embed_dim'], eps=1e-5),
            mlp_norm=RMSNorm(dim=config['embed_dim'], eps=1e-5),
        )

        # cross attention layers, mixing text and vision,
        # placed every `fusion_interval` layers
        if idx % config['fusion_interval'] == 0:
            xattn = MultiHeadAttention(
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(config['embed_dim'], config['num_heads'] * head_dim, bias=False),
                k_proj=nn.Linear(config['embed_dim'], num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(config['embed_dim'], num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(config['embed_dim'], config['embed_dim'], bias=False),
                q_norm=RMSNorm(dim=head_dim, eps=1e-05),
                k_norm=RMSNorm(dim=head_dim, eps=1e-05),
                pos_embeddings=rope,
                max_seq_len=config['encoder_max_seq_len'],
                is_causal=False,
                attn_dropout=0.0,
            )

            xattn_mlp = llama3_mlp(dim=config['embed_dim'], hidden_dim=hidden_dim)
            xattn_layer = TransformerCrossAttentionLayer(
                attn=xattn,
                mlp=xattn_mlp,
                ca_norm=RMSNorm(dim=config['embed_dim']),
                mlp_norm=RMSNorm(dim=config['embed_dim']),
                ca_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )
            fusion_layer = FusionLayer(layer=decoder_layer, fusion_layer=xattn_layer)
            layers.append(fusion_layer)
        else:
            layers.append(decoder_layer)

    tok_embeddings = nn.Embedding(config['vocab_size'], config['embed_dim'])
    output_proj = nn.Linear(config['embed_dim'], config['vocab_size'], bias=False)

    decoder = TransformerDecoder(tok_embeddings=tok_embeddings,
                                 layers=layers,
                                 max_seq_len=config['max_seq_len'],
                                 num_heads=config['num_heads'],
                                 head_dim=head_dim,
                                 norm=RMSNorm(config['embed_dim'], eps=1e-05),
                                 output=output_proj)

    if pretrained:
        weight_path = hf_hub_download(repo_id=pretrained, filename='model.safetensors')
        from safetensors import safe_open
        with safe_open(weight_path, framework='pt') as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        decoder.load_state_dict(state_dict, strict=False)

    return decoder


class LinePromptedMultiScaleResampler(nn.Module):
    """
    Prompt-conditioned visual resampler that builds an ordered line memory for
    the decoder by soft-pooling encoder tokens around prompt-derived anchors.

    The memory layout is:
    1. `line_num_tokens` ordered tokens sampled along the prompt geometry
    2. `global_num_tokens` optional page-summary tokens
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 encoder_embed_dims: list[int],
                 encoder_sizes: list[tuple[int, int]],
                 decoder_embed_dim: int,
                 line_num_tokens: int,
                 global_num_tokens: int = 0,
                 ds_factors: list[int] = None,
                 refine_layers: int = 1,
                 refine_num_heads: Optional[int] = None,
                 sigma_u_factor: float = 1.5,
                 sigma_v_factor: float = 0.5,
                 use_pos_embeddings: bool = True):
        super().__init__()
        if line_num_tokens < 1:
            raise ValueError('line_num_tokens must be >= 1.')
        if global_num_tokens < 0:
            raise ValueError('global_num_tokens must be >= 0.')
        if sigma_u_factor <= 0.0 or sigma_v_factor <= 0.0:
            raise ValueError('sigma_u_factor and sigma_v_factor must be > 0.')
        if ds_factors is None:
            ds_factors = [4, 2, 1]
        if len(encoder_embed_dims) != len(encoder_sizes) or len(encoder_embed_dims) != len(ds_factors):
            raise ValueError('encoder_embed_dims, encoder_sizes, and ds_factors must have the same length.')

        self.line_num_tokens = int(line_num_tokens)
        self.global_num_tokens = int(global_num_tokens)
        self.num_samples = self.line_num_tokens + self.global_num_tokens
        self.num_scales = len(encoder_embed_dims)
        self.sigma_u_factor = float(sigma_u_factor)
        self.sigma_v_factor = float(sigma_v_factor)
        self.score_scale = decoder_embed_dim ** -0.5

        if refine_num_heads is None:
            refine_num_heads = num_heads

        mlp_ratio = 4
        self.adapter = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.pos_embeddings = nn.ModuleList()
        self.scale_embeddings = nn.Parameter(torch.randn(self.num_scales, decoder_embed_dim) * 0.02)

        for idx, (encoder_embed_dim, size, ds_factor) in enumerate(zip(encoder_embed_dims, encoder_sizes, ds_factors)):
            hidden_dim = int(mlp_ratio * encoder_embed_dim)
            head_dim = encoder_embed_dim // num_heads

            if ds_factor > 1:
                self.downsample.append(nn.Sequential(
                    nn.Conv2d(encoder_embed_dim, encoder_embed_dim,
                              kernel_size=ds_factor, stride=ds_factor,
                              groups=encoder_embed_dim, bias=False),
                    nn.Conv2d(encoder_embed_dim, encoder_embed_dim,
                              kernel_size=1, bias=False),
                ))
            else:
                self.downsample.append(nn.Identity())

            ds_size = (size[0] // ds_factor, size[1] // ds_factor)
            self.register_buffer(f'token_coords_{idx}', self._build_token_coords(ds_size), persistent=False)

            layers = []
            for _ in range(num_layers):
                self_attn = MultiHeadAttention(embed_dim=encoder_embed_dim,
                                               num_heads=num_heads,
                                               num_kv_heads=num_heads,
                                               head_dim=head_dim,
                                               q_proj=nn.Linear(encoder_embed_dim, num_heads * head_dim, bias=False),
                                               k_proj=nn.Linear(encoder_embed_dim, num_heads * head_dim, bias=False),
                                               v_proj=nn.Linear(encoder_embed_dim, num_heads * head_dim, bias=False),
                                               output_proj=nn.Linear(encoder_embed_dim, encoder_embed_dim, bias=False),
                                               pos_embeddings=None,
                                               attn_dropout=0.0,
                                               is_causal=False)

                mlp = FeedForward(gate_proj=nn.Linear(encoder_embed_dim, hidden_dim),
                                  down_proj=nn.Linear(hidden_dim, encoder_embed_dim),
                                  up_proj=None)

                layer = TransformerSelfAttentionLayer(attn=self_attn,
                                                      mlp=mlp,
                                                      sa_norm=RMSNorm(encoder_embed_dim, eps=1e-5),
                                                      mlp_norm=RMSNorm(encoder_embed_dim, eps=1e-5),
                                                      sa_scale=TanhGate(),
                                                      mlp_scale=TanhGate())
                layers.append(layer)
            layers.append(nn.Linear(encoder_embed_dim, decoder_embed_dim))
            self.adapter.append(nn.Sequential(*layers))
            if use_pos_embeddings:
                self.pos_embeddings.append(PositionEmbeddingRandom(decoder_embed_dim, ds_size))
            else:
                self.pos_embeddings.append(nn.Identity())

        self.type_embeddings = nn.Embedding(2, decoder_embed_dim)
        self.prompt_mlp = nn.Sequential(
            nn.Linear(6, decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
        )
        self.anchor_mlp = nn.Sequential(
            nn.Linear(9, decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
        )
        self.prompt_norm = RMSNorm(decoder_embed_dim, eps=1e-5)
        self.anchor_norm = RMSNorm(decoder_embed_dim, eps=1e-5)

        self.line_pos_embeddings = nn.Parameter(torch.randn(self.line_num_tokens, decoder_embed_dim) * 0.02)
        self.register_buffer('line_progress',
                             torch.linspace(0.0, 1.0, self.line_num_tokens, dtype=torch.float32),
                             persistent=False)

        self.line_query_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
        self.line_key_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
        self.line_value_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
        self.line_out_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)

        if self.global_num_tokens:
            self.global_queries = nn.Parameter(torch.randn(self.global_num_tokens, decoder_embed_dim) * 0.02)
            self.global_pos_embeddings = nn.Parameter(torch.randn(self.global_num_tokens, decoder_embed_dim) * 0.02)
            self.global_query_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
            self.global_key_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
            self.global_value_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
            self.global_out_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
        else:
            self.global_queries = None
            self.global_pos_embeddings = None
            self.global_query_proj = None
            self.global_key_proj = None
            self.global_value_proj = None
            self.global_out_proj = None

        self.refine_layers = nn.ModuleList()
        for _ in range(refine_layers):
            head_dim = decoder_embed_dim // refine_num_heads
            self_attn = MultiHeadAttention(
                embed_dim=decoder_embed_dim,
                num_heads=refine_num_heads,
                num_kv_heads=refine_num_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(decoder_embed_dim, refine_num_heads * head_dim, bias=False),
                k_proj=nn.Linear(decoder_embed_dim, refine_num_heads * head_dim, bias=False),
                v_proj=nn.Linear(decoder_embed_dim, refine_num_heads * head_dim, bias=False),
                output_proj=nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False),
                pos_embeddings=None,
                attn_dropout=0.0,
                is_causal=False,
            )
            self.refine_layers.append(
                TransformerSelfAttentionLayer(
                    attn=self_attn,
                    mlp=FeedForward(
                        gate_proj=nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim),
                        down_proj=nn.Linear(4 * decoder_embed_dim, decoder_embed_dim),
                        up_proj=None,
                    ),
                    sa_norm=RMSNorm(decoder_embed_dim, eps=1e-5),
                    mlp_norm=RMSNorm(decoder_embed_dim, eps=1e-5),
                )
            )

        self.output_norm = RMSNorm(decoder_embed_dim, eps=1e-5)

    @staticmethod
    def _build_token_coords(size: tuple[int, int]) -> torch.Tensor:
        h, w = size
        grid = torch.ones((h, w), dtype=torch.float32)
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        return torch.stack([x_embed, y_embed], dim=-1).flatten(0, 1)

    def _sample_curve_anchors(self, curves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.line_progress.to(device=curves.device, dtype=curves.dtype)
        one_minus_t = 1.0 - t

        p0, p1, p2, p3 = curves.unbind(dim=1)
        points = (
            (one_minus_t ** 3).view(1, -1, 1) * p0.unsqueeze(1) +
            (3.0 * one_minus_t ** 2 * t).view(1, -1, 1) * p1.unsqueeze(1) +
            (3.0 * one_minus_t * t ** 2).view(1, -1, 1) * p2.unsqueeze(1) +
            (t ** 3).view(1, -1, 1) * p3.unsqueeze(1)
        )

        deriv = (
            (3.0 * one_minus_t ** 2).view(1, -1, 1) * (p1 - p0).unsqueeze(1) +
            (6.0 * one_minus_t * t).view(1, -1, 1) * (p2 - p1).unsqueeze(1) +
            (3.0 * t ** 2).view(1, -1, 1) * (p3 - p2).unsqueeze(1)
        )
        tangents = F.normalize(deriv, dim=-1, eps=1e-6)

        diffs = points[:, 1:, :] - points[:, :-1, :]
        line_length = diffs.norm(dim=-1).sum(dim=1, keepdim=True).clamp_min(1e-3)
        bbox_height = (curves[..., 1].amax(dim=1, keepdim=True) -
                       curves[..., 1].amin(dim=1, keepdim=True)).clamp_min(1.0 / 256.0)
        return points, tangents, line_length, bbox_height

    def _sample_box_anchors(self, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xmin = boxes[:, 0, 0:1]
        xmax = boxes[:, 1, 0:1]
        cy = boxes[:, 2, 1:2]
        width = boxes[:, 3, 0:1].abs().clamp_min(1e-3)
        height = boxes[:, 3, 1:2].abs().clamp_min(1.0 / 256.0)

        progress = self.line_progress.to(device=boxes.device, dtype=boxes.dtype).view(1, -1, 1)
        xs = xmin.unsqueeze(1) + progress * (xmax - xmin).unsqueeze(1)
        ys = cy.unsqueeze(1).expand(-1, self.line_num_tokens, -1)
        points = torch.cat([xs, ys], dim=-1)

        tangents = torch.zeros_like(points)
        tangents[..., 0] = 1.0
        return points, tangents, width, height

    def _encode_prompt(self,
                       curves: Optional[torch.Tensor],
                       boxes: Optional[torch.Tensor],
                       dtype: torch.dtype,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if curves is not None and boxes is not None:
            raise ValueError('curves and boxes are mutually exclusive.')
        if curves is None and boxes is None:
            raise ValueError('One of curves or boxes must be provided.')

        if curves is not None:
            geom = curves.to(device=device, dtype=dtype)
            if geom.ndim != 3 or geom.shape[1:] != (4, 2):
                raise ValueError(f'Expected curve prompts with shape [b, 4, 2], got {tuple(geom.shape)}.')
            points, tangents, line_length, line_height = self._sample_curve_anchors(geom)
            type_idx = 0
        else:
            geom = boxes.to(device=device, dtype=dtype)
            if geom.ndim != 3 or geom.shape[1:] != (4, 2):
                raise ValueError(f'Expected box prompts with shape [b, 4, 2], got {tuple(geom.shape)}.')
            points, tangents, line_length, line_height = self._sample_box_anchors(geom)
            type_idx = 1

        normals = torch.stack([-tangents[..., 1], tangents[..., 0]], dim=-1)
        line_dir = F.normalize(points[:, -1, :] - points[:, 0, :], dim=-1, eps=1e-6)
        line_center = points.mean(dim=1)
        prompt_feat = torch.cat([line_center, line_dir, line_length, line_height], dim=-1)
        prompt_vec = self.prompt_mlp(prompt_feat)
        prompt_vec = prompt_vec + self.type_embeddings.weight[type_idx].to(device=device, dtype=dtype).unsqueeze(0)
        prompt_vec = self.prompt_norm(prompt_vec)

        sigma_u = (line_length / max(self.line_num_tokens - 1, 1)).clamp_min(1.0 / 512.0) * self.sigma_u_factor
        sigma_v = line_height.clamp_min(1.0 / 256.0) * self.sigma_v_factor
        return points, tangents, normals, prompt_vec, torch.cat([sigma_u, sigma_v], dim=-1)

    def _refine_memory(self, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.refine_layers:
            memory = layer(memory)
        return self.output_norm(memory)

    def forward(self,
                encoder_hidden_states: list[torch.Tensor],
                curves: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, (list, tuple)):
            raise ValueError('encoder_hidden_states must be a list/tuple of feature maps.')
        if len(encoder_hidden_states) != self.num_scales:
            raise ValueError(f'Expected {self.num_scales} feature maps, got {len(encoder_hidden_states)}.')

        ref = encoder_hidden_states[0]
        points, tangents, normals, prompt_vec, sigmas = self._encode_prompt(curves=curves,
                                                                             boxes=boxes,
                                                                             dtype=ref.dtype,
                                                                             device=ref.device)
        batch_size = prompt_vec.size(0)
        sigma_u = sigmas[:, 0:1].unsqueeze(-1)
        sigma_v = sigmas[:, 1:2].unsqueeze(-1)

        scale_tokens = []
        scale_coords = []
        for idx, hidden_state in enumerate(encoder_hidden_states):
            h = self.downsample[idx](hidden_state)
            h = h.flatten(-2).transpose(-1, -2)
            h = self.adapter[idx](h)
            h = self.pos_embeddings[idx](h)

            if h.size(0) == 1 and batch_size > 1:
                h = h.expand(batch_size, -1, -1)
            elif h.size(0) != batch_size:
                raise ValueError(
                    f'Batch mismatch between encoder features ({h.size(0)}) and prompt geometry ({batch_size}).'
                )

            h = h + self.scale_embeddings[idx].view(1, 1, -1).to(dtype=h.dtype, device=h.device)
            coords = getattr(self, f'token_coords_{idx}').to(device=h.device, dtype=h.dtype)
            scale_tokens.append(h)
            scale_coords.append(coords.unsqueeze(0).expand(batch_size, -1, -1))

        tokens = torch.cat(scale_tokens, dim=1)
        coords = torch.cat(scale_coords, dim=1)

        progress = self.line_progress.to(device=tokens.device, dtype=tokens.dtype).view(1, -1, 1)
        anchor_feat = torch.cat([points,
                                 tangents,
                                 normals,
                                 progress.expand(batch_size, -1, -1),
                                 sigmas[:, 0:1].unsqueeze(1).expand(-1, self.line_num_tokens, -1),
                                 sigmas[:, 1:2].unsqueeze(1).expand(-1, self.line_num_tokens, -1)],
                                dim=-1)
        anchor_tokens = self.anchor_mlp(anchor_feat)
        anchor_tokens = anchor_tokens + prompt_vec.unsqueeze(1) + self.line_pos_embeddings.unsqueeze(0).to(dtype=tokens.dtype,
                                                                                                           device=tokens.device)
        anchor_tokens = self.anchor_norm(anchor_tokens)

        line_queries = self.line_query_proj(anchor_tokens)
        token_keys = self.line_key_proj(tokens)
        token_values = self.line_value_proj(tokens)

        delta = coords.unsqueeze(1) - points.unsqueeze(2)
        u = (delta * tangents.unsqueeze(2)).sum(dim=-1)
        v = (delta * normals.unsqueeze(2)).sum(dim=-1)
        geom_logits = -0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2)
        content_logits = torch.einsum('bld,bsd->bls', line_queries, token_keys) * self.score_scale
        line_weights = torch.softmax(geom_logits + content_logits, dim=-1)
        line_memory = torch.einsum('bls,bsd->bld', line_weights, token_values)
        line_memory = self.line_out_proj(line_memory) + anchor_tokens

        if self.global_num_tokens:
            global_queries = self.global_queries.unsqueeze(0).to(dtype=tokens.dtype, device=tokens.device)
            global_queries = global_queries + self.global_pos_embeddings.unsqueeze(0).to(dtype=tokens.dtype,
                                                                                        device=tokens.device)
            global_queries = global_queries + prompt_vec.unsqueeze(1)

            global_q = self.global_query_proj(global_queries)
            global_k = self.global_key_proj(tokens)
            global_v = self.global_value_proj(tokens)
            global_weights = torch.softmax(torch.einsum('bgd,bsd->bgs', global_q, global_k) * self.score_scale, dim=-1)
            global_memory = torch.einsum('bgs,bsd->bgd', global_weights, global_v)
            global_memory = self.global_out_proj(global_memory) + global_queries
            memory = torch.cat([line_memory, global_memory], dim=1)
        else:
            memory = line_memory

        return self._refine_memory(memory)
