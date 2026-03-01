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

__all__ = ['bytellama_vision_decoder', 'LocatorReaderConditioner']


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


class _SupportBiasedAttention(nn.Module):
    """
    Multi-headed cross-attention with an additive support bias over the visual
    token bank.
    """

    def __init__(self, query_dim: int, memory_dim: int, embed_dim: int,
                 num_heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(query_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, embed_dim, bias=False)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self,
                query: torch.Tensor,
                memory: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Q, _ = query.shape
        _, K, _ = memory.shape
        H = self.num_heads

        q = self.q_proj(query).view(B, Q, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, K, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, K, H, self.head_dim).transpose(1, 2)

        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        if bias is not None:
            scores = scores + bias[:, None, None, :]

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.einsum('bhqk,bhkd->bhqd', weights, v)
        out = out.transpose(1, 2).contiguous().view(B, Q, -1)
        return self.output_proj(out)


class LocatorReaderConditioner(nn.Module):
    """
    Prompt-conditioned latent bottleneck for page features.

    1. Locator tokens attend globally over the page token bank.
    2. The locator summary predicts a soft support prior over the page.
    3. Ordered reader tokens reread the full page with that support bias.
    4. Optional global tokens capture page context for the decoder.
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 encoder_embed_dims: list[int],
                 encoder_sizes: list[tuple[int, int]],
                 decoder_embed_dim: int,
                 reader_num_tokens: int,
                 locator_num_tokens: int = 8,
                 global_num_tokens: int = 0,
                 ds_factors: list[int] = None,
                 num_rounds: int = 2,
                 refine_layers: int = 1,
                 refine_num_heads: Optional[int] = None,
                 attn_dropout: float = 0.0,
                 use_pos_embeddings: bool = True):
        super().__init__()
        if reader_num_tokens < 1:
            raise ValueError('reader_num_tokens must be >= 1.')
        if locator_num_tokens < 1:
            raise ValueError('locator_num_tokens must be >= 1.')
        if global_num_tokens < 0:
            raise ValueError('global_num_tokens must be >= 0.')
        if num_rounds < 1:
            raise ValueError('num_rounds must be >= 1.')
        if ds_factors is None:
            ds_factors = [4, 2, 1]
        if len(encoder_embed_dims) != len(encoder_sizes) or len(encoder_embed_dims) != len(ds_factors):
            raise ValueError('encoder_embed_dims, encoder_sizes, and ds_factors must have the same length.')

        self.reader_num_tokens = int(reader_num_tokens)
        self.locator_num_tokens = int(locator_num_tokens)
        self.global_num_tokens = int(global_num_tokens)
        self.num_rounds = int(num_rounds)
        self.num_samples = self.reader_num_tokens + self.global_num_tokens
        self.num_scales = len(encoder_embed_dims)
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

                layers.append(
                    TransformerSelfAttentionLayer(attn=self_attn,
                                                  mlp=mlp,
                                                  sa_norm=RMSNorm(encoder_embed_dim, eps=1e-5),
                                                  mlp_norm=RMSNorm(encoder_embed_dim, eps=1e-5),
                                                  sa_scale=TanhGate(),
                                                  mlp_scale=TanhGate())
                )
            layers.append(nn.Linear(encoder_embed_dim, decoder_embed_dim))
            self.adapter.append(nn.Sequential(*layers))
            if use_pos_embeddings:
                self.pos_embeddings.append(PositionEmbeddingRandom(decoder_embed_dim, ds_size))
            else:
                self.pos_embeddings.append(nn.Identity())

        self.type_embeddings = nn.Embedding(2, decoder_embed_dim)
        self.prompt_mlp = nn.Sequential(
            nn.Linear(20, decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
        )
        self.prompt_norm = RMSNorm(decoder_embed_dim, eps=1e-5)

        self.locator_base = nn.Parameter(torch.randn(self.locator_num_tokens, decoder_embed_dim) * 0.02)
        self.reader_base = nn.Parameter(torch.randn(self.reader_num_tokens, decoder_embed_dim) * 0.02)
        self.reader_pos_embeddings = nn.Parameter(torch.randn(self.reader_num_tokens, decoder_embed_dim) * 0.02)
        self.register_buffer('reader_progress',
                             torch.linspace(-1.0, 1.0, self.reader_num_tokens, dtype=torch.float32),
                             persistent=False)

        self.reader_geom_mlp = nn.Sequential(
            nn.Linear(8, decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
        )
        self.reader_input_norm = RMSNorm(decoder_embed_dim, eps=1e-5)

        self.locator_cross_layers = nn.ModuleList()
        self.locator_self_layers = nn.ModuleList()
        self.reader_cross_layers = nn.ModuleList()
        self.reader_self_layers = nn.ModuleList()
        self.reader_feedback_proj = nn.ModuleList()
        self.locator_norms = nn.ModuleList()
        self.reader_norms = nn.ModuleList()

        for _ in range(self.num_rounds):
            self.locator_cross_layers.append(
                _SupportBiasedAttention(decoder_embed_dim, decoder_embed_dim, decoder_embed_dim,
                                        num_heads=refine_num_heads, attn_dropout=attn_dropout)
            )
            self.reader_cross_layers.append(
                _SupportBiasedAttention(decoder_embed_dim, decoder_embed_dim, decoder_embed_dim,
                                        num_heads=refine_num_heads, attn_dropout=attn_dropout)
            )
            self.locator_norms.append(RMSNorm(decoder_embed_dim, eps=1e-5))
            self.reader_norms.append(RMSNorm(decoder_embed_dim, eps=1e-5))

            self.locator_self_layers.append(self._build_self_attention_layer(decoder_embed_dim, refine_num_heads, attn_dropout))
            self.reader_self_layers.append(self._build_self_attention_layer(decoder_embed_dim, refine_num_heads, attn_dropout))

        self.reader_round_norms = nn.ModuleList()
        for _ in range(self.num_rounds - 1):
            self.reader_feedback_proj.append(nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False))
            self.reader_round_norms.append(RMSNorm(decoder_embed_dim, eps=1e-5))

        self.support_query_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
        self.support_token_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
        self.direction_head = nn.Linear(decoder_embed_dim, 2)
        self.temperature_head = nn.Linear(decoder_embed_dim, 1)

        if self.global_num_tokens:
            self.global_base = nn.Parameter(torch.randn(self.global_num_tokens, decoder_embed_dim) * 0.02)
            self.global_pos_embeddings = nn.Parameter(torch.randn(self.global_num_tokens, decoder_embed_dim) * 0.02)
            self.global_cross = _SupportBiasedAttention(decoder_embed_dim, decoder_embed_dim, decoder_embed_dim,
                                                        num_heads=refine_num_heads, attn_dropout=attn_dropout)
            self.global_norm = RMSNorm(decoder_embed_dim, eps=1e-5)
        else:
            self.global_base = None
            self.global_pos_embeddings = None
            self.global_cross = None
            self.global_norm = None

        self.refine_layers = nn.ModuleList()
        for _ in range(refine_layers):
            self.refine_layers.append(self._build_self_attention_layer(decoder_embed_dim, refine_num_heads, attn_dropout))
        self.output_norm = RMSNorm(decoder_embed_dim, eps=1e-5)

    @staticmethod
    def _build_token_coords(size: tuple[int, int]) -> torch.Tensor:
        h, w = size
        grid = torch.ones((h, w), dtype=torch.float32)
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        return torch.stack([x_embed, y_embed], dim=-1).flatten(0, 1)

    @staticmethod
    def _build_self_attention_layer(embed_dim: int, num_heads: int, attn_dropout: float = 0.0) -> TransformerSelfAttentionLayer:
        head_dim = embed_dim // num_heads
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
            attn_dropout=attn_dropout,
            is_causal=False,
        )
        return TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=FeedForward(
                gate_proj=nn.Linear(embed_dim, 4 * embed_dim),
                down_proj=nn.Linear(4 * embed_dim, embed_dim),
                up_proj=None,
            ),
            sa_norm=RMSNorm(embed_dim, eps=1e-5),
            mlp_norm=RMSNorm(embed_dim, eps=1e-5),
        )

    def _encode_prompt(self,
                       curves: Optional[torch.Tensor],
                       boxes: Optional[torch.Tensor],
                       dtype: torch.dtype,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if curves is not None and boxes is not None:
            raise ValueError('curves and boxes are mutually exclusive.')
        if curves is None and boxes is None:
            raise ValueError('One of curves or boxes must be provided.')

        if curves is not None:
            geom = curves.to(device=device, dtype=dtype)
            if geom.ndim != 3 or geom.shape[1:] != (4, 2):
                raise ValueError(f'Expected curve prompts with shape [b, 4, 2], got {tuple(geom.shape)}.')
            type_idx = 0
            center = geom.mean(dim=1)
            direction = F.normalize(geom[:, -1, :] - geom[:, 0, :], dim=-1, eps=1e-6)
        else:
            geom = boxes.to(device=device, dtype=dtype)
            if geom.ndim != 3 or geom.shape[1:] != (4, 2):
                raise ValueError(f'Expected box prompts with shape [b, 4, 2], got {tuple(geom.shape)}.')
            type_idx = 1
            center = geom[:, 2, :]
            direction = torch.zeros_like(center)
            direction[:, 0] = 1.0

        flat = geom.flatten(1)
        xs = geom[..., 0]
        ys = geom[..., 1]
        xmin = xs.min(dim=1, keepdim=True).values
        xmax = xs.max(dim=1, keepdim=True).values
        ymin = ys.min(dim=1, keepdim=True).values
        ymax = ys.max(dim=1, keepdim=True).values
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        w = (xmax - xmin).clamp_min(1e-6)
        h = (ymax - ymin).clamp_min(1e-6)
        area = w * h
        aspect = w / h
        dx = (geom[:, 3, 0:1] - geom[:, 0, 0:1])
        dy = (geom[:, 3, 1:2] - geom[:, 0, 1:2])

        prompt_feat = torch.cat([flat,
                                 xmin, ymin, xmax, ymax,
                                 cx, cy, w, h,
                                 area, aspect, dx, dy], dim=1)
        prompt_vec = self.prompt_mlp(prompt_feat)
        prompt_vec = prompt_vec + self.type_embeddings.weight[type_idx].to(device=device, dtype=dtype).unsqueeze(0)
        return self.prompt_norm(prompt_vec), center, direction

    def _build_visual_bank(self,
                           encoder_hidden_states: list[torch.Tensor],
                           batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        coords = []
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
            c = getattr(self, f'token_coords_{idx}').to(device=h.device, dtype=h.dtype)
            tokens.append(h)
            coords.append(c.unsqueeze(0).expand(batch_size, -1, -1))

        return torch.cat(tokens, dim=1), torch.cat(coords, dim=1)

    def _support_from_locators(self,
                               locators: torch.Tensor,
                               tokens: torch.Tensor,
                               coords: torch.Tensor,
                               prompt_center: torch.Tensor,
                               prompt_direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        locator_summary = locators.mean(dim=1)
        support_query = self.support_query_proj(locator_summary)
        support_logits = torch.einsum('bd,bsd->bs', support_query, self.support_token_proj(tokens)) * self.score_scale

        support_weights = torch.softmax(support_logits, dim=-1)
        center = torch.einsum('bs,bsd->bd', support_weights, coords)

        direction = self.direction_head(locator_summary)
        direction = direction + 0.25 * prompt_direction
        direction = F.normalize(direction, dim=-1, eps=1e-6)

        fallback = prompt_direction.clone()
        fallback = F.normalize(fallback + torch.tensor([1e-3, 0.0], device=fallback.device, dtype=fallback.dtype),
                               dim=-1,
                               eps=1e-6)
        tiny_dir = direction.norm(dim=-1, keepdim=True) < 1e-6
        direction = torch.where(tiny_dir, fallback, direction)

        temperature = F.softplus(self.temperature_head(locator_summary)) + 1.0
        support_bias = support_logits / temperature
        support_bias = support_bias - support_bias.mean(dim=-1, keepdim=True)
        center = 0.8 * center + 0.2 * prompt_center
        return support_bias, center, direction

    def _build_reader_queries(self,
                              prompt_vec: torch.Tensor,
                              center: torch.Tensor,
                              direction: torch.Tensor,
                              tokens: torch.Tensor) -> torch.Tensor:
        normal = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
        progress = self.reader_progress.to(device=tokens.device, dtype=tokens.dtype).view(1, -1, 1)
        progress = progress.expand(center.size(0), -1, -1)

        center_feat = center.unsqueeze(1).expand(-1, self.reader_num_tokens, -1)
        dir_feat = direction.unsqueeze(1).expand(-1, self.reader_num_tokens, -1)
        normal_feat = normal.unsqueeze(1).expand(-1, self.reader_num_tokens, -1)
        geom_feat = torch.cat([progress,
                               progress * dir_feat[..., :1],
                               progress * dir_feat[..., 1:],
                               center_feat,
                               dir_feat,
                               normal_feat[..., :1]], dim=-1)

        queries = self.reader_base.unsqueeze(0).to(dtype=tokens.dtype, device=tokens.device)
        queries = queries + self.reader_pos_embeddings.unsqueeze(0).to(dtype=tokens.dtype, device=tokens.device)
        queries = queries + prompt_vec.unsqueeze(1)
        queries = queries + self.reader_geom_mlp(geom_feat)
        return self.reader_input_norm(queries)

    def _build_global_queries(self,
                              prompt_vec: torch.Tensor,
                              locator_summary: torch.Tensor,
                              tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.global_num_tokens:
            return None
        queries = self.global_base.unsqueeze(0).to(dtype=tokens.dtype, device=tokens.device)
        queries = queries + self.global_pos_embeddings.unsqueeze(0).to(dtype=tokens.dtype, device=tokens.device)
        queries = queries + prompt_vec.unsqueeze(1) + locator_summary.unsqueeze(1)
        return self.global_norm(queries)

    def forward(self,
                encoder_hidden_states: list[torch.Tensor],
                curves: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, (list, tuple)):
            raise ValueError('encoder_hidden_states must be a list/tuple of feature maps.')
        if len(encoder_hidden_states) != self.num_scales:
            raise ValueError(f'Expected {self.num_scales} feature maps, got {len(encoder_hidden_states)}.')

        ref = encoder_hidden_states[0]
        prompt_vec, prompt_center, prompt_direction = self._encode_prompt(curves=curves,
                                                                          boxes=boxes,
                                                                          dtype=ref.dtype,
                                                                          device=ref.device)
        batch_size = prompt_vec.size(0)
        tokens, coords = self._build_visual_bank(encoder_hidden_states, batch_size)

        locators = self.locator_base.unsqueeze(0).to(dtype=tokens.dtype, device=tokens.device)
        locators = locators + prompt_vec.unsqueeze(1)
        reader = None
        support_bias = None
        center = prompt_center
        direction = prompt_direction

        for idx in range(self.num_rounds):
            if idx > 0:
                locators = locators + self.reader_feedback_proj[idx - 1](reader.mean(dim=1)).unsqueeze(1)

            locators = locators + self.locator_cross_layers[idx](self.locator_norms[idx](locators), tokens)
            locators = self.locator_self_layers[idx](locators)

            support_bias, center, direction = self._support_from_locators(locators,
                                                                          tokens,
                                                                          coords,
                                                                          prompt_center,
                                                                          prompt_direction)

            reader_queries = self._build_reader_queries(prompt_vec, center, direction, tokens)
            if reader is None:
                reader = reader_queries
            else:
                reader = self.reader_round_norms[idx - 1](reader) + 0.5 * reader_queries
            reader = reader + self.reader_cross_layers[idx](self.reader_norms[idx](reader), tokens, bias=support_bias)
            reader = self.reader_self_layers[idx](reader)

        locator_summary = locators.mean(dim=1)
        if self.global_num_tokens:
            global_queries = self._build_global_queries(prompt_vec, locator_summary, tokens)
            global_memory = global_queries + self.global_cross(global_queries, tokens, bias=0.5 * support_bias)
            memory = torch.cat([reader, global_memory], dim=1)
        else:
            memory = reader

        for layer in self.refine_layers:
            memory = layer(memory)
        return self.output_norm(memory)
