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
import math
import torch
import logging

from torch import nn
from typing import Optional
import torch.nn.functional as F

from party.tokenizer import TOKEN_NUM
from party.modules import (MultiHeadAttention, RMSNorm, TanhGate,
                           TransformerCrossAttentionLayer, TransformerDecoder,
                           FeedForward, TransformerSelfAttentionLayer,
                           FusionLayer, scale_hidden_dim_for_mlp,
                           Llama3ScaledRoPE, llama3_mlp)


logger = logging.getLogger(__name__)

__all__ = ['bytellama_vision_decoder', 'PartyHybridNeck']


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
                pos_embeddings=None,
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


# ---------------------------------------------------------------------------
# Hybrid neck building blocks
# ---------------------------------------------------------------------------

class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def _group_count(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _build_2d_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    if norm_type == 'group':
        return nn.GroupNorm(_group_count(channels), channels)
    if norm_type == 'layer':
        return _LayerNorm2d(channels)
    raise ValueError(f'Unsupported neck normalization: {norm_type}')


def _build_activation(name: str) -> nn.Module:
    if name == 'gelu':
        return nn.GELU()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'silu':
        return nn.SiLU(inplace=True)
    raise ValueError(f'Unsupported activation: {name}')


def _build_2d_sincos_position_embedding(height: int,
                                        width: int,
                                        embed_dim: int,
                                        temperature: float = 10000.0,
                                        *,
                                        device: Optional[torch.device] = None,
                                        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError('Embedding dimension must be divisible by 4 for 2D sin-cos PE.')
    grid_y = torch.arange(height, dtype=torch.float32, device=device)
    grid_x = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
    omega = 1.0 / (temperature ** omega)
    out_x = grid_x.reshape(-1, 1) * omega.reshape(1, -1)
    out_y = grid_y.reshape(-1, 1) * omega.reshape(1, -1)
    pe = torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)
    return pe.to(dtype=dtype).unsqueeze(0)


class _ConvNormAct2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 *,
                 groups: int = 1,
                 padding: Optional[int] = None,
                 norm_type: str = 'group',
                 act: Optional[str] = 'silu'):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.norm = _build_2d_norm(norm_type, out_channels)
        self.act = nn.Identity() if act is None else _build_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _ResidualConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 norm_type: str = 'group',
                 act: str = 'silu'):
        super().__init__()
        self.conv1 = _ConvNormAct2d(in_channels, out_channels, 3, norm_type=norm_type, act=act)
        self.conv2 = _ConvNormAct2d(out_channels, out_channels, 3, norm_type=norm_type, act=None)
        self.shortcut = (nn.Identity() if in_channels == out_channels else
                         _ConvNormAct2d(in_channels, out_channels, 1, norm_type=norm_type, act=None))
        self.act = _build_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class _FusionConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 depth: int = 2,
                 norm_type: str = 'group',
                 act: str = 'silu'):
        super().__init__()
        layers = [_ResidualConvBlock(in_channels, out_channels, norm_type=norm_type, act=act)]
        for _ in range(depth - 1):
            layers.append(_ResidualConvBlock(out_channels, out_channels, norm_type=norm_type, act=act))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _SpatialTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float = 0.0,
                 act: str = 'gelu'):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, dim_feedforward),
                                 _build_activation(act),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.norm1(x)
        if pos_embed is not None:
            y = y + pos_embed
        attn_out, _ = self.self_attn(y, y, value=self.norm1(x), need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class _SpatialTransformerEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 act: str = 'gelu'):
        super().__init__()
        self.layers = nn.ModuleList([_SpatialTransformerEncoderLayer(embed_dim,
                                                                     num_heads,
                                                                     dim_feedforward,
                                                                     dropout=dropout,
                                                                     act=act)
                                     for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        return x


# ---------------------------------------------------------------------------
# PartyHybridNeck
# ---------------------------------------------------------------------------

class PartyHybridNeck(nn.Module):
    """
    A hybrid multiscale neck with shared channel projection, optional
    transformer encoding on selected scales, top-down FPN fusion, and
    bottom-up PAN fusion before flattening into decoder memory tokens.

    Args:
        encoder_embed_dims: Per-stage channel dimensions from the encoder.
        encoder_sizes: Per-stage spatial (h, w) sizes from the encoder.
        decoder_embed_dim: Output embedding dimension matching the decoder.
        hidden_dim: Shared channel dimension for the FPN/PAN convolutions.
        num_heads: Number of attention heads in the spatial transformer.
        num_encoder_layers: Number of transformer layers on selected scales.
        use_encoder_idx: Which scale indices receive transformer encoding
                         (default: last scale only).
        output_ds_factors: Per-scale output downsampling factors.
        norm_type: 2D normalization type ('group', 'batch', 'layer').
        dim_feedforward: Hidden dim of the spatial transformer MLP.
        fusion_depth: Number of residual conv blocks in each FPN/PAN stage.
    """
    def __init__(self,
                 encoder_embed_dims: list[int],
                 encoder_sizes: list[tuple[int, int]],
                 decoder_embed_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_encoder_layers: int = 1,
                 use_encoder_idx: Optional[list[int]] = None,
                 output_ds_factors: Optional[list[int]] = None,
                 norm_type: str = 'group',
                 dim_feedforward: int = 1024,
                 fusion_depth: int = 2):
        super().__init__()
        num_scales = len(encoder_embed_dims)
        if output_ds_factors is None:
            output_ds_factors = [2] * num_scales
        if len(output_ds_factors) != num_scales:
            raise ValueError('output_ds_factors must match the number of encoder feature maps.')
        if use_encoder_idx is None:
            use_encoder_idx = [num_scales - 1]
        use_encoder_idx = sorted(use_encoder_idx)

        self.hidden_dim = hidden_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_sizes = list(encoder_sizes)
        self.output_ds_factors = list(output_ds_factors)
        self.use_encoder_idx = use_encoder_idx
        self.output_sizes: list[tuple[int, int]] = [
            (h // ds, w // ds)
            for (h, w), ds in zip(encoder_sizes, self.output_ds_factors)
        ]

        # 1×1 projection to shared hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
                _build_2d_norm(norm_type, hidden_dim),
            )
            for channels in encoder_embed_dims
        ])

        # Optional spatial transformer on selected scales
        self.encoders = nn.ModuleList([
            _SpatialTransformerEncoder(hidden_dim,
                                       num_heads,
                                       dim_feedforward,
                                       num_encoder_layers)
            for _ in self.use_encoder_idx
        ])

        # Top-down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(num_scales - 1):
            self.lateral_convs.append(
                _ConvNormAct2d(hidden_dim, hidden_dim, 1, norm_type=norm_type, act='silu')
            )
            self.fpn_blocks.append(
                _FusionConvBlock(hidden_dim * 2, hidden_dim, depth=fusion_depth, norm_type=norm_type, act='silu')
            )

        # Bottom-up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(num_scales - 1):
            self.downsample_convs.append(
                _ConvNormAct2d(hidden_dim, hidden_dim, 3, stride=2, norm_type=norm_type, act='silu')
            )
            self.pan_blocks.append(
                _FusionConvBlock(hidden_dim * 2, hidden_dim, depth=fusion_depth, norm_type=norm_type, act='silu')
            )

        # Per-scale output downsampling + projection
        self.output_downsample = nn.ModuleList([
            self._make_output_downsample(hidden_dim, ds, norm_type)
            for ds in self.output_ds_factors
        ])
        self.output_proj = nn.ModuleList([
            nn.Conv2d(hidden_dim, decoder_embed_dim, kernel_size=1, bias=False)
            for _ in range(num_scales)
        ])
        self.level_embeddings = nn.Parameter(torch.zeros(num_scales, decoder_embed_dim))
        nn.init.normal_(self.level_embeddings, std=0.02)

    @staticmethod
    def _make_output_downsample(channels: int,
                                ds_factor: int,
                                norm_type: str) -> nn.Module:
        if ds_factor <= 1:
            return nn.Identity()
        return nn.Sequential(
            _ConvNormAct2d(channels, channels, ds_factor, stride=ds_factor,
                           groups=channels, padding=0, norm_type=norm_type, act='silu'),
            _ConvNormAct2d(channels, channels, 1, norm_type=norm_type, act='silu'),
        )

    def _encode_scale(self, feat: torch.Tensor, encoder: _SpatialTransformerEncoder) -> torch.Tensor:
        batch_size, _, height, width = feat.shape
        src = feat.flatten(2).transpose(1, 2)
        pos_embed = _build_2d_sincos_position_embedding(
            height, width, self.hidden_dim, device=src.device, dtype=src.dtype
        )
        src = encoder(src, pos_embed=pos_embed)
        return src.transpose(1, 2).reshape(batch_size, self.hidden_dim, height, width).contiguous()

    def _flatten_output(self, feat: torch.Tensor, level_idx: int) -> torch.Tensor:
        feat = self.output_downsample[level_idx](feat)
        feat = self.output_proj[level_idx](feat)
        _, _, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        pos_embed = _build_2d_sincos_position_embedding(
            height, width, self.decoder_embed_dim, device=tokens.device, dtype=tokens.dtype
        )
        level_embed = self.level_embeddings[level_idx].view(1, 1, -1)
        return tokens + pos_embed + level_embed

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        # Project all scales to shared hidden_dim
        proj_feats = [proj(feat) for proj, feat in zip(self.input_proj, encoder_hidden_states)]

        # Transformer encoding on selected scales
        for encoder, feat_idx in zip(self.encoders, self.use_encoder_idx):
            proj_feats[feat_idx] = self._encode_scale(proj_feats[feat_idx], encoder)

        # Top-down FPN
        inner_outs = [proj_feats[-1]]
        for level_idx, feat_idx in enumerate(range(len(proj_feats) - 1, 0, -1)):
            high = self.lateral_convs[level_idx](inner_outs[0])
            low = proj_feats[feat_idx - 1]
            inner_outs[0] = high
            upsampled = F.interpolate(high, size=low.shape[-2:], mode='nearest')
            inner_out = self.fpn_blocks[level_idx](torch.cat([upsampled, low], dim=1))
            inner_outs.insert(0, inner_out)

        # Bottom-up PAN
        outs = [inner_outs[0]]
        for level_idx in range(len(inner_outs) - 1):
            low = outs[-1]
            high = inner_outs[level_idx + 1]
            downsampled = self.downsample_convs[level_idx](low)
            outs.append(self.pan_blocks[level_idx](torch.cat([downsampled, high], dim=1)))

        # Flatten and concatenate all scales
        return torch.cat([self._flatten_output(feat, level_idx)
                          for level_idx, feat in enumerate(outs)], dim=1)
