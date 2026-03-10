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

from torch import nn
from typing import Optional

from party.tokenizer import TOKEN_NUM
from party.modules import (MultiHeadAttention, RMSNorm, TanhGate,
                           TransformerCrossAttentionLayer, TransformerDecoder,
                           FeedForward, TransformerSelfAttentionLayer,
                           FusionLayer, scale_hidden_dim_for_mlp,
                           Llama3ScaledRoPE, llama3_mlp, PrototypeHead,
                           PositionEmbeddingRandom)


logger = logging.getLogger(__name__)

__all__ = ['bytellama_vision_decoder', 'PartyMultiScaleAdapter']


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
                             temperature_init: float = 10.0,
                             margin: float = 0.0,
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
              'fusion_interval': fusion_interval,
              'temperature_init': temperature_init,
              'margin': margin}

    if pretrained:
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo_id=pretrained, filename='config.json'), 'r') as fp:
            config.update(json.load(fp))
            # Output/input token spaces are task-specific and initialized from scratch.
            config['vocab_size'] = vocab_size
            config['temperature_init'] = temperature_init
            config['margin'] = margin

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
    output_proj = PrototypeHead(config['embed_dim'],
                                config['vocab_size'],
                                temperature_init=config['temperature_init'],
                                margin=config['margin'])
    output_proj.tie_embeddings(tok_embeddings)

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
        # Task vocabulary is dynamic and differs from the pretrained decoder.
        state_dict.pop('tok_embeddings.weight', None)
        state_dict.pop('output.weight', None)
        decoder.load_state_dict(state_dict, strict=False)

    return decoder


class PartyMultiScaleAdapter(nn.Module):
    """
    Multi-scale adapter head that processes features from multiple encoder
    stages independently, then concatenates them into a single sequence.

    Args:
        num_layers: Number of self-attention layers per scale.
        num_heads: Number of attention heads per scale.
        encoder_embed_dims: Channel dimensions for each encoder stage.
        encoder_sizes: Spatial dimensions (h, w) for each encoder stage.
        decoder_embed_dim: Output embedding dimension matching the decoder.
        ds_factors: Downsampling factor per scale. Defaults to [4, 2, 1] which
                    equalizes the spatial resolution across scales.
    """
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 encoder_embed_dims: list[int],
                 encoder_sizes: list[tuple[int, int]],
                 decoder_embed_dim: int,
                 ds_factors: list[int] = None,
                 use_pos_embeddings: bool = True):
        super().__init__()
        if ds_factors is None:
            ds_factors = [4, 2, 1]
        mlp_ratio = 4
        self.adapter = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.pos_embeddings = nn.ModuleList()
        self.output_sizes: list[tuple[int, int]] = []

        for encoder_embed_dim, size, ds_factor in zip(encoder_embed_dims, encoder_sizes, ds_factors):
            hidden_dim = int(mlp_ratio * encoder_embed_dim)
            head_dim = encoder_embed_dim // num_heads

            # depthwise-separable downsampling convolution
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
            self.output_sizes.append(ds_size)

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

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        os = []
        for idx, hidden_state in enumerate(encoder_hidden_states):
            hidden_state = self.downsample[idx](hidden_state)
            hidden_state = hidden_state.flatten(-2).transpose(-1, -2)
            hidden_state = self.adapter[idx](hidden_state)
            os.append(self.pos_embeddings[idx](hidden_state))
        return torch.cat(os, dim=1)
