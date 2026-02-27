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
                           Llama3ScaledRoPE, llama3_mlp,
                           PositionEmbeddingRandom)


logger = logging.getLogger(__name__)

__all__ = ['bytellama_vision_decoder', 'PromptConditionedMultiScaleAdapter']


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


class PromptConditionedMultiScaleAdapter(nn.Module):
    """
    Fused visual conditioner that combines multi-scale adaptation and prompt
    conditioning into a single module. For each scale it:

    1. downscales + adapts encoder feature maps to decoder embedding dim
    2. conditions tokens with a prompt-dependent gate
    3. scores tokens with a prompt-dependent matcher
    4. keeps top-k tokens per scale and concatenates them

    The module always returns a fixed number of memory tokens
    ``num_samples`` for decoder cross-attention.
    """
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 encoder_embed_dims: list[int],
                 encoder_sizes: list[tuple[int, int]],
                 decoder_embed_dim: int,
                 num_samples: int,
                 ds_factors: list[int] = None,
                 refine_layers: int = 1,
                 refine_num_heads: Optional[int] = None,
                 gate_init: float = 0.0,
                 min_tokens_per_scale: int = 16,
                 use_pos_embeddings: bool = True):
        super().__init__()
        if num_samples < 1:
            raise ValueError('num_samples must be >= 1.')
        if min_tokens_per_scale < 1:
            raise ValueError('min_tokens_per_scale must be >= 1.')
        if ds_factors is None:
            ds_factors = [4, 2, 1]
        if len(encoder_embed_dims) != len(encoder_sizes) or len(encoder_embed_dims) != len(ds_factors):
            raise ValueError('encoder_embed_dims, encoder_sizes, and ds_factors must have the same length.')

        self.num_samples = num_samples
        self.min_tokens_per_scale = min_tokens_per_scale
        self.num_scales = len(encoder_embed_dims)
        if refine_num_heads is None:
            refine_num_heads = num_heads

        mlp_ratio = 4
        self.adapter = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.pos_embeddings = nn.ModuleList()
        self.output_sizes: list[tuple[int, int]] = []
        self.scale_token_lens: list[int] = []

        for encoder_embed_dim, size, ds_factor in zip(encoder_embed_dims, encoder_sizes, ds_factors):
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
            self.output_sizes.append(ds_size)
            self.scale_token_lens.append(ds_size[0] * ds_size[1])

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

        geom_in_dim = 20  # 8 raw coords + 12 derived box/line stats
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_in_dim, decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
        )
        self.geom_norm = RMSNorm(decoder_embed_dim, eps=1e-5)
        self.type_embeddings = nn.Embedding(2, decoder_embed_dim)

        self.gate_token_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
        self.gate_prompt_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.score_token_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)
        self.score_prompt_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False)

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
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=FeedForward(
                    gate_proj=nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim),
                    down_proj=nn.Linear(4 * decoder_embed_dim, decoder_embed_dim),
                    up_proj=None,
                ),
                sa_norm=RMSNorm(decoder_embed_dim, eps=1e-5),
                mlp_norm=RMSNorm(decoder_embed_dim, eps=1e-5),
                sa_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )
            with torch.no_grad():
                layer.sa_scale.scale.fill_(gate_init)
                layer.mlp_scale.scale.fill_(gate_init)
            self.refine_layers.append(layer)

    def _encode_prompt(self,
                       curves: Optional[torch.Tensor],
                       boxes: Optional[torch.Tensor],
                       dtype: torch.dtype,
                       device: torch.device) -> torch.Tensor:
        if curves is not None and boxes is not None:
            raise ValueError('curves and boxes are mutually exclusive.')
        if curves is None and boxes is None:
            raise ValueError('One of curves or boxes must be provided.')

        if curves is not None:
            geom = curves
            type_idx = 0
        else:
            geom = boxes
            type_idx = 1

        if geom.ndim != 3 or geom.shape[1:] != (4, 2):
            raise ValueError(f'Expected prompt geometry with shape [b, 4, 2], got {tuple(geom.shape)}.')

        geom = geom.to(device=device, dtype=dtype)
        flat = geom.flatten(1)  # [b, 8]

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
        derived = torch.cat([xmin, ymin, xmax, ymax, cx, cy, w, h, area, aspect, dx, dy], dim=1)

        geom_feat = torch.cat([flat, derived], dim=1)
        prompt_vec = self.geom_mlp(geom_feat)
        type_embed = self.type_embeddings.weight[type_idx].to(dtype=prompt_vec.dtype, device=prompt_vec.device)
        prompt_vec = prompt_vec + type_embed.unsqueeze(0)
        return self.geom_norm(prompt_vec)

    def _allocate_topk(self, seq_lens: list[int]) -> list[int]:
        total_len = sum(seq_lens)
        if total_len <= 0:
            return [0] * len(seq_lens)

        # start from proportional allocation
        raw = [self.num_samples * s / total_len for s in seq_lens]
        k = [min(s, max(1, int(x))) for s, x in zip(seq_lens, raw)]

        if self.num_samples >= len(seq_lens):
            # encourage each scale to keep at least a few tokens when possible
            for idx, s in enumerate(seq_lens):
                target = min(s, self.min_tokens_per_scale)
                if k[idx] < target:
                    k[idx] = target

        # reduce if over budget
        while sum(k) > self.num_samples:
            candidates = [idx for idx, val in enumerate(k) if val > 1]
            if not candidates:
                break
            idx = max(candidates, key=lambda i: k[i] - raw[i])
            k[idx] -= 1

        # grow if under budget and there is capacity
        while sum(k) < self.num_samples:
            capacities = [seq_lens[idx] - k[idx] for idx in range(len(k))]
            candidates = [idx for idx, cap in enumerate(capacities) if cap > 0]
            if not candidates:
                break
            idx = max(candidates, key=lambda i: raw[i] - k[i])
            k[idx] += 1
        return k

    def forward(self,
                encoder_hidden_states: list[torch.Tensor],
                curves: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, (list, tuple)):
            raise ValueError('encoder_hidden_states must be a list/tuple of feature maps.')
        if len(encoder_hidden_states) != self.num_scales:
            raise ValueError(f'Expected {self.num_scales} feature maps, got {len(encoder_hidden_states)}.')

        ref = encoder_hidden_states[0]
        prompt_vec = self._encode_prompt(curves=curves,
                                         boxes=boxes,
                                         dtype=ref.dtype,
                                         device=ref.device)
        batch_size = prompt_vec.size(0)

        seq_lens = []
        scale_tokens = []
        scale_scores = []

        prompt_gate = self.gate_prompt_proj(prompt_vec).unsqueeze(1)
        prompt_score = self.score_prompt_proj(prompt_vec).unsqueeze(1)

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

            gate = torch.sigmoid(self.gate_token_proj(h) + prompt_gate)
            h = h * gate
            scores = (self.score_token_proj(h) * prompt_score).sum(dim=-1)

            seq_lens.append(h.size(1))
            scale_tokens.append(h)
            scale_scores.append(scores)

        k_alloc = self._allocate_topk(seq_lens)
        selected = []
        all_tokens = []
        for h, scores, k_i in zip(scale_tokens, scale_scores, k_alloc):
            all_tokens.append(h)
            if k_i <= 0:
                continue
            if k_i >= h.size(1):
                selected.append(h)
                continue
            idx = torch.topk(scores, k=k_i, dim=1).indices
            gathered = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            selected.append(gathered)

        if not selected:
            memory = torch.cat(all_tokens, dim=1)
        else:
            memory = torch.cat(selected, dim=1)

        if memory.size(1) < self.num_samples:
            global_token = torch.cat(all_tokens, dim=1).mean(dim=1, keepdim=True)
            pad = global_token.expand(batch_size, self.num_samples - memory.size(1), -1)
            memory = torch.cat([memory, pad], dim=1)
        elif memory.size(1) > self.num_samples:
            memory = memory[:, :self.num_samples, :]

        for layer in self.refine_layers:
            memory = layer(memory)
        return memory
