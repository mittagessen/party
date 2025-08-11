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
import logging
from typing import Generator, Optional, Union, List, Tuple, TYPE_CHECKING

import json
import torch
from torch import nn

from party.modules import (MultiHeadAttention, RMSNorm, TanhGate,
                           TransformerCrossAttentionLayer, TransformerDecoder,
                           FeedForward, TransformerSelfAttentionLayer,
                           FusionLayer, scale_hidden_dim_for_mlp,
                           Llama3ScaledRoPE, llama3_mlp, PromptEncoder)

from party.tokenizer import OctetTokenizer, TOKEN_NUM

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)

__all__ = ['bytellama_vision_decoder', 'PartyModel']


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
                             encoder_max_seq_len: int = 4800,  # start of fusion parameters
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
        encoder_max_seq_len (int): maximum sequence length the encoder will be run with, as used
            by :func:`~party.modules.KVCache`.
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
    for idx in range(1, num_layers + 1):

        # Self attention layers for text decoder
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
            attn = MultiHeadAttention(
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

            mlp = llama3_mlp(dim=config['embed_dim'], hidden_dim=hidden_dim)
            xattn_layer = TransformerCrossAttentionLayer(
                attn=attn,
                mlp=mlp,
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
        rweight = torch.zeros(TOKEN_NUM - 259, config['embed_dim'])
        torch.nn.init.xavier_uniform_(rweight)
        state_dict['tok_embeddings.weight'] = torch.cat([state_dict['tok_embeddings.weight'], rweight], dim=0)
        rweight = rweight.clone()
        torch.nn.init.xavier_uniform_(rweight)
        state_dict['output.weight'] = torch.cat([state_dict['output.weight'], rweight], dim=0)
        decoder.load_state_dict(state_dict, strict=False)

    return decoder


def party_adapter(num_layers: int,
                  num_heads: int,
                  encoder_embed_dim: int,
                  decoder_embed_dim: int) -> nn.Sequential:
    """
    Builds an adapter head consisting of `num_layers` self attention layers
    followed by a linear projection of encoder_embed_dim to decoder_embed_dim.
    """
    mlp_ratio = 4
    hidden_dim = int(mlp_ratio * encoder_embed_dim)
    head_dim = encoder_embed_dim // num_heads
    num_kv_heads = num_heads
    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(embed_dim=encoder_embed_dim,
                                       num_heads=num_heads,
                                       num_kv_heads=num_heads,
                                       head_dim=head_dim,
                                       q_proj=nn.Linear(encoder_embed_dim, num_heads * head_dim, bias=False),
                                       k_proj=nn.Linear(encoder_embed_dim, num_kv_heads * head_dim, bias=False),
                                       v_proj=nn.Linear(encoder_embed_dim, num_kv_heads * head_dim, bias=False),
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
    return nn.Sequential(*layers)


class PartyModel(nn.Module):
    """
    The party fusion model.

    Args:
        encoder: A timm image encoder model
        decoder: Text decoder model
        encoder_embed_dim: Embedding dimension of the encoder
        decoder_embed_dim: Embedding dimension of the decoder
    """

    tokenizer = OctetTokenizer()

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 encoder_embed_dim: int,
                 decoder_embed_dim: int,
                 adapter_num_layers: int = 4,
                 adapter_num_heads: int = 8):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.adapter = party_adapter(adapter_num_layers,
                                     adapter_num_heads,
                                     encoder_embed_dim,
                                     decoder_embed_dim)

        self.line_embedding = PromptEncoder(decoder_embed_dim)

        self.ready_for_generation = False

    @classmethod
    def from_safetensors(cls, filename: Union[str, 'PathLike']) -> 'PartyModel':
        """
        Loads model weights from a safetensors-based kraken serialization.
        """
        import timm
        from safetensors import safe_open

        with safe_open(filename, framework='pt') as f:
            metadata = f.metadata()
            if metadata['model_type'] != 'kraken_llama_party':
                raise ValueError(f'{filename} is not a llama party model. Got type {metadata["model_type"]}, expected: kraken_llama_party.')
            config = json.loads(metadata['config'])
            prompt_mode = config.pop('prompt_mode')
            encoder_config = {k[8:]: v for k, v in config.items() if k.startswith('encoder_')}
            decoder_config = {k[8:]: v for k, v in config.items() if k.startswith('decoder_')}

            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        # enable fused attn in encoder
        timm.layers.use_fused_attn(experimental=True)

        encoder_model = timm.create_model(encoder_config['name'],
                                          pretrained=False,
                                          num_classes=0,
                                          img_size=encoder_config['input_size'],
                                          global_pool='')

        l_idx = encoder_model.prune_intermediate_layers(indices=(-2,), prune_head=True, prune_norm=True)[0]
        decoder_model = bytellama_vision_decoder(**decoder_config)

        model = cls(encoder=encoder_model,
                    decoder=decoder_model,
                    encoder_embed_dim=encoder_model.feature_info[l_idx]['num_chs'],
                    decoder_embed_dim=decoder_model.tok_embeddings.embedding_dim)

        model.load_state_dict(state_dict)

        model.line_prompt_mode = prompt_mode

        return model

    def setup_caches(self,
                     batch_size: int,
                     dtype: torch.dtype,
                     *,
                     encoder_max_seq_len: int = None,
                     decoder_max_seq_len: int = None):
        """
        Sets up key-value attention caches for inference for ``self.decoder``.
        For each layer in ``self.decoder.layers``:
        - :class:`party.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
        - :class:`party.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
        - :class:`party.modules.fusion.FusionLayer` will use both ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum encoder cache sequence length.
            decoder_max_seq_len (int): maximum decoder cache sequence length.
        """
        self.decoder.setup_caches(batch_size,
                                  dtype,
                                  encoder_max_seq_len=encoder_max_seq_len,
                                  decoder_max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.decoder.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.decoder.caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.
        """
        self.decoder.reset_caches()

    def forward(self,
                tokens: torch.Tensor,
                *,
                encoder_input: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_curves: Optional[torch.Tensor] = None,
                encoder_boxes: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            encoder_input: Optional input for the encoder.
            encoder_hidden_states: Optional encoder embeddings with curve
                                   embeddings already added.
            encoder_curves: Optional curves to be embedded and added to encoder
                            embeddings. Mutually exclusive with `encoder_boxes`.
            encoder_boxes: Optional boxes to be embedded and added to encoder
                            embeddings. Mutually exclusive with `encoder_curves`.
            input_pos: Optional tensor which contains the position ids of each
                       token. During training, this is used to indicate the
                       positions of each token relative to its sample when
                       packed, shape ``[b x s]``.  During inference, this
                       indicates the position of the current token.  If none,
                       assume the index of the token is its position id.
                       Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape ``[b x s x v]`` or a list of layer \
                output tensors defined by ``output_hidden_states`` with the \
                final output tensor appended to the list.

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # During decoding, encoder_input will only be provided
        # for new inputs. Previous encoder outputs are cached
        # in the decoder cache.
        if encoder_input is not None:

            encoder_hidden_states = self.forward_encoder_embeddings(encoder_input)
            # expand encoder_hidden_states from (1, s_e, d) to (b, s_e, d)
            encoder_hidden_states = encoder_hidden_states.repeat(tokens.size(0), 1, 1)

            # add curve embeddings to encoder hidden states after adaptatio to decoder_embed_dim
            line_embeds = self.line_embedding(curves=encoder_curves,
                                              boxes=encoder_boxes).unsqueeze(1).expand(-1, encoder_hidden_states.size(1), -1)
            encoder_hidden_states = encoder_hidden_states + line_embeds

        output = self.decoder(tokens=tokens,
                              mask=mask,
                              encoder_input=encoder_hidden_states,
                              encoder_mask=encoder_mask,
                              input_pos=input_pos)
        return output

    def forward_encoder_embeddings(self, encoder_input):
        """
        Computes the encoder embeddings *without* adding the curve positional
        embeddings.
        """
        encoder_hidden_states = self.encoder(encoder_input)
        b, e = encoder_hidden_states.shape[0], encoder_hidden_states.shape[-1]
        encoder_hidden_states = encoder_hidden_states.view(b, -1, e)
        return self.adapter(encoder_hidden_states)

    @torch.inference_mode()
    def prepare_for_generation(self,
                               batch_size: int = 8,
                               max_encoder_seq_len: int = 19200,
                               max_generated_tokens: int = 384,
                               device: torch.device = torch.device('cpu')):

        if self.ready_for_generation:
            logger.debug('Model has already been prepared for generation!')

        self._batch_size = batch_size
        self._max_generated_tokens = max_generated_tokens
        self._max_encoder_seq_len = max_encoder_seq_len

        # set up caches
        self.setup_caches(batch_size=batch_size,
                          encoder_max_seq_len=max_encoder_seq_len,
                          decoder_max_seq_len=max_generated_tokens,
                          dtype=next(self.encoder.parameters()).dtype)

        # create batch size number of BOS tokens
        self.ready_for_generation = True

    @torch.inference_mode()
    def predict_tokens(self,
                       encoder_input: torch.FloatTensor,
                       curves: Optional[torch.FloatTensor] = None,
                       boxes: Optional[torch.FloatTensor] = None,
                       languages: Optional[List[str]] = None) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Predicts text from an input page image and a number of cubic Bézier
        curves.

        Args:
            encoder_input: Image input for the encoder with shape ``1 x c x h x w``
            curves: Curves to be embedded and added to the encoder embeddings (``n x 4 x 2``)
            boxes: Boxes to be embedded and added to the encoder embeddings (``n x 4 x 2``)
            batch_size: Number of curves to generate text for simultaneously.

        Yields:
            One tensor of integer labels and one tensor of float confidences,
            each with shape ``n x s`` where ``s` is the length of the longest
            generated sequence or `max_generated_tokens`. BOS and EOS have
            already been stripped from the token sequences. Entries beyond the
            EOS are padded with zeroes.
        """
        if curves is not None and boxes is not None:
            raise ValueError('`curves` and `boxes` are mutually exclusive.')
        if curves is None and boxes is None:
            raise ValueError('One of `curves` or `boxes` needs to be set.')
        logger.debug('Computing encoder embeddings')

        encoder_hidden_states = self.forward_encoder_embeddings(encoder_input).repeat(self._batch_size, 1, 1)
        device = encoder_hidden_states.device

        _prompt = torch.tensor(self.tokenizer.encode('', langs=languages, add_eos=False),
                               device=device,
                               dtype=torch.long).repeat(self._batch_size, 1)
        _prompt_length = _prompt.size(1)

        eos_token = torch.tensor(self.tokenizer.eos_id, device=device, dtype=torch.long)

        line_pos = curves if curves is not None else boxes
        batches = torch.split(line_pos, self._batch_size)

        total_response_length = _prompt_length + self._max_generated_tokens
        # generate a regular causal mask
        masks = torch.tril(torch.ones(total_response_length,
                                      self._max_generated_tokens,
                                      dtype=torch.bool,
                                      device=device)).unsqueeze(0)
        input_pos = torch.arange(0, total_response_length, device=device).unsqueeze(0)

        # Mask is shape (batch_size, max_seq_len, image_embedding_len)
        encoder_mask = torch.ones((self._batch_size,
                                   _prompt_length,
                                   encoder_hidden_states.size(1)),
                                  dtype=torch.bool,
                                  device=device)

        for batch_idx, batch in enumerate(batches):

            bsz = batch.size(0)

            self.reset_caches()

            logger.info(f'Processing batch {batch_idx} of {len(batches)}')
            if bsz != self._batch_size:
                logger.debug(f'Resizing caches for last batch ({self._batch_size} -> {bsz})')
                self.setup_caches(batch_size=bsz,
                                  encoder_max_seq_len=self._max_encoder_seq_len,
                                  decoder_max_seq_len=self._max_generated_tokens,
                                  dtype=next(self.encoder.parameters()).dtype)

            logger.debug('Adding line embeddings to encoder states.')
            # add line embeddings to encoder hidden states
            line_embeds = self.line_embedding(batch).unsqueeze(1).expand(-1, encoder_hidden_states.size(1), -1)
            exp_encoder_hidden_states = encoder_hidden_states[:bsz, ...] + line_embeds

            logger.debug('Prefilling cache.')
            # prefill step
            curr_masks = masks[:, :_prompt_length]
            logits = self.forward(tokens=_prompt[:bsz, ...],
                                  encoder_hidden_states=exp_encoder_hidden_states,
                                  encoder_mask=encoder_mask[:bsz, ...],
                                  mask=curr_masks,
                                  input_pos=input_pos[:, :_prompt_length].squeeze())
            tokens = torch.argmax(logits, dim=-1)[:, -1:]
            confs = logits[:, -1].softmax(-1)
            generated_tokens = [tokens[:, -1]]
            generated_confidences = [confs.gather(-1, tokens).squeeze(1)]
            logger.debug(f'Generated {generated_tokens[-1]} with conf {generated_confidences[-1]}')
            curr_pos = _prompt_length

            # keeps track of EOS tokens emitted by each sequence in a batch
            eos_token_reached = torch.zeros(bsz, dtype=torch.bool, device=device)
            eos_token_reached |= tokens[:, -1] == eos_token

            # mask used for setting all values from EOS token to pad_id in output sequences.
            eos_token_mask = torch.ones(bsz, 0, dtype=torch.int32, device=device)

            if eos_token_reached.all():
                break

            for _ in range(self._max_generated_tokens - (1 + _prompt_length)):
                logger.debug('Generating...')
                # update eos_token_mask if an EOS token was emitted in a previous step
                eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(bsz, 1)], dim=-1)

                curr_input_pos = input_pos[:, curr_pos]
                curr_masks = masks[:, curr_pos, None, :]

                # no need for encoder embeddings anymore as they're in the cache now
                logits = self.forward(tokens=tokens.clone(),
                                      mask=curr_masks,
                                      input_pos=curr_input_pos)
                tokens = torch.argmax(logits, dim=-1)
                confs = logits[:, -1].softmax(-1)
                generated_tokens.append(tokens[:, -1])
                generated_confidences.append(confs.gather(-1, tokens).squeeze(1))
                logger.debug(f'Generated {generated_tokens[-1]} with conf {generated_confidences[-1]}')

                curr_pos += 1

                eos_token_reached |= tokens[:, -1] == eos_token
                if eos_token_reached.all():
                    break

            eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(bsz, 1)], dim=-1)

            # mask out generated tokens beyond EOS token
            generated_tokens = torch.stack(generated_tokens).T
            generated_confidences = torch.stack(generated_confidences).T

            generated_tokens *= eos_token_mask
            generated_confidences *= eos_token_mask
            yield generated_tokens[..., :-1], generated_confidences[..., :-1]

    @torch.inference_mode
    def predict_string(self,
                       encoder_input: torch.FloatTensor,
                       curves: Optional[torch.FloatTensor] = None,
                       boxes: Optional[torch.FloatTensor] = None,
                       languages: Optional[List[str]] = None) -> Generator[str, None, None]:
        """
        Predicts text from an input page image and a number of cubic Bézier
        curves.

        Args:
            encoder_input: Image input for the encoder with shape ``[1 x c x h x w]``
            curves: Curves to be embedded and added to the encoder embeddings (``n x 4 x 2``)
            languages: ISO693-3 identifiers of the languages of the lines.

        Yields:
        """
        for preds, confs in self.predict_tokens(encoder_input=encoder_input,
                                                curves=curves,
                                                boxes=boxes,
                                                languages=languages):
            for pred, conf in zip(preds, confs):
                yield self.tokenizer.decode_with_confs(pred[pred != 0], conf)
