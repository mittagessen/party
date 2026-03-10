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
"""Model format conversion routines"""
import json
import torch

from safetensors.torch import save_file
from typing import Optional, Union, TYPE_CHECKING

from party.tokenizer import CodePointTokenizer

if TYPE_CHECKING:
    from os import PathLike


def checkpoint_to_kraken(checkpoint_path: Union[str, 'PathLike'],
                         filename: Union[str, 'PathLike'],
                         model_card: Optional[str] = None):
    """
    Converts a lightning checkpoint and optional HTRMoPo model card to the new
    safetensors-based kraken serialization format.
    """
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    tokenizer_state = state_dict.get('_tokenizer_state')
    if tokenizer_state:
        decoder_vocab_size = CodePointTokenizer.load(tokenizer_state).vocab_size
    else:
        decoder_vocab_size = None
        for key, value in state_dict['state_dict'].items():
            if key.endswith('decoder.tok_embeddings.weight') or key.endswith('tok_embeddings.weight'):
                decoder_vocab_size = int(value.shape[0])
                break
        if decoder_vocab_size is None:
            raise ValueError('Unable to determine decoder vocabulary size from checkpoint.')
    # we do not have configurable encoders/decoders
    config = {"prompt_mode": state_dict['datamodule_hyper_parameters']['prompt_mode'],
              "decoder_vocab_size": decoder_vocab_size,
              "decoder_num_layers": 30,
              "decoder_num_heads": 9,
              "decoder_num_kv_heads": 3,
              "decoder_embed_dim": 576,
              "decoder_max_seq_len": 384,
              "decoder_intermediate_dim": 1536,
              "decoder_attn_dropout": 0.0,
              "decoder_norm_eps": 1e-05,
              "decoder_rope_base": 10000,
              "decoder_encoder_max_seq_len": 19200,
              "tokenizer_state": tokenizer_state,
              "encoder_input_size": state_dict['hyper_parameters']['encoder_input_size'],
              "encoder_name": state_dict['hyper_parameters']['encoder']}
    model_type = 'kraken_llama_party'
    metadata = {'model_type': model_type,
                'config': json.dumps(config)}
    if model_card:
        metadata['model_card'] = model_card

    states = {k.removeprefix('model.'): v for k, v in state_dict['state_dict'].items()}
    # we can just save the state dict as our constructor sets up the tensor
    # sharing.
    save_file(states, filename, metadata=metadata)


def update_model_card(model_path: Union[str, 'PathLike'],
                      model_card: Optional[str] = None):
    """
    Replaces the current model card inside a model in the safetensors-based
    kraken serialization format with a new one.
    """
    pass
