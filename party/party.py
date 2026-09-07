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
import timm
import torch
import logging

from torch import nn
import numpy as np

from dataclasses import replace
from kraken.models import RecognitionBaseModel
from kraken.containers import BaselineOCRRecord, BBoxOCRRecord
from lightning.fabric import Fabric
from collections.abc import Generator
from typing import Optional, Union, TYPE_CHECKING, Any

from party.modules import (PromptEncoder, MultiHeadAttention, FeedForward,
                           TransformerSelfAttentionLayer, RMSNorm, TanhGate,
                           PrototypeHead)
from party.tokenizer import CodePointTokenizer, CODEPOINT_OFFSET, MASK_ID
from party.configs import get_model_variant, PartyRecognitionInferenceConfig
from party.fusion import bytellama_vision_decoder
from party.dataset import get_default_transforms, _to_curve, _to_bbox

if TYPE_CHECKING:
    from os import PathLike
    from PIL import Image
    from kraken.configs import Config
    from kraken.containers import Segmentation, ocr_record, BaselineLine

logger = logging.getLogger(__name__)

__all__ = ['PartyModel']


def _box_prompt_fn(line: Union['BaselineLine', 'BBoxLine'],
                   im_size: tuple[int, int]) -> Optional[list[float]]:
    """
    Converts a BBoxLine or BaselineLine to a bounding box representation.
    """
    if line.type == 'bbox':
        xmin, ymin, xmax, ymax = line.bbox
        boundary = [(xmin, ymin), (xmax, ymax)]
    else:
        boundary = line.boundary
    bbox = _to_bbox(boundary, im_size)
    return bbox.as_py() if bbox is not None else None


def _curve_prompt_fn(line: 'BaselineLine',
                     im_size: tuple[int, int]) -> Optional[list[float]]:
    """
    Converts a BaselineLine to a cubic Bézier curve.
    """
    curve = _to_curve(line.baseline, im_size)
    return curve.as_py() if curve is not None else None


class SingleScaleAdapter(nn.Module):
    """
    Single-scale adapter mapping encoder features to the decoder embed dim.
    """
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 encoder_embed_dim: int,
                 decoder_embed_dim: int):
        super().__init__()
        mlp_ratio = 4
        hidden_dim = int(mlp_ratio * encoder_embed_dim)
        head_dim = encoder_embed_dim // num_heads
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
        self.adapter = nn.Sequential(*layers)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.adapter(encoder_hidden_states.flatten(1, 2))


class PartyModel(nn.Module, RecognitionBaseModel):
    """
    The party fusion model.

    Args:
        encoder: A timm image encoder model
        decoder: Text decoder model
        encoder_embed_dim: Embedding dimension of the encoder
        decoder_embed_dim: Embedding dimension of the decoder
    """
    model_type = ['recognition']
    _kraken_min_version = '6.0.0'

    def __init__(self, **kwargs):
        super().__init__()

        tokenizer_state = kwargs.pop('tokenizer_state', None)
        tokenizer_arg = kwargs.pop('tokenizer', None)
        proto_temperature_init = kwargs.pop('proto_temperature_init', 10.0)
        proto_margin = kwargs.pop('proto_margin', 0.0)

        self.user_metadata: dict[str, Any] = {'accuracy': [],
                                              'metrics': []}
        self.user_metadata.update(kwargs)
        self.user_metadata['proto_temperature_init'] = proto_temperature_init
        self.user_metadata['proto_margin'] = proto_margin
        # autoregressive weights load into this model without complaint but
        # decode into garbage, so record how it was trained to decode
        self.user_metadata['decoder_mode'] = 'diffusion'

        if isinstance(tokenizer_arg, CodePointTokenizer):
            self.tokenizer = tokenizer_arg
        elif isinstance(tokenizer_arg, dict):
            self.tokenizer = CodePointTokenizer.load(tokenizer_arg)
        else:
            self.tokenizer = CodePointTokenizer.load(tokenizer_state)
        self.tokenizer.freeze()
        self.user_metadata['tokenizer'] = self.tokenizer.save()

        model_variant = kwargs['model_variant']
        image_size = kwargs['image_size']
        pretrained = kwargs.get('pretrained', True)
        variant = get_model_variant(model_variant)

        # 1. Build encoder
        timm.layers.use_fused_attn(experimental=True)
        encoder = timm.create_model(variant['encoder']['name'],
                                    pretrained=pretrained,
                                    img_size=image_size,
                                    features_only=True,
                                    out_indices=variant['encoder']['out_indices'])
        encoder_embed_dim = encoder.feature_info.channels()[0]
        reduction = encoder.feature_info.reduction()[0]
        adapter_seq_len = (image_size[0] // reduction) * (image_size[1] // reduction)
        self.encoder_max_seq_len = adapter_seq_len

        # 2. Build decoder
        decoder = bytellama_vision_decoder(
            vocab_size=self.tokenizer.vocab_size,
            pretrained=variant['decoder']['name'] if pretrained else None,
            num_layers=variant['decoder']['num_layers'],
            num_heads=variant['decoder']['num_heads'],
            num_kv_heads=variant['decoder']['num_kv_heads'],
            embed_dim=variant['decoder']['embed_dim'],
            intermediate_dim=variant['decoder']['intermediate_dim'],
            encoder_max_seq_len=adapter_seq_len,
            fusion_interval=variant['fusion_interval'],
            temperature_init=proto_temperature_init,
            margin=proto_margin)
        decoder_embed_dim = decoder.tok_embeddings.embedding_dim

        # 3. Build adapter
        adapter = SingleScaleAdapter(num_layers=variant['adapter']['num_layers'],
                                     num_heads=variant['adapter']['num_heads'],
                                     encoder_embed_dim=encoder_embed_dim,
                                     decoder_embed_dim=decoder_embed_dim)

        # 4. Line embedding (additive only)
        line_embedding = PromptEncoder(decoder_embed_dim)

        self.nn = nn.ModuleDict({'encoder': encoder,
                                 'decoder': decoder,
                                 'adapter': adapter,
                                 'line_embedding': line_embedding})

        self.ready_for_generation = False

    def setup_caches(self,
                     batch_size: int,
                     dtype: torch.dtype,
                     *,
                     encoder_max_seq_len: int = None,
                     decoder_max_seq_len: int = None,
                     self_attn_cache: bool = False):
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
            self_attn_cache (bool): whether to cache decoder self-attention.
                Defaults to False: the sampler re-reads the whole canvas every
                denoising pass and would overrun the cache, while the encoder
                keys/values it does cache are stable across passes.
        """
        self.nn['decoder'].setup_caches(batch_size,
                                        dtype,
                                        encoder_max_seq_len=encoder_max_seq_len,
                                        decoder_max_seq_len=decoder_max_seq_len,
                                        self_attn_cache=self_attn_cache)
        self._cache_batch_size = batch_size

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.nn['decoder'].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.nn['decoder'].caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.
        """
        self.nn['decoder'].reset_caches()

    def forward(self,
                tokens: torch.Tensor,
                *,
                encoder_input: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_curves: Optional[torch.Tensor] = None,
                encoder_boxes: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> Union[torch.Tensor, list[torch.Tensor]]:
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
            adapter_output = self.forward_encoder_embeddings(encoder_input)
            line_embeddings = self.nn['line_embedding'](curves=encoder_curves,
                                                        boxes=encoder_boxes)
            encoder_hidden_states = adapter_output + line_embeddings.unsqueeze(1)

        output = self.nn['decoder'](tokens=tokens,
                                    mask=mask,
                                    encoder_input=encoder_hidden_states,
                                    encoder_mask=encoder_mask,
                                    input_pos=input_pos)
        return output

    def _prototype_head(self) -> PrototypeHead:
        output = self.nn['decoder'].output
        if not isinstance(output, PrototypeHead):
            raise TypeError('Decoder output projection is not a PrototypeHead.')
        return output

    @torch.no_grad()
    def extend_prototypes(self, new_vectors: torch.Tensor) -> nn.Embedding:
        """
        Extends the shared token/prototype embedding table with new vectors.
        """
        head = self._prototype_head()
        resized = head.extend_prototypes(new_vectors)
        head.tie_embeddings(resized)
        self.nn['decoder'].tok_embeddings = resized
        return resized

    @torch.no_grad()
    def add_codepoints_with_prototypes(self,
                                       codepoints: list[int],
                                       prototype_vectors: torch.Tensor) -> list[int]:
        """
        Registers new code points and appends matching prototype vectors.

        Returns:
            Token IDs of newly added code points.
        """
        if prototype_vectors.ndim != 2:
            raise ValueError(f'Expected 2D prototype tensor, got {tuple(prototype_vectors.shape)}')
        if len(codepoints) != prototype_vectors.shape[0]:
            raise ValueError(
                f'Number of code points ({len(codepoints)}) does not match '
                f'number of vectors ({prototype_vectors.shape[0]}).'
            )

        new_codepoints = []
        vector_by_codepoint: dict[int, torch.Tensor] = {}
        for codepoint, vector in zip(codepoints, prototype_vectors):
            cp = int(codepoint)
            if self.tokenizer.token_for_codepoint(cp) is not None:
                continue
            new_codepoints.append(cp)
            vector_by_codepoint[cp] = vector

        if not new_codepoints:
            return []

        self.tokenizer.thaw()
        try:
            added_codepoints = self.tokenizer.register_codepoint_values(new_codepoints)
        finally:
            self.tokenizer.freeze()

        if not added_codepoints:
            return []

        ordered_vectors = torch.stack([vector_by_codepoint[cp] for cp in added_codepoints], dim=0)
        self.extend_prototypes(ordered_vectors)
        self.user_metadata['tokenizer'] = self.tokenizer.save()
        return [self.tokenizer.token_for_codepoint(cp) for cp in added_codepoints]

    @torch.no_grad()
    def add_codepoints_from_support(self,
                                    support_codepoints: list[int],
                                    support_vectors: torch.Tensor) -> list[int]:
        """
        Initializes prototypes for unseen code points as support-set means.
        """
        if support_vectors.ndim != 2:
            raise ValueError(f'Expected 2D support tensor, got {tuple(support_vectors.shape)}')
        if len(support_codepoints) != support_vectors.shape[0]:
            raise ValueError(
                f'Number of support labels ({len(support_codepoints)}) does not match '
                f'number of support vectors ({support_vectors.shape[0]}).'
            )

        grouped: dict[int, list[torch.Tensor]] = {}
        for codepoint, vector in zip(support_codepoints, support_vectors):
            grouped.setdefault(int(codepoint), []).append(vector)

        codepoints = list(grouped.keys())
        mean_vectors = torch.stack([torch.stack(vectors, dim=0).mean(dim=0) for vectors in grouped.values()], dim=0)
        return self.add_codepoints_with_prototypes(codepoints, mean_vectors)

    @torch.no_grad()
    def load_pretrained_party_weights(self, path: Union[str, 'PathLike']):
        """
        Loads weights from a "conventional" (non-prototype) party safetensors
        file into this prototype model.

        Token embeddings and output head weights are dropped before loading so
        that the freshly-initialized prototype head and tokenizer-sized
        embedding table are preserved. Encoder, adapter, line embedding, and
        decoder body (self-/cross-attention, norms) weights transfer.

        Handles both raw state-dict-style safetensors files (keys like
        ``nn.encoder.…``) and kraken bundle-format files (keys like
        ``<uuid>.nn.encoder.…``).
        """
        import json
        from safetensors import safe_open

        with safe_open(str(path), framework='pt') as f:
            keys = list(f.keys())
            file_metadata = f.metadata() or {}
            state_dict = {k: f.get_tensor(k) for k in keys}

        # Kraken multi-model bundle: every tensor key is prefixed with a
        # per-model UUID listed in the file-level `kraken_meta` blob. Strip
        # exactly one matching prefix so the keys land in PartyModel's own
        # namespace (`nn.encoder.…` etc.).
        prefix = ''
        kraken_meta_raw = file_metadata.get('kraken_meta')
        if kraken_meta_raw:
            try:
                kraken_meta = json.loads(kraken_meta_raw)
            except Exception:
                kraken_meta = {}
            uuids = [uid for uid in kraken_meta
                     if any(k.startswith(f'{uid}.') for k in keys)]
            if len(uuids) > 1:
                raise ValueError(f'Pretrained safetensors contains multiple models ({uuids}); '
                                 f'cannot pick one automatically.')
            if uuids:
                prefix = f'{uuids[0]}.'

        if prefix:
            state_dict = {k[len(prefix):]: v
                          for k, v in state_dict.items()
                          if k.startswith(prefix)}

        drop_prefixes = ('nn.decoder.tok_embeddings.', 'nn.decoder.output.')
        filtered = {k: v for k, v in state_dict.items()
                    if not any(k.startswith(p) for p in drop_prefixes)}

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        # mask_embedding is the only parameter the diffusion decoder adds, so
        # autoregressive weights never carry it.
        converted = [k for k in missing if k.endswith('mask_embedding')]
        if converted:
            self.nn['decoder'].init_mask_embedding(first_token_id=CODEPOINT_OFFSET)
            logger.info('Converting autoregressive weights to masked diffusion: '
                        'seeded mask_embedding from the token embedding table. '
                        'The model needs fine-tuning under the diffusion '
                        'objective before it will decode in parallel.')

        # The prototype-specific keys (tok_embeddings, output.prototypes,
        # output.temperature) are expected to be missing because we just
        # filtered them out. Anything else is a real concern.
        real_missing = [k for k in missing
                        if not k.startswith(drop_prefixes) and k not in converted]
        if real_missing:
            logger.warning(f'Pretrained party weights missing keys: {real_missing}')
        if unexpected:
            logger.warning(f'Pretrained party weights unexpected keys: {unexpected}')

    def forward_encoder_embeddings(self, encoder_input):
        """
        Computes the encoder embeddings *without* adding the curve positional
        embeddings.
        """
        # timm features_only returns a list; we always request a single stage.
        feature_map, = self.nn['encoder'](encoder_input)
        return self.nn['adapter'](feature_map)

    def prepare_for_generation(self,
                               batch_size: int,
                               *,
                               max_generated_tokens: int = 512,
                               diffusion_steps: int = 4,
                               confidence_threshold: Optional[float] = None,
                               truncate_at_eos: bool = True) -> None:
        """
        Allocates encoder KV caches and primes the model for parallel decoding.

        ``max_generated_tokens`` is the canvas width, terminated by the first
        predicted EOS. Decoding cost scales with it, so it is worth setting from
        the dataset's ``max_tokens_in_line``.
        """
        self.eval()
        decoder_max_seq_len = self.nn['decoder'].max_seq_len
        if max_generated_tokens > decoder_max_seq_len:
            logger.warning(f'max_generated_tokens ({max_generated_tokens}) exceeds decoder '
                           f'max_seq_len ({decoder_max_seq_len}); clamping.')
            max_generated_tokens = decoder_max_seq_len
        self._batch_size = batch_size
        self._max_generated_tokens = max_generated_tokens
        self._diffusion_steps = max(1, int(diffusion_steps))
        self._confidence_threshold = confidence_threshold
        self._truncate_at_eos = bool(truncate_at_eos)
        self.setup_caches(batch_size=batch_size,
                          encoder_max_seq_len=self.encoder_max_seq_len,
                          decoder_max_seq_len=max_generated_tokens,
                          dtype=next(self.parameters()).dtype)

    def prepare_for_inference(self, config: 'Config'):
        """
        Configures the model for inference.
        """
        if self.ready_for_generation:
            logger.debug('Model has already been prepared for generation!')

        if not isinstance(config, PartyRecognitionInferenceConfig):
            upgraded = PartyRecognitionInferenceConfig()
            upgraded.__dict__.update(config.__dict__)
            config = upgraded

        self._inf_config = config
        self.m_dtype = next(self.parameters()).dtype

        self.prepare_for_generation(batch_size=config.batch_size,
                                    max_generated_tokens=config.max_generated_tokens,
                                    diffusion_steps=config.diffusion_steps,
                                    confidence_threshold=config.confidence_threshold,
                                    truncate_at_eos=config.truncate_at_eos)

        self._fabric = Fabric(accelerator=self._inf_config.accelerator,
                              devices=self._inf_config.device,
                              precision=self._inf_config.precision)

        self.nn = self._fabric._precision.convert_module(self.nn)
        self.nn = self._fabric.to_device(self.nn)

        self.im_transforms = get_default_transforms(self.user_metadata['image_size'], dtype=self.m_dtype)

        self.ready_for_generation = True

    @torch.inference_mode()
    def predict(self, im: 'Image.Image', segmentation: 'Segmentation') -> Generator['ocr_record', None, None]:
        with self._fabric.init_tensor():
            image_input = self.im_transforms(im).unsqueeze(0)

            if self._inf_config.prompt_mode == 'curves' and segmentation.type == 'bbox':
                raise ValueError('Prompt mode set to curves but segmentation is of bounding box type.')
            prompt_mode = self._inf_config.prompt_mode or (
                'curves' if segmentation.type == 'baselines' else 'boxes')

            valid_lines = []
            lines_data = []
            for idx, line in enumerate(segmentation.lines):
                line_data = _curve_prompt_fn(line, im.size) if prompt_mode == 'curves' else _box_prompt_fn(line, im.size)
                if line_data is None:
                    logger.info(f'Skipping line {idx} due to invalid prompt geometry.')
                    continue
                valid_lines.append(line)
                lines_data.append(line_data)

            if not lines_data:
                logger.warning('No valid line prompts available for inference.')
                return

            lines = torch.tensor(lines_data).view(-1, 4, 2)

            if self._inf_config.add_lang_token:
                languages = [line.language or segmentation.language or None
                             for line in valid_lines]
            else:
                languages = None

            for (pred_text, pred_confs, pred_langs), line in zip(
                self.predict_string(encoder_input=image_input,
                                    curves=lines if prompt_mode == 'curves' else None,
                                    boxes=lines if prompt_mode == 'boxes' else None,
                                    languages=languages),
                valid_lines
            ):
                line = replace(line, language=list(pred_langs) if pred_langs else None)
                n_chars = len(pred_text)

                if prompt_mode == 'curves':
                    if n_chars > 0 and line.baseline:
                        bl = np.array(line.baseline)
                        seg_lengths = np.sqrt((np.diff(bl, axis=0)**2).sum(axis=1))
                        total_length = seg_lengths.sum()
                        step = total_length / n_chars
                        cuts = [(int(i * step), int((i + 1) * step)) for i in range(n_chars)]
                    else:
                        cuts = []

                    yield BaselineOCRRecord(prediction=pred_text,
                                            cuts=cuts,
                                            confidences=pred_confs,
                                            line=line,
                                            display_order=False)
                else:
                    bbox_line = line if line.type == 'bbox' else line.to_bbox(
                        text_direction=segmentation.text_direction)
                    if n_chars > 0:
                        xmin, ymin, xmax, ymax = bbox_line.bbox
                        step = (xmax - xmin) / n_chars
                        cuts = [((int(xmin + i * step), ymin),
                                 (int(xmin + (i + 1) * step), ymin),
                                 (int(xmin + (i + 1) * step), ymax),
                                 (int(xmin + i * step), ymax))
                                for i in range(n_chars)]
                    else:
                        cuts = []

                    yield BBoxOCRRecord(prediction=pred_text,
                                        cuts=cuts,
                                        confidences=pred_confs,
                                        line=bbox_line,
                                        display_order=False)

    @torch.inference_mode()
    def predict_tokens(self,
                       encoder_input: torch.FloatTensor,
                       curves: Optional[torch.FloatTensor] = None,
                       boxes: Optional[torch.FloatTensor] = None,
                       languages: Optional[list[Optional[list[str]]]] = None) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Predicts text from an input page image and a number of cubic Bézier
        curves or bounding boxes.

        Each line is decoded as a single block: the canvas starts masked and is
        filled over ``diffusion_steps`` passes, each unmasking the most
        confident positions. Every position is predicted in parallel, so the
        number of forward passes is independent of the transcription length.

        Args:
            encoder_input: Image input for the encoder with shape ``1 x c x h x w``
            curves: Curves to be embedded and added to the encoder embeddings (``n x 4 x 2``)
            boxes: Boxes to be embedded and added to the encoder embeddings (``n x 4 x 2``)
            languages: Optional per-line language conditioning. One entry per
                       line in ``curves``/``boxes``; each entry is either a
                       list of ISO-639-3 language identifiers to condition that
                       line on, or ``None`` to let the model predict them. Pass
                       ``None`` (the default) to leave every line
                       unconditioned. Conditioning tokens start the canvas
                       unmasked, so no left-padding is needed and positions are
                       uniform across the batch.

        Yields:
            One tensor of integer labels and one tensor of float confidences,
            each with shape ``n x s``. BOS and the conditioning prefix have
            already been stripped, and entries from EOS onward are zeroed.
        """
        if curves is not None and boxes is not None:
            raise ValueError('`curves` and `boxes` are mutually exclusive.')
        if curves is None and boxes is None:
            raise ValueError('One of `curves` or `boxes` needs to be set.')

        line_pos = curves if curves is not None else boxes
        n_lines = line_pos.size(0)
        if languages is not None and len(languages) != n_lines:
            raise ValueError(f'`languages` must contain one entry per line '
                             f'({n_lines}); got {len(languages)}.')

        logger.debug('Computing encoder embeddings')
        adapter_output = self.forward_encoder_embeddings(encoder_input)
        device = adapter_output.device

        # Conditioning prefix per line: BOS plus any requested language tokens.
        line_prompts = [self.tokenizer.encode('',
                                              langs=languages[i] if languages is not None else None,
                                              add_eos=False)
                        for i in range(n_lines)]

        canvas_len = self._max_generated_tokens
        steps = self._diffusion_steps
        threshold = self._confidence_threshold
        truncate = self._truncate_at_eos

        n_batches = (n_lines + self._batch_size - 1) // self._batch_size
        for batch_idx in range(n_batches):
            start = batch_idx * self._batch_size
            end = min(start + self._batch_size, n_lines)
            bsz = end - start
            batch = line_pos[start:end]
            batch_prompts = line_prompts[start:end]
            max_prompt_len = max(len(p) for p in batch_prompts)

            if max_prompt_len >= canvas_len:
                raise ValueError(f'Conditioning prefix of {max_prompt_len} tokens leaves no '
                                 f'room on a canvas of {canvas_len}.')

            logger.info(f'Processing batch {batch_idx + 1} of {n_batches}')
            if bsz != self._cache_batch_size:
                logger.debug(f'Resizing encoder caches ({self._cache_batch_size} -> {bsz})')
                self.setup_caches(batch_size=bsz,
                                  encoder_max_seq_len=self.encoder_max_seq_len,
                                  decoder_max_seq_len=canvas_len,
                                  dtype=next(self.nn['encoder'].parameters()).dtype)
            self.reset_caches()

            logger.debug('Computing additive line embeddings.')
            line_embeddings = self.nn['line_embedding'](curves=batch if curves is not None else None,
                                                        boxes=batch if boxes is not None else None)
            encoder_hidden_states = adapter_output + line_embeddings.unsqueeze(1)

            # the canvas starts fully masked apart from the prefix, which is
            # left-aligned so RoPE positions are a plain arange for every row
            x = torch.full((bsz, canvas_len), MASK_ID, dtype=torch.long, device=device)
            for i, prompt in enumerate(batch_prompts):
                x[i, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)
            confidences = torch.zeros((bsz, canvas_len), dtype=torch.float, device=device)

            # Live canvas width, shrunk once every row has terminated.
            live = canvas_len
            for step in range(steps):
                masked = x[:, :live] == MASK_ID
                if not masked.any():
                    break

                # encoder keys/values are stable, so they are projected once
                # and read from the cross-attention cache on later passes
                logits = self.forward(tokens=x[:, :live],
                                      encoder_hidden_states=encoder_hidden_states if step == 0 else None,
                                      encoder_mask=None)
                probs = logits.softmax(-1)
                slot_conf, slot_tok = probs.max(-1)

                # slot i scores position i+1, so shift right to index by the
                # position decided. The last slot falls off the canvas.
                pos_conf = torch.cat([slot_conf.new_zeros(bsz, 1), slot_conf[:, :-1]], dim=1)
                pos_tok = torch.cat([slot_tok.new_zeros(bsz, 1), slot_tok[:, :-1]], dim=1)

                # only still-masked positions are candidates: a committed one
                # sees its own value and reports inflated confidence
                score = pos_conf.masked_fill(~masked, -1.0)

                # unmask the most confident 1/(remaining passes) of what is
                # left, filling the canvas in exactly `steps` passes
                remaining = steps - step
                quota = torch.ceil(masked.sum(-1) / remaining).long().clamp_min(1)
                order = score.argsort(dim=-1, descending=True)
                rank = torch.empty_like(order)
                rank.scatter_(1, order, torch.arange(live, device=device).expand(bsz, live))
                take = masked & (rank < quota.unsqueeze(1))
                if threshold is not None:
                    take |= masked & (pos_conf >= threshold)

                x[:, :live] = torch.where(take, pos_tok, x[:, :live])
                confidences[:, :live] = torch.where(take, pos_conf, confidences[:, :live])

                # everything after a committed EOS is PAD by construction;
                # filling it here drops it out of later passes
                eos_at = (x[:, :live] == self.tokenizer.eos_id)
                if eos_at.any():
                    after_eos = eos_at.cumsum(-1) > eos_at.long()
                    x[:, :live] = torch.where(after_eos & (x[:, :live] == MASK_ID),
                                              torch.full_like(x[:, :live], self.tokenizer.pad_id),
                                              x[:, :live])
                    if truncate and eos_at.any(-1).all():
                        # every row terminated, so drop the dead tail. Slicing
                        # from the left keeps RoPE positions correct.
                        live = int(eos_at.float().argmax(-1).max().item()) + 1
                        logger.debug(f'Truncating canvas to {live} after EOS.')

            # strip the prefix and zero from EOS on, as the autoregressive
            # decoder yielded. Predicted language tokens are kept, as
            # decode_with_confs reads the language tags back out of them.
            keep = torch.ones_like(x, dtype=torch.bool)
            for i, prompt in enumerate(batch_prompts):
                keep[i, :len(prompt)] = False
            eos_at = x == self.tokenizer.eos_id
            live_mask = ~(eos_at.cumsum(-1) > 0)

            width = canvas_len - min(len(p) for p in batch_prompts)
            tokens_out = torch.zeros((bsz, width), dtype=torch.long, device=device)
            confs_out = torch.zeros((bsz, width), dtype=torch.float, device=device)
            for i in range(bsz):
                row = (keep[i] & live_mask[i]).nonzero(as_tuple=True)[0]
                tokens_out[i, :row.numel()] = x[i, row]
                confs_out[i, :row.numel()] = confidences[i, row]
            yield tokens_out, confs_out

    @torch.inference_mode()
    def predict_string(self,
                       encoder_input: torch.FloatTensor,
                       curves: Optional[torch.FloatTensor] = None,
                       boxes: Optional[torch.FloatTensor] = None,
                       languages: Optional[list[Optional[list[str]]]] = None) -> Generator[str, None, None]:
        """
        Predicts text from an input page image and a number of cubic Bézier
        curves.

        Args:
            encoder_input: Image input for the encoder with shape ``[1 x c x h x w]``
            curves: Curves to be embedded and added to the encoder embeddings (``n x 4 x 2``)
            languages: Optional per-line language conditioning. One entry per
                       input line; each entry is a list of ISO-639-3 language
                       identifiers for that line, or ``None`` to leave it
                       unconditioned. Pass ``None`` (default) to disable
                       language conditioning for all lines.

        Yields:
        """
        for preds, confs in self.predict_tokens(encoder_input=encoder_input,
                                                curves=curves,
                                                boxes=boxes,
                                                languages=languages):
            for pred, conf in zip(preds, confs):
                yield self.tokenizer.decode_with_confs(pred[pred != 0], conf)
