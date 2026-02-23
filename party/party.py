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

from dataclasses import asdict, replace
from kraken.models import RecognitionBaseModel
from kraken.containers import BaselineOCRRecord, BBoxOCRRecord, BBoxLine
from lightning.fabric import Fabric
from collections.abc import Generator
from typing import Optional, Union, TYPE_CHECKING, Any

from party.modules import PromptCrossAttention
from party.tokenizer import OctetTokenizer
from party.fusion import PartyMultiScaleAdapter, bytellama_vision_decoder
from party.dataset import get_default_transforms, _to_curve, _to_bbox

if TYPE_CHECKING:
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
    bbox = _to_bbox([line.bbox] if line.type == 'bbox' else line.boundary, im_size)
    return bbox.as_py() if bbox is not None else None


def _curve_prompt_fn(line: 'BaselineLine',
                     im_size: tuple[int, int]) -> Optional[list[float]]:
    """
    Converts a BaselineLine to a cubic Bézier curve.
    """
    curve = _to_curve(line.baseline, im_size)
    return curve.as_py() if curve is not None else None


def _baseline_to_bbox(line: 'BaselineLine') -> 'BBoxLine':
    """
    Converts a BaselineLine to a BBoxLine.
    """
    d = asdict(line)
    d.pop('baseline')
    d.pop('type')
    flat_box = [point for pol in d.pop('boundary') for point in pol]
    xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
    ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
    d['bbox'] = (xmin, ymin, xmax, ymax)
    return BBoxLine(**d)


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
    tokenizer = OctetTokenizer()

    def __init__(self, **kwargs):
        super().__init__()

        if (pretrained := kwargs.get('pretrained', None)) is None:
            raise ValueError('pretrained argument is missing in args.')
        if (image_size := kwargs.get('image_size', None)) is None:
            raise ValueError('image_size argument is missing in args.')

        self.user_metadata: dict[str, Any] = {'accuracy': [],
                                              'metrics': []}
        self.user_metadata.update(kwargs)

        out_indices = (1, 2, 3)
        ds_factors = [4, 2, 1]
        fusion_interval = kwargs.get('fusion_interval', 3)

        encoder = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k',
                                    pretrained=pretrained,
                                    features_only=True,
                                    out_indices=out_indices)

        encoder_embed_dims = encoder.feature_info.channels()
        encoder_reductions = encoder.feature_info.reduction()
        encoder_sizes = [(image_size[0] // red, image_size[1] // red)
                         for red in encoder_reductions]

        prompt_num_samples = kwargs.get('prompt_num_samples', 384)
        prompt_num_layers = kwargs.get('prompt_num_layers', 2)
        prompt_num_heads = kwargs.get('prompt_num_heads', 8)

        # decoder cross-attention cache length equals the filtered prompt tokens
        self.encoder_max_seq_len = prompt_num_samples

        decoder = bytellama_vision_decoder(pretrained='mittagessen/bytellama-40m-oscar',
                                           encoder_max_seq_len=self.encoder_max_seq_len,
                                           fusion_interval=fusion_interval)
        decoder_embed_dim = decoder.tok_embeddings.embedding_dim

        adapter = PartyMultiScaleAdapter(num_layers=1,
                                         num_heads=8,
                                         encoder_embed_dims=encoder_embed_dims,
                                         encoder_sizes=encoder_sizes,
                                         decoder_embed_dim=decoder_embed_dim,
                                         ds_factors=ds_factors)
        line_embedding = PromptCrossAttention(embed_dim=decoder_embed_dim,
                                              num_heads=prompt_num_heads,
                                              num_layers=prompt_num_layers,
                                              num_samples=prompt_num_samples)

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
        self.nn['decoder'].setup_caches(batch_size,
                                        dtype,
                                        encoder_max_seq_len=encoder_max_seq_len,
                                        decoder_max_seq_len=decoder_max_seq_len)

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
            # Prompt cross-attention conditioning: sampled curve/box points
            # attend into adapter tokens and produce compact line-focused features.
            encoder_hidden_states = self.nn['line_embedding'](encoder_features=adapter_output,
                                                              curves=encoder_curves,
                                                              boxes=encoder_boxes)

        output = self.nn['decoder'](tokens=tokens,
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
        encoder_hidden_states = self.nn['encoder'](encoder_input)
        return self.nn['adapter'](encoder_hidden_states)

    def prepare_for_inference(self, config: 'Config'):
        """
        Configures the model for inference.
        """
        if self.ready_for_generation:
            logger.debug('Model has already been prepared for generation!')

        self.eval()
        self._inf_config = config

        # create line extraction worker pool
        from torch.multiprocessing import Pool
        if getattr(self, '_image_extraction_pool', None) is None:
            self._line_extraction_pool = Pool(self._inf_config.num_line_workers)
            import atexit
            atexit.register(self._line_extraction_pool.terminate)

        self.m_dtype = next(self.parameters()).dtype
        self._batch_size = config.batch_size
        self._max_generated_tokens = config.max_generated_tokens

        # set up caches
        self.setup_caches(batch_size=config.batch_size,
                          encoder_max_seq_len=self.encoder_max_seq_len,
                          decoder_max_seq_len=config.max_generated_tokens,
                          dtype=self.m_dtype)

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

            cfg_prompt_mode = self._inf_config.prompt_mode
            if cfg_prompt_mode is not None:
                if cfg_prompt_mode == 'curves' and segmentation.type == 'bbox':
                    raise ValueError('Prompt mode set to curves but segmentation is of bounding box type.')
                prompt_mode = cfg_prompt_mode
            elif segmentation.type == 'baselines':
                prompt_mode = 'curves'
            else:
                prompt_mode = 'boxes'

            valid_lines = []
            valid_line_indices = []
            lines_data = []
            for idx, line in enumerate(segmentation.lines):
                line_data = _curve_prompt_fn(line, im.size) if prompt_mode == 'curves' else _box_prompt_fn(line, im.size)
                if line_data is None:
                    logger.info(f'Skipping line {idx} due to invalid prompt geometry.')
                    continue
                valid_lines.append(line)
                valid_line_indices.append(idx)
                lines_data.append(line_data)

            if not lines_data:
                logger.warning('No valid line prompts available for inference.')
                return

            lines = torch.tensor(lines_data).view(-1, 4, 2)

            languages = segmentation.language if self._inf_config.add_lang_token else None
            if isinstance(languages, (list, tuple)) and len(languages) == len(segmentation.lines):
                languages = [languages[idx] for idx in valid_line_indices]

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
                    if n_chars > 0:
                        if line.type == 'bbox':
                            xmin, ymin, xmax, ymax = line.bbox
                        else:
                            flat_box = [point for pol in line.boundary for point in pol]
                            xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
                            ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
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
                                        line=_baseline_to_bbox(line) if line.type != 'bbox' else line,
                                        display_order=False)

    @torch.inference_mode()
    def predict_tokens(self,
                       encoder_input: torch.FloatTensor,
                       curves: Optional[torch.FloatTensor] = None,
                       boxes: Optional[torch.FloatTensor] = None,
                       languages: Optional[list[str]] = None) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
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

        adapter_output = self.forward_encoder_embeddings(encoder_input)
        device = adapter_output.device

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
                                   self.encoder_max_seq_len),
                                  dtype=torch.bool,
                                  device=device)

        for batch_idx, batch in enumerate(batches):

            bsz = batch.size(0)

            self.reset_caches()

            logger.info(f'Processing batch {batch_idx} of {len(batches)}')
            if bsz != self._batch_size:
                logger.debug(f'Resizing caches for last batch ({self._batch_size} -> {bsz})')
                self.setup_caches(batch_size=bsz,
                                  encoder_max_seq_len=self.encoder_max_seq_len,
                                  decoder_max_seq_len=self._max_generated_tokens,
                                  dtype=next(self.nn['encoder'].parameters()).dtype)

            logger.debug('Computing line-focused features via prompt cross-attention.')
            line_features = self.nn['line_embedding'](encoder_features=adapter_output,
                                                      curves=batch if curves is not None else None,
                                                      boxes=batch if boxes is not None else None)

            logger.debug('Prefilling cache.')
            # prefill step
            curr_masks = masks[:, :_prompt_length]
            logits = self.forward(tokens=_prompt[:bsz, ...],
                                  encoder_hidden_states=line_features,
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
                       languages: Optional[list[str]] = None) -> Generator[str, None, None]:
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
