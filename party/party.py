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
"""Additive prompt recognition model."""
import logging
import timm
import torch
import numpy as np

from torch import nn

from dataclasses import asdict, replace
from kraken.models import RecognitionBaseModel
from kraken.containers import BaselineOCRRecord, BBoxOCRRecord, BBoxLine
from lightning.fabric import Fabric
from collections.abc import Generator
from typing import Optional, Union, TYPE_CHECKING, Any

from party.modules import PromptEncoder
from party.tokenizer import OctetTokenizer
from party.fusion import (
    bytellama_vision_decoder,
    default_adapter_ds_factors,
    PartyMultiScaleAdapter,
    SingleScaleAdapter,
)
from party.dataset import get_default_transforms, _to_curve, _to_bbox
from party.configs import (
    DEFAULT_DECODER_EMBED_DIM,
    DEFAULT_DECODER_NAME,
    DEFAULT_ENCODER_NAME,
    DEFAULT_FUSION_INTERVAL,
)

if TYPE_CHECKING:
    from PIL import Image
    from kraken.configs import Config
    from kraken.containers import Segmentation, ocr_record, BaselineLine

logger = logging.getLogger(__name__)

__all__ = ['PartyModel']


def _box_prompt_fn(
    line: Union['BaselineLine', 'BBoxLine'],
    im_size: tuple[int, int],
) -> Optional[list[float]]:
    bbox = _to_bbox([line.bbox] if line.type == 'bbox' else line.boundary, im_size)
    return bbox.as_py() if bbox is not None else None


def _curve_prompt_fn(line: 'BaselineLine', im_size: tuple[int, int]) -> Optional[list[float]]:
    curve = _to_curve(line.baseline, im_size)
    return curve.as_py() if curve is not None else None


def _baseline_to_bbox(line: 'BaselineLine') -> 'BBoxLine':
    d = asdict(line)
    d.pop('baseline')
    d.pop('type')
    flat_box = [point for pol in d.pop('boundary') for point in pol]
    xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
    ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
    d['bbox'] = (xmin, ymin, xmax, ymax)
    return BBoxLine(**d)


def _build_adapter(
    encoder_out_indices: tuple[int, ...],
    adapter_num_layers: int,
    adapter_num_heads: int,
    adapter_ds_factors: list[int],
    encoder_channels: list[int],
    encoder_sizes: list[tuple[int, int]],
) -> tuple[nn.Module, int]:
    if len(encoder_out_indices) == 1:
        if adapter_ds_factors != [1]:
            raise ValueError('single-scale encoder features require adapter_ds_factors=[1].')
        return (
            SingleScaleAdapter(
                num_layers=adapter_num_layers,
                num_heads=adapter_num_heads,
                encoder_embed_dim=encoder_channels[0],
                decoder_embed_dim=DEFAULT_DECODER_EMBED_DIM,
            ),
            encoder_sizes[0][0] * encoder_sizes[0][1],
        )

    if len(adapter_ds_factors) != len(encoder_out_indices):
        raise ValueError('adapter_ds_factors must have the same length as encoder_out_indices.')

    adapter = PartyMultiScaleAdapter(
        num_layers=adapter_num_layers,
        num_heads=adapter_num_heads,
        encoder_embed_dims=encoder_channels,
        encoder_sizes=encoder_sizes,
        decoder_embed_dim=DEFAULT_DECODER_EMBED_DIM,
        ds_factors=adapter_ds_factors,
    )
    encoder_max_seq_len = sum(
        (size[0] // ds_factor) * (size[1] // ds_factor)
        for size, ds_factor in zip(encoder_sizes, adapter_ds_factors)
    )
    return adapter, encoder_max_seq_len


class PartyModel(nn.Module, RecognitionBaseModel):
    """
    The additive prompt party model.
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

        self.user_metadata: dict[str, Any] = {'accuracy': [], 'metrics': []}
        self.user_metadata.update(kwargs)

        out_indices = tuple(kwargs.get('encoder_out_indices', (2,)))
        adapter_ds_factors = list(kwargs.get('adapter_ds_factors', default_adapter_ds_factors(len(out_indices))))
        adapter_num_layers = kwargs.get('adapter_num_layers', 4 if len(out_indices) == 1 else 1)
        adapter_num_heads = kwargs.get('adapter_num_heads', 8)
        self.user_metadata.update(
            {
                'image_size': image_size,
                'encoder_out_indices': out_indices,
                'adapter_num_layers': adapter_num_layers,
                'adapter_num_heads': adapter_num_heads,
                'adapter_ds_factors': adapter_ds_factors,
            }
        )

        encoder = timm.create_model(
            DEFAULT_ENCODER_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        encoder_embed_dims = encoder.feature_info.channels()
        encoder_reductions = encoder.feature_info.reduction()
        encoder_sizes = [
            (image_size[0] // red, image_size[1] // red)
            for red in encoder_reductions
        ]

        adapter, self.encoder_max_seq_len = _build_adapter(
            encoder_out_indices=out_indices,
            adapter_num_layers=adapter_num_layers,
            adapter_num_heads=adapter_num_heads,
            adapter_ds_factors=adapter_ds_factors,
            encoder_channels=encoder_embed_dims,
            encoder_sizes=encoder_sizes,
        )

        decoder = bytellama_vision_decoder(
            pretrained=DEFAULT_DECODER_NAME,
            encoder_max_seq_len=self.encoder_max_seq_len,
            fusion_interval=DEFAULT_FUSION_INTERVAL,
        )
        decoder_embed_dim = decoder.tok_embeddings.embedding_dim
        if decoder_embed_dim != DEFAULT_DECODER_EMBED_DIM:
            raise ValueError(
                f'Unexpected decoder embedding dimension {decoder_embed_dim}; '
                f'expected {DEFAULT_DECODER_EMBED_DIM}.'
            )

        self.nn = nn.ModuleDict(
            {
                'encoder': encoder,
                'decoder': decoder,
                'adapter': adapter,
                'line_embedding': PromptEncoder(embed_dim=decoder_embed_dim),
            }
        )

        self.ready_for_generation = False

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int = None,
        decoder_max_seq_len: int = None,
    ):
        self.nn['decoder'].setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )

    def caches_are_setup(self) -> bool:
        return self.nn['decoder'].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        return self.nn['decoder'].caches_are_enabled()

    def reset_caches(self):
        self.nn['decoder'].reset_caches()

    def forward_encoder_embeddings(self, encoder_input: torch.Tensor) -> torch.Tensor:
        return self.nn['adapter'](self.nn['encoder'](encoder_input))

    def forward_line_features(
        self,
        encoder_hidden_states: torch.Tensor,
        curves: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if curves is not None and boxes is not None:
            raise ValueError('curves and boxes are mutually exclusive.')
        if curves is None and boxes is None:
            raise ValueError('One of curves or boxes must be provided.')

        line_embeddings = self.nn['line_embedding'](curves=curves, boxes=boxes)
        batch_size = line_embeddings.shape[0]

        if encoder_hidden_states.shape[0] == 1 and batch_size != 1:
            encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)
        elif encoder_hidden_states.shape[0] != batch_size:
            raise ValueError(
                f'encoder_hidden_states batch size ({encoder_hidden_states.shape[0]}) '
                f'does not match prompt batch size ({batch_size}).'
            )

        return encoder_hidden_states + line_embeddings.unsqueeze(1)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_curves: Optional[torch.Tensor] = None,
        encoder_boxes: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        if encoder_hidden_states is None and encoder_input is not None:
            encoder_hidden_states = self.forward_encoder_embeddings(encoder_input)
            encoder_hidden_states = self.forward_line_features(
                encoder_hidden_states,
                curves=encoder_curves,
                boxes=encoder_boxes,
            )

        return self.nn['decoder'](
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_hidden_states,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )

    def prepare_for_inference(self, config: 'Config'):
        if self.ready_for_generation:
            logger.debug('Model has already been prepared for generation!')

        self.eval()
        self._inf_config = config

        from torch.multiprocessing import Pool
        if getattr(self, '_line_extraction_pool', None) is None:
            self._line_extraction_pool = Pool(self._inf_config.num_line_workers)
            import atexit
            atexit.register(self._line_extraction_pool.terminate)

        self.m_dtype = next(self.parameters()).dtype
        self._batch_size = config.batch_size
        self._max_generated_tokens = config.max_generated_tokens

        self.setup_caches(
            batch_size=config.batch_size,
            encoder_max_seq_len=self.encoder_max_seq_len,
            decoder_max_seq_len=config.max_generated_tokens,
            dtype=self.m_dtype,
        )

        self._fabric = Fabric(
            accelerator=self._inf_config.accelerator,
            devices=self._inf_config.device,
            precision=self._inf_config.precision,
        )

        self.nn = self._fabric._precision.convert_module(self.nn)
        self.nn = self._fabric.to_device(self.nn)

        self.im_transforms = get_default_transforms(self.user_metadata['image_size'], dtype=self.m_dtype)

        self.ready_for_generation = True

    @torch.inference_mode()
    def predict(
        self,
        im: 'Image.Image',
        segmentation: 'Segmentation',
    ) -> Generator['ocr_record', None, None]:
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
                self.predict_string(
                    encoder_input=image_input,
                    curves=lines if prompt_mode == 'curves' else None,
                    boxes=lines if prompt_mode == 'boxes' else None,
                    languages=languages,
                ),
                valid_lines,
            ):
                line = replace(line, language=list(pred_langs) if pred_langs else None)
                n_chars = len(pred_text)

                if prompt_mode == 'curves':
                    if n_chars > 0 and line.baseline:
                        bl = np.array(line.baseline)
                        seg_lengths = np.sqrt((np.diff(bl, axis=0) ** 2).sum(axis=1))
                        total_length = seg_lengths.sum()
                        step = total_length / n_chars
                        cuts = [(int(i * step), int((i + 1) * step)) for i in range(n_chars)]
                    else:
                        cuts = []

                    yield BaselineOCRRecord(
                        prediction=pred_text,
                        cuts=cuts,
                        confidences=pred_confs,
                        line=line,
                        display_order=False,
                    )
                else:
                    if n_chars > 0:
                        if line.type == 'bbox':
                            xmin, ymin, xmax, ymax = line.bbox
                        else:
                            flat_box = [point for pol in line.boundary for point in pol]
                            xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
                            ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
                        step = (xmax - xmin) / n_chars
                        cuts = [
                            (
                                (int(xmin + i * step), ymin),
                                (int(xmin + (i + 1) * step), ymin),
                                (int(xmin + (i + 1) * step), ymax),
                                (int(xmin + i * step), ymax),
                            )
                            for i in range(n_chars)
                        ]
                    else:
                        cuts = []

                    yield BBoxOCRRecord(
                        prediction=pred_text,
                        cuts=cuts,
                        confidences=pred_confs,
                        line=_baseline_to_bbox(line) if line.type != 'bbox' else line,
                        display_order=False,
                    )

    @torch.inference_mode()
    def predict_tokens(
        self,
        encoder_input: torch.FloatTensor,
        curves: Optional[torch.FloatTensor] = None,
        boxes: Optional[torch.FloatTensor] = None,
        languages: Optional[list[str]] = None,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        if curves is not None and boxes is not None:
            raise ValueError('`curves` and `boxes` are mutually exclusive.')
        if curves is None and boxes is None:
            raise ValueError('One of `curves` or `boxes` needs to be set.')
        logger.debug('Computing encoder embeddings')

        adapter_output = self.forward_encoder_embeddings(encoder_input)
        device = adapter_output.device

        prompt = torch.tensor(
            self.tokenizer.encode('', langs=languages, add_eos=False),
            device=device,
            dtype=torch.long,
        ).repeat(self._batch_size, 1)
        prompt_length = prompt.size(1)

        eos_token = torch.tensor(self.tokenizer.eos_id, device=device, dtype=torch.long)

        line_pos = curves if curves is not None else boxes
        batches = torch.split(line_pos, self._batch_size)

        total_response_length = prompt_length + self._max_generated_tokens
        masks = torch.tril(
            torch.ones(
                total_response_length,
                self._max_generated_tokens,
                dtype=torch.bool,
                device=device,
            )
        ).unsqueeze(0)
        input_pos = torch.arange(0, total_response_length, device=device).unsqueeze(0)

        encoder_mask = torch.ones(
            (self._batch_size, prompt_length, self.encoder_max_seq_len),
            dtype=torch.bool,
            device=device,
        )

        for batch_idx, batch in enumerate(batches):
            bsz = batch.size(0)

            self.reset_caches()

            logger.info(f'Processing batch {batch_idx} of {len(batches)}')
            if bsz != self._batch_size:
                logger.debug(f'Resizing caches for last batch ({self._batch_size} -> {bsz})')
                self.setup_caches(
                    batch_size=bsz,
                    encoder_max_seq_len=self.encoder_max_seq_len,
                    decoder_max_seq_len=self._max_generated_tokens,
                    dtype=next(self.nn['encoder'].parameters()).dtype,
                )

            logger.debug('Computing additive line prompt features.')
            line_features = self.forward_line_features(
                adapter_output,
                curves=batch if curves is not None else None,
                boxes=batch if boxes is not None else None,
            )

            logger.debug('Prefilling cache.')
            curr_masks = masks[:, :prompt_length]
            logits = self.forward(
                tokens=prompt[:bsz, ...],
                encoder_hidden_states=line_features,
                encoder_mask=encoder_mask[:bsz, ...],
                mask=curr_masks,
                input_pos=input_pos[:, :prompt_length].squeeze(),
            )
            tokens = torch.argmax(logits, dim=-1)[:, -1:]
            confs = logits[:, -1].softmax(-1)
            generated_tokens = [tokens[:, -1]]
            generated_confidences = [confs.gather(-1, tokens).squeeze(1)]
            curr_pos = prompt_length

            eos_token_reached = torch.zeros(bsz, dtype=torch.bool, device=device)
            eos_token_reached |= tokens[:, -1] == eos_token
            eos_token_mask = torch.ones(bsz, 0, dtype=torch.int32, device=device)

            if eos_token_reached.all():
                break

            for _ in range(self._max_generated_tokens - (1 + prompt_length)):
                eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(bsz, 1)], dim=-1)

                curr_input_pos = input_pos[:, curr_pos]
                curr_masks = masks[:, curr_pos, None, :]

                logits = self.forward(
                    tokens=tokens.clone(),
                    mask=curr_masks,
                    input_pos=curr_input_pos,
                )
                tokens = torch.argmax(logits, dim=-1)
                confs = logits[:, -1].softmax(-1)
                generated_tokens.append(tokens[:, -1])
                generated_confidences.append(confs.gather(-1, tokens).squeeze(1))

                curr_pos += 1

                eos_token_reached |= tokens[:, -1] == eos_token
                if eos_token_reached.all():
                    break

            eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(bsz, 1)], dim=-1)

            generated_tokens = torch.stack(generated_tokens).T
            generated_confidences = torch.stack(generated_confidences).T

            generated_tokens *= eos_token_mask
            generated_confidences *= eos_token_mask
            yield generated_tokens[..., :-1], generated_confidences[..., :-1]

    @torch.inference_mode()
    def predict_string(
        self,
        encoder_input: torch.FloatTensor,
        curves: Optional[torch.FloatTensor] = None,
        boxes: Optional[torch.FloatTensor] = None,
        languages: Optional[list[str]] = None,
    ) -> Generator[str, None, None]:
        for preds, confs in self.predict_tokens(
            encoder_input=encoder_input,
            curves=curves,
            boxes=boxes,
            languages=languages,
        ):
            for pred, conf in zip(preds, confs):
                yield self.tokenizer.decode_with_confs(pred[pred != 0], conf)
