#
# Copyright 2024 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
party.pred
~~~~~~~~~~

API for inference
"""
import torch
import logging

from party.dataset import get_default_transforms, _to_curve, _to_bbox

from typing import TYPE_CHECKING, Union, Tuple, Optional, Literal, Generator

logging.captureWarnings(True)
logger = logging.getLogger('party')


if TYPE_CHECKING:
    from party.fusion import PartyModel
    from PIL import Image
    from kraken.containers import Segmentation, BaselineLine, BBoxLine, ocr_record
    from lightning.fabric import Fabric

__all__ = ['batched_pred']


def _box_prompt_fn(line: Union['BaselineLine', 'BBoxLine'], im_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts a BBoxLine or BaselineLine to a bounding box representation.
    """
    return _to_bbox([line.bbox] if line.type == 'bbox' else line.boundary, im_size).as_py()


def _curve_prompt_fn(line: 'BaselineLine', im_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts a BaselineLine to a quadratic Bézier curve.
    """
    return _to_curve(line.baseline, im_size).as_py()


class batched_pred(object):
    """
    Batched single-model prediction with a generative model.

    Args:
        model: PartyModel for generation.
        im: Pillow image
        bounds: Segmentation for input image
        fabric: Fabric context manager to cast models and tensors.
        prompt_mode: How to embed line positional prompts. Per default prompts
                     are determined by the segmentation type if the model
                     indicates either curves or boxes are supported. If the
                     model supports only boxes and the input segmentation is
                     baseline-type, bounding boxes will be generated from the
                     bounding polygon if available. If the model expects curves
                     and the segmentation is of bounding box-type an exception
                     will be raised. If explicit values are set.
        batch_size: Number of lines to predict in parallel

    Yields:
        An ocr_record containing the recognized text, dummy character
        positions, and confidence values for each character.

    Raises:
        ValueError when the model expects curves and the segmentation of bounding box-type.
    """
    def __init__(self,
                 model: 'PartyModel',
                 im: 'Image.Image',
                 bounds: 'Segmentation',
                 fabric: 'Fabric',
                 prompt_mode: Optional[Literal['curves', 'boxes']] = None,
                 batch_size: int = 2) -> Generator['ocr_record', None, None]:
        m_prompt_mode = model.line_prompt_mode
        s_prompt_mode = bounds.type

        line_prompt_fn = None
        if m_prompt_mode == 'curves' and s_prompt_mode == 'bbox':
            raise ValueError(f'Model expects curves and segmentation is bbox-type. Aborting.')
        if prompt_mode is None:
            if m_prompt_mode == 'boxes' and s_prompt_mode == 'baseline':
                logger.info('Model expect boxes and segmentation is baseline-type. Casting bounding polygons to bounding boxes.')
                line_prompt_fn = _box_prompt_fn
                prompt_mode = 'boxes'
            elif m_prompt_mode == 'both':
                line_prompt_fn = _box_prompt_fn if s_prompt_mode == 'bbox' else _curve_prompt_fn
                prompt_mode = 'boxes' if s_prompt_mode == 'bbox' else 'curves'
            else:
                line_prompt_fn = _box_prompt_fn if m_prompt_mode == 'boxes' else _curve_prompt_fn
                prompt_mode = 'boxes' if m_prompt_mode == 'boxes' else 'curves'
        elif m_prompt_mode != 'both' or m_prompt_mode != prompt_mode:
            raise ValueError(f'Model expects prompt {m_prompt_mode} and explicit line prompt mode {prompt_mode} selected.')
        else:
            line_prompt_fn = _box_prompt_fn if prompt_mode == 'boxes' else _curve_prompt_fn

        # load image transforms
        im_transforms = get_default_transforms()

        # prepare model for generation
        model.prepare_for_generation(batch_size=batch_size)
        model = model.eval()

        with fabric.init_tensor(), torch.inference_mode():
            image_input = im_transforms(im).unsqueeze(0)
            lines = torch.tensor([line_prompt_fn(line, im.size) for line in bounds.lines])
            lines = lines.view(-1, 4, 2)
            self.len = len(lines)

            self._pred = model.predict_string(encoder_input=image_input,
                                              curves=lines if prompt_mode == 'curves' else None,
                                              boxes=lines if prompt_mode == 'boxes' else None)

    def __next__(self):
        return next(self._pred)

    def __iter__(self):
        return self

    def __len__(self):
        return self.len
