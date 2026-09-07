#
# Copyright 2026 Benjamin Kiessling
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
Masked diffusion objective over the code point canvas.
"""
import torch

from typing import Optional

from party.tokenizer import (BOS_ID, CODEPOINT_OFFSET, EOS_ID, LANG_OFFSET,
                             MASK_ID, PAD_ID)

__all__ = ['shift', 'prompt_mask', 'to_canvas', 'apply_absorbing_mask',
           'position_weights', 'causal_window_mask', 'sample_mask_rate',
           'sample_lang_prompt', 'content_mask']


def shift(tokens: torch.Tensor, fill) -> torch.Tensor:
    """Left-shifts a [b, s] tensor, filling the last column with ``fill``."""
    fill_col = torch.full((tokens.shape[0], 1),
                          fill,
                          dtype=tokens.dtype, device=tokens.device)
    return torch.hstack((tokens[..., 1:], fill_col))


def to_canvas(tokens: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """
    Turns the dataset's ignored padding tail into PAD content. Under a full
    canvas the tail is predicted, not ignored: the first EOS is what determines
    the length at inference.
    """
    return torch.where(tokens == ignore_index,
                       torch.full_like(tokens, PAD_ID),
                       tokens)


def prompt_mask(canvas: torch.Tensor,
                keep_langs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Marks conditioning positions, i.e. BOS and the language tokens after it.

    ``keep_langs`` is a per-sample flag selecting whether that row's language
    tokens condition (True) or are masked and predicted (False). Both modes have
    to occur in training or language prediction is lost. None keeps all of them,
    as at inference when languages are supplied.
    """
    is_bos = canvas == BOS_ID
    is_lang = (canvas >= LANG_OFFSET) & (canvas < CODEPOINT_OFFSET)
    if keep_langs is not None:
        is_lang = is_lang & keep_langs.to(device=canvas.device).view(-1, 1)
    return is_bos | is_lang


def content_mask(targets: torch.Tensor) -> torch.Tensor:
    """Marks targets carrying transcription signal, i.e. code points and EOS."""
    return (targets >= CODEPOINT_OFFSET) | (targets == EOS_ID)


def sample_mask_rate(batch_size: int,
                     seq_len: int,
                     *,
                     low: float = 0.0,
                     high: float = 1.0,
                     stratified: bool = True,
                     generator: Optional[torch.Generator] = None,
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Draws one mask rate per sample from ``U(low, high)``.

    Stratification draws one rate per stratum instead of letting the whole batch
    land on a similar noise level, which is the main source of gradient noise
    here. The lower bound is floored at ``1/seq_len`` so at least one position is
    expected to be masked.
    """
    low = max(low, 1.0 / max(seq_len, 1))
    high = max(high, low)
    u = torch.rand(batch_size, generator=generator, device=device)
    if stratified:
        offsets = torch.arange(batch_size, device=device, dtype=u.dtype)
        u = (offsets + u) / batch_size
    return low + (high - low) * u


def sample_lang_prompt(batch_size: int,
                       p: float,
                       generator: Optional[torch.Generator] = None,
                       device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """
    Draws the per-sample language conditioning mode for :func:`prompt_mask`.
    Returns None at ``p >= 1``, which lets ``prompt_mask`` skip the work.
    """
    if p >= 1.0:
        return None
    return torch.rand(batch_size, generator=generator, device=device) < p


def apply_absorbing_mask(canvas: torch.Tensor,
                         t: torch.Tensor,
                         prompt: torch.Tensor,
                         generator: Optional[torch.Generator] = None
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replaces each maskable position with ``MASK_ID`` with probability ``t``,
    returning the noised canvas and the noise mask.

    ``t`` is a per-sample tensor (not a float) to keep it out of the compiled
    graph as a constant, as with the ArcFace margin.
    """
    prob = torch.rand(canvas.shape,
                      generator=generator,
                      device=canvas.device,
                      dtype=torch.float)
    t = t.to(dtype=prob.dtype, device=prob.device)
    m = (prob < t.unsqueeze(1)) & ~prompt
    return torch.where(m, torch.full_like(canvas, MASK_ID), canvas), m


def position_weights(targets: torch.Tensor,
                     pad_loss_weight: torch.Tensor,
                     ignore_index: int) -> torch.Tensor:
    """
    Per-position class weights. Code points and EOS carry full weight; the PAD
    tail is downweighted, as it is most of a full canvas and would otherwise
    dominate the objective. EOS stays at full weight as the length signal.
    """
    weights = torch.ones_like(targets, dtype=torch.float)
    pad_loss_weight = pad_loss_weight.to(dtype=weights.dtype, device=weights.device)
    weights = torch.where(targets == PAD_ID, pad_loss_weight, weights)
    return torch.where(targets == ignore_index, torch.zeros_like(weights), weights)


def causal_window_mask(seq_len: int,
                       window: int,
                       device: torch.device) -> torch.Tensor:
    """
    Attention mask letting each query see ``window`` positions to its right, for
    annealing the decoder from causal (0) to bidirectional. Shape is [1, s, s],
    which broadcasts over the batch.

    Callers should pass ``mask=None`` rather than a full-True mask: an explicit
    boolean mask drops SDPA off the flash attention path.
    """
    q_pos = torch.arange(seq_len, device=device).view(-1, 1)
    k_pos = torch.arange(seq_len, device=device).view(1, -1)
    return (k_pos <= q_pos + window).unsqueeze(0)
