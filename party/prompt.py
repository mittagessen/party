# coding=utf-8
# Copyright 2024 Benjamin Kiessling
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
"""Multimodal prompt encoder"""

import copy
import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ['PromptEncoder']


class PromptEncoder(nn.Module):
    """
    Encodes prompts for input to party's decoder.

    Args:
        embed_dim: The prompts' embedding dimension. Needs to be divisible
        by 8.
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # 4 curve points + 2 bbox corners, box center, box extents
        self.point_embeddings = nn.Embedding(8, embed_dim // 4)
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, embed_dim // 8)))

    def _positional_embed(self, coords):
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def _embed_curves(self, curves: torch.FloatTensor):
        point_embedding = self._positional_embed(curves)
        point_embedding += self.point_embeddings.weight[:4]
        return point_embedding.view(curves.shape[0], -1)

    def _embed_boxes(self, boxes: torch.FloatTensor):
        box_embedding = self._positional_embed(boxes)
        box_embedding += self.point_embeddings.weight[4:]
        return box_embedding.view(boxes.shape[0], -1)

    def forward(self,
                curves: Optional[torch.FloatTensor] = None,
                boxes: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """
        Embeds different types of prompts, either quadratic Bézier curves or
        bounding boxes.

        Args:
          curves: Normalized point coordinates of shape (B_1, 4, 2)
          boxes: Normalized bounding box corner coordinates of shape (B_2, 4, 2)

        Returns:
          Embeddings for the points with shape (B_1+B_2, E)
        """
        embeddings = torch.empty((0, self.embed_dim),
                                 device=self.point_embeddings.weight.device)
        if curves is not None:
            curve_embeddings = self._embed_curves(curves)
            embeddings = torch.cat([embeddings, curve_embeddings])
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            embeddings = torch.cat([embeddings, box_embeddings])

        return embeddings
