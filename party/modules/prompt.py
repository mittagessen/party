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
"""Additive line prompt encoder."""
import torch

from torch import nn
from typing import Optional

__all__ = ['PromptEncoder']


@torch.compiler.disable()
class PromptEncoder(nn.Module):
    """
    Encodes line prompts as a single additive embedding vector.

    Curves and boxes are both represented as four 2D points:
    - curves: cubic Bezier control points
    - boxes: xmin/ymin, xmax/ymax, center, width/height
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.point_embeddings = nn.Embedding(8, embed_dim // 4)
        self.register_buffer(
            'positional_encoding_gaussian_matrix',
            torch.randn((2, embed_dim // 8)),
        )

    def _positional_embed(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def _embed_curves(self, curves: torch.Tensor) -> torch.Tensor:
        point_embedding = self._positional_embed(curves)
        point_embedding += self.point_embeddings.weight[:4]
        return point_embedding.view(curves.shape[0], -1)

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        box_embedding = self._positional_embed(boxes)
        box_embedding += self.point_embeddings.weight[4:]
        return box_embedding.view(boxes.shape[0], -1)

    def forward(
        self,
        curves: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = torch.empty(
            (0, self.embed_dim),
            device=self.point_embeddings.weight.device,
        )
        if curves is not None:
            embeddings = torch.cat([embeddings, self._embed_curves(curves)])
        if boxes is not None:
            embeddings = torch.cat([embeddings, self._embed_boxes(boxes)])
        return embeddings
