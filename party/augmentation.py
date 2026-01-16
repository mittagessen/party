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
Utility functions for data loading and training of VGSL networks.
"""
import logging
import math
import torch

from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode, _get_perspective_coeffs


logger = logging.getLogger(__name__)


def _transform_points(points, M, size):
    w, h = size

    # use inverse matrix for forward mapping of points
    if M.shape[0] == 2:
        M = torch.cat([M, torch.tensor([[0.0, 0.0, 1.0]])], dim=0)
        M = torch.inverse(M)
        M = M[:2, :]
    else:
        M = torch.inverse(M)

    points = points * torch.tensor([w, h], dtype=torch.float32)
    # into homogeneous coordinates
    points = torch.cat([points, torch.ones(points.shape[0], 1)], dim=-1)
    points = (M.float() @ points.T).T
    if M.shape[0] == 3:
        # perspective transform
        points = points[:, :2] / points[:, 2, None]
    else:
        # affine
        points = points[:, :2]
    return points / torch.tensor([w, h], dtype=torch.float32)


class RandomResizedCrop(torch.nn.Module):
    def __init__(self,
                 size: tuple[int, int],
                 scale: tuple[float, float] = (0.5, 1.0),
                 ratio: tuple[float, float] = (0.75, 1.3333333333333333),
                 interpolation = InterpolationMode.BILINEAR,
                 p: float = 0.5):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.p = p

    def forward(self, image, lines):
        if torch.rand(1) < self.p:
            orig_h, orig_w = image.shape[-2:]

            top, left, h, w = v2.RandomResizedCrop.get_params(image, self.scale, self.ratio)

            image = F.resized_crop(image, top, left, h, w, self.size, self.interpolation)

            new_lines = []
            for (tokens, curve, bbox) in lines:
                is_valid = True
                if curve is not None:
                    new_curve = curve * torch.tensor([orig_w, orig_h], dtype=torch.float32)
                    new_curve = new_curve - torch.tensor([left, top], dtype=torch.float32)
                    new_curve = new_curve / torch.tensor([w, h], dtype=torch.float32)
                    if torch.any(new_curve < 0) or torch.any(new_curve > 1):
                        is_valid = False
                    curve = new_curve
                if bbox is not None:
                    new_bbox = bbox * torch.tensor([orig_w, orig_h], dtype=torch.float32)
                    new_bbox = new_bbox - torch.tensor([left, top], dtype=torch.float32)
                    new_bbox = new_bbox / torch.tensor([w, h], dtype=torch.float32)
                    if torch.any(new_bbox < 0) or torch.any(new_bbox > 1):
                        is_valid = False
                    bbox = new_bbox
                if is_valid:
                    new_lines.append((tokens, curve, bbox))
            lines = new_lines
        return image, lines


class RandomRotation(torch.nn.Module):
    def __init__(self, 
                 degrees: tuple[float, float] = (-5.0, 5.0),
                 interpolation=InterpolationMode.BILINEAR,
                 p: float = 0.5):
        super().__init__()
        self.degrees = degrees
        self.interpolation = interpolation
        self.p = p

    def forward(self, image, lines):
        if torch.rand(1) < self.p:
            angle = torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
            image = F.rotate(image, angle, self.interpolation)

            h, w = image.shape[-2:]
            center = [w / 2, h / 2]

            # create rotation matrix
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            M = torch.tensor([[cos_a, -sin_a, center[0] - center[0] * cos_a + center[1] * sin_a],
                              [sin_a, cos_a, center[1] - center[0] * sin_a - center[1] * cos_a]],
                             dtype=torch.float32)

            new_lines = []
            for (tokens, curve, bbox) in lines:
                is_valid = True
                if curve is not None:
                    curve = _transform_points(curve, M, (w, h))
                    if torch.any(curve < 0) or torch.any(curve > 1):
                        is_valid = False
                if bbox is not None:
                    bbox = _transform_points(bbox, M, (w, h))
                    if torch.any(bbox < 0) or torch.any(bbox > 1):
                        is_valid = False
                if is_valid:
                    new_lines.append((tokens, curve, bbox))
            lines = new_lines
        return image, lines


class RandomPerspectiveWarp(torch.nn.Module):
    def __init__(self,
                 distortion_scale: float = 0.2,
                 p: float = 0.5,
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation

    def forward(self, image, lines):
        if torch.rand(1) < self.p:
            h, w = image.shape[-2:]
            startpoints, endpoints = v2.RandomPerspective.get_params(w, h, self.distortion_scale)

            image = F.perspective(image, startpoints, endpoints, self.interpolation)

            a, b, c, d, e, f, g, h_ = _get_perspective_coeffs(startpoints, endpoints)
            M = torch.tensor([[a, b, c], [d, e, f], [g, h_, 1]], dtype=torch.float32)

            new_lines = []
            for (tokens, curve, bbox) in lines:
                is_valid = True
                if curve is not None:
                    curve = _transform_points(curve, M, (w, h))
                    if torch.any(curve < 0) or torch.any(curve > 1):
                        is_valid = False
                if bbox is not None:
                    bbox = _transform_points(bbox, M, (w, h))
                    if torch.any(bbox < 0) or torch.any(bbox > 1):
                        is_valid = False
                if is_valid:
                    new_lines.append((tokens, curve, bbox))
            lines = new_lines
        return image, lines


class Augmenter(torch.nn.Module):
    """
    An augmenter that combines geometric and photometric transformations.
    """
    def __init__(self,
                 image_size: tuple[int, int] = (2560, 1920)):
        super().__init__()
        self.crop = RandomResizedCrop(size=image_size, p=0.1)
        self.rotate = RandomRotation(p=0.1)
        self.perspective = RandomPerspectiveWarp(p=0.1)

        self.photometric_transforms = v2.Compose([v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                                                  v2.RandomGrayscale(),
                                                  v2.GaussianBlur(kernel_size=3)])

    def forward(self, image, lines) -> tuple[torch.Tensor, list]:
        """
        Args:
            image:
            lines: list of tuples.

        Returns:

        """
        image = v2.ToImage()(image)
        image = self.photometric_transforms(image)
        image, lines = self.crop(image, lines)
        image, lines = self.rotate(image, lines)
        image, lines = self.perspective(image, lines)
        return image, lines
