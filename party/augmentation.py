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
import math

import torch

from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.functional import InterpolationMode, _get_perspective_coeffs
from typing import Callable


def _transform_points(points: torch.Tensor, matrix: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """
    Maps normalized prompt coordinates through the same geometric transform as
    the page image.

    Torchvision's image warps use inverse mapping internally, so prompt points
    must be transformed with the inverse matrix as well to stay aligned.
    """
    w, h = size
    dtype = points.dtype
    device = points.device

    if matrix.shape[0] == 2:
        pad_row = torch.tensor([[0.0, 0.0, 1.0]], dtype=matrix.dtype, device=matrix.device)
        matrix = torch.cat([matrix, pad_row], dim=0)
        matrix = torch.inverse(matrix)[:2, :]
    else:
        matrix = torch.inverse(matrix)

    scale = torch.tensor([w, h], dtype=dtype, device=device)
    hom_points = torch.cat([points * scale,
                            torch.ones((points.shape[0], 1), dtype=dtype, device=device)],
                           dim=-1)
    mapped = (matrix.to(dtype=dtype, device=device) @ hom_points.T).T
    if matrix.shape[0] == 3:
        mapped = mapped[:, :2] / mapped[:, 2, None]
    else:
        mapped = mapped[:, :2]
    return mapped / scale


def _points_within_image(points: torch.Tensor, eps: float = 1e-6) -> bool:
    return bool(torch.isfinite(points).all() and
                torch.all(points >= -eps) and
                torch.all(points <= 1.0 + eps))


def _box_to_corners(bbox: torch.Tensor) -> torch.Tensor:
    xy_min = bbox[0]
    xy_max = bbox[1]
    return torch.stack([xy_min,
                        torch.stack([xy_max[0], xy_min[1]]),
                        xy_max,
                        torch.stack([xy_min[0], xy_max[1]])],
                       dim=0)


def _box_from_corners(corners: torch.Tensor, eps: float = 1e-6) -> torch.Tensor | None:
    if not _points_within_image(corners, eps=eps):
        return None

    xy_min = corners.min(dim=0).values
    xy_max = corners.max(dim=0).values
    size = xy_max - xy_min
    if torch.any(size <= eps):
        return None

    center = 0.5 * (xy_min + xy_max)
    bbox = torch.stack([xy_min, xy_max, center, size], dim=0)
    return bbox.clamp(0.0, 1.0)


def _crop_points(points: torch.Tensor,
                 offset: torch.Tensor,
                 crop_size: torch.Tensor,
                 source_size: torch.Tensor) -> torch.Tensor:
    offset = offset.to(dtype=points.dtype, device=points.device)
    crop_size = crop_size.to(dtype=points.dtype, device=points.device)
    source_size = source_size.to(dtype=points.dtype, device=points.device)
    return ((points * source_size) - offset) / crop_size


def _crop_bbox(bbox: torch.Tensor,
               offset: torch.Tensor,
               crop_size: torch.Tensor,
               source_size: torch.Tensor) -> torch.Tensor | None:
    corners = _crop_points(_box_to_corners(bbox), offset, crop_size, source_size)
    return _box_from_corners(corners)


def _transform_bbox(bbox: torch.Tensor,
                    matrix: torch.Tensor,
                    size: tuple[int, int]) -> torch.Tensor | None:
    corners = _transform_points(_box_to_corners(bbox), matrix, size)
    return _box_from_corners(corners)


def _line_points(curve: torch.Tensor | None,
                 bbox: torch.Tensor | None) -> torch.Tensor:
    if curve is not None:
        return curve
    if bbox is not None:
        return _box_to_corners(bbox)
    raise ValueError('Line needs either curve or bbox geometry.')


def _transform_lines(
    lines: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]],
    point_transform: Callable[[torch.Tensor], torch.Tensor],
    box_transform: Callable[[torch.Tensor], torch.Tensor | None],
) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
    transformed_lines = []
    for tokens, curve, bbox in lines:
        is_valid = True
        if curve is not None:
            curve = point_transform(curve)
            if not _points_within_image(curve):
                is_valid = False
            else:
                curve = curve.clamp(0.0, 1.0)
        if bbox is not None:
            bbox = box_transform(bbox)
            if bbox is None:
                is_valid = False
        if is_valid:
            transformed_lines.append((tokens, curve, bbox))
    return transformed_lines


def _sample_int_inclusive(low: int, high: int) -> int:
    if low >= high:
        return low
    return int(torch.randint(low, high + 1, (1,)).item())


class RandomResizedCrop(torch.nn.Module):
    def __init__(self,
                 size: tuple[int, int],
                 scale: tuple[float, float] = (0.5, 1.0),
                 ratio: tuple[float, float] = (0.75, 1.3333333333333333),
                 interpolation=InterpolationMode.BILINEAR,
                 p: float = 0.1,
                 max_attempts: int = 8):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.p = p
        self.max_attempts = max_attempts

    def forward(self,
                image: torch.Tensor,
                lines: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]
                ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]]:
        if torch.rand(1) >= self.p or not lines:
            return image, lines

        orig_h, orig_w = image.shape[-2:]
        source_size = torch.tensor([orig_w, orig_h], dtype=torch.float32)

        for _ in range(self.max_attempts):
            _, ref_curve, ref_bbox = lines[torch.randint(len(lines), (1,)).item()]
            ref_points = _line_points(ref_curve, ref_bbox).to(dtype=torch.float32)
            ref_points = ref_points * source_size
            ref_min = ref_points.min(dim=0).values
            ref_max = ref_points.max(dim=0).values

            _, _, h, w = v2.RandomResizedCrop.get_params(image, self.scale, self.ratio)
            if (ref_max[0] - ref_min[0]).item() > w or (ref_max[1] - ref_min[1]).item() > h:
                continue

            left_min = max(0, math.ceil((ref_max[0] - w).item()))
            left_max = min(orig_w - w, math.floor(ref_min[0].item()))
            top_min = max(0, math.ceil((ref_max[1] - h).item()))
            top_max = min(orig_h - h, math.floor(ref_min[1].item()))
            if left_min > left_max or top_min > top_max:
                continue

            left = _sample_int_inclusive(left_min, left_max)
            top = _sample_int_inclusive(top_min, top_max)
            offset = torch.tensor([left, top], dtype=torch.float32)
            crop_size = torch.tensor([w, h], dtype=torch.float32)
            transformed_lines = _transform_lines(
                lines,
                point_transform=lambda pts: _crop_points(pts, offset, crop_size, source_size),
                box_transform=lambda box: _crop_bbox(box, offset, crop_size, source_size),
            )
            if transformed_lines:
                image = F.resized_crop(image, top, left, h, w, self.size, self.interpolation)
                return image, transformed_lines

        return image, lines


class RandomRotation(torch.nn.Module):
    def __init__(self,
                 degrees: tuple[float, float] = (-5.0, 5.0),
                 interpolation=InterpolationMode.BILINEAR,
                 p: float = 0.1):
        super().__init__()
        self.degrees = degrees
        self.interpolation = interpolation
        self.p = p

    def forward(self,
                image: torch.Tensor,
                lines: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]
                ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]]:
        if torch.rand(1) >= self.p or not lines:
            return image, lines

        original_image = image
        angle = torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
        image = F.rotate(image, angle, self.interpolation)

        h, w = image.shape[-2:]
        center_x = w / 2
        center_y = h / 2
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        matrix = torch.tensor([[cos_a, -sin_a, center_x - center_x * cos_a + center_y * sin_a],
                               [sin_a, cos_a, center_y - center_x * sin_a - center_y * cos_a]],
                              dtype=torch.float32)

        transformed_lines = _transform_lines(
            lines,
            point_transform=lambda pts: _transform_points(pts, matrix, (w, h)),
            box_transform=lambda box: _transform_bbox(box, matrix, (w, h)),
        )
        if not transformed_lines:
            return original_image, lines
        return image, transformed_lines


class RandomPerspectiveWarp(torch.nn.Module):
    def __init__(self,
                 distortion_scale: float = 0.2,
                 interpolation=InterpolationMode.BILINEAR,
                 p: float = 0.1):
        super().__init__()
        self.distortion_scale = distortion_scale
        self.interpolation = interpolation
        self.p = p

    def forward(self,
                image: torch.Tensor,
                lines: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]
                ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]]:
        if torch.rand(1) >= self.p or not lines:
            return image, lines

        original_image = image
        h, w = image.shape[-2:]
        startpoints, endpoints = v2.RandomPerspective.get_params(w, h, self.distortion_scale)
        image = F.perspective(image, startpoints, endpoints, self.interpolation)

        a, b, c, d, e, f, g, h_ = _get_perspective_coeffs(startpoints, endpoints)
        matrix = torch.tensor([[a, b, c],
                               [d, e, f],
                               [g, h_, 1.0]],
                              dtype=torch.float32)

        transformed_lines = _transform_lines(
            lines,
            point_transform=lambda pts: _transform_points(pts, matrix, (w, h)),
            box_transform=lambda box: _transform_bbox(box, matrix, (w, h)),
        )
        if not transformed_lines:
            return original_image, lines
        return image, transformed_lines


class Augmenter(torch.nn.Module):
    """
    An augmenter that combines geometric and photometric transformations.
    """
    def __init__(self,
                 image_size: tuple[int, int] = (2560, 1920)):
        super().__init__()
        self.image_size = image_size
        self.resize = v2.Resize(image_size, interpolation=InterpolationMode.BILINEAR)
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
        if tuple(image.shape[-2:]) != tuple(self.image_size):
            image = self.resize(image)
        image = self.photometric_transforms(image)
        image, lines = self.crop(image, lines)
        image, lines = self.rotate(image, lines)
        image, lines = self.perspective(image, lines)
        if tuple(image.shape[-2:]) != tuple(self.image_size):
            image = self.resize(image)
        return image, lines
