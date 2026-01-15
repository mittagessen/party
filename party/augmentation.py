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
import torch
import cv2
import numpy as np

import albumentations.augmentations.crops.functional as Fc
import albumentations.augmentations.geometric.functional as Fg

from albumentations import (Blur, Compose, MedianBlur, MotionBlur,
                            OneOf, PixelDropout, ToFloat, ColorJitter)


logger = logging.getLogger(__name__)


class Augmenter:
    """
    An augmenter that combines geometric and photometric transformations.
    The coordinates are transformed using the functional API of albumentations.
    """
    def __init__(self,
                 rotate: bool = True,
                 crop: bool = True,
                 perspective: bool = True,
                 photometric: bool = True):

        cv2.setNumThreads(0)
        self.rotate = rotate
        self.crop = crop
        self.perspective = perspective

        if photometric:
            self._photometric_transforms = Compose([
                                        ToFloat(),
                                        PixelDropout(p=0.2),
                                        ColorJitter(p=0.5),
                                        OneOf([
                                            MotionBlur(p=0.2),
                                            MedianBlur(blur_limit=3, p=0.1),
                                            Blur(blur_limit=3, p=0.1),
                                        ], p=0.2)
                                       ], p=0.5)
        else:
            # still need to convert to float
            self._photometric_transforms = ToFloat()

    def _transform_points(self, points, M, size):
        w, h = size
        points = points * torch.tensor([w, h], dtype=torch.float32)
        # into homogeneous coordinates
        points = torch.cat([points, torch.ones(points.shape[0], 1)], dim=-1)
        points = (torch.from_numpy(M).float() @ points.T).T
        if M.shape[0] == 3:
            # perspective transform
            points = points[:, :2] / points[:, 2, None]
        else:
            # affine
            points = points[:, :2]
        return points / torch.tensor([w, h], dtype=torch.float32)

    def __call__(self, image, lines):
        # image is numpy array
        # lines is a list of tuples (tokens, curve, bbox)
        h, w = image.shape[:2]

        if self.perspective and torch.rand(1) < 0.3:
            dist = torch.empty(1).uniform_(0.0, 0.08).item()
            src_points = np.array(
                [
                    [0, 0],
                    [w - 1, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                ],
                dtype=np.float32,
            )
            dist_w = dist * w
            dist_h = dist * h
            dst_points = src_points + (np.random.uniform(-1, 1, src_points.shape) * np.array([dist_w, dist_h])).astype(np.float32)

            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # get bounding box of transformed image
            corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)
            x_min, y_min = np.min(transformed_corners, axis=0).ravel()
            x_max, y_max = np.max(transformed_corners, axis=0).ravel()

            # calculate scaling factor
            scale_x = w / (x_max - x_min)
            scale_y = h / (y_max - y_min)
            scale = min(scale_x, scale_y)

            # combine transformations
            translate_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
            scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
            M = scale_matrix @ translate_matrix @ M

            image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE, borderValue=0)

            new_lines = []
            for (tokens, curve, bbox) in lines:
                is_valid = True
                if curve is not None:
                    curve = self._transform_points(curve, M, (w, h))
                    if torch.any(curve < 0) or torch.any(curve > 1):
                        is_valid = False
                if bbox is not None:
                    bbox = self._transform_points(bbox, M, (w, h))
                    if torch.any(bbox < 0) or torch.any(bbox > 1):
                        is_valid = False
                if is_valid:
                    new_lines.append((tokens, curve, bbox))
            lines = new_lines

        if self.rotate and torch.rand(1) < 0.5:
            angle = torch.empty(1).uniform_(-5.0, 5.0).item()
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE, borderValue=0)

            new_lines = []
            for (tokens, curve, bbox) in lines:
                is_valid = True
                if curve is not None:
                    curve = self._transform_points(curve, M, (w, h))
                    if torch.any(curve < 0) or torch.any(curve > 1):
                        is_valid = False
                if bbox is not None:
                    bbox = self._transform_points(bbox, M, (w, h))
                    if torch.any(bbox < 0) or torch.any(bbox > 1):
                        is_valid = False
                if is_valid:
                    new_lines.append((tokens, curve, bbox))
            lines = new_lines

        if self.crop and torch.rand(1) < 0.5:
            scale = torch.empty(1).uniform_(0.7, 1.0).item()
            orig_h, orig_w = h, w
            crop_h = int(orig_h * scale)
            crop_w = int(orig_w * scale)
            y1 = torch.randint(0, orig_h - crop_h + 1, (1,)).item()
            x1 = torch.randint(0, orig_w - crop_w + 1, (1,)).item()
            y2 = y1 + crop_h
            x2 = x1 + crop_w
            image = Fc.crop(image, x1, y1, x2, y2)
            # new size
            h, w = image.shape[:2]

            new_lines = []
            for (tokens, curve, bbox) in lines:
                is_valid = True
                if curve is not None:
                    new_curve = curve * torch.tensor([orig_w, orig_h], dtype=torch.float32)
                    new_curve = new_curve - torch.tensor([x1, y1], dtype=torch.float32)
                    new_curve = new_curve / torch.tensor([w, h], dtype=torch.float32)
                    if torch.any(new_curve < 0) or torch.any(new_curve > 1):
                        is_valid = False
                    curve = new_curve
                if bbox is not None:
                    new_bbox = bbox * torch.tensor([orig_w, orig_h], dtype=torch.float32)
                    new_bbox = new_bbox - torch.tensor([x1, y1], dtype=torch.float32)
                    new_bbox = new_bbox / torch.tensor([w, h], dtype=torch.float32)
                    if torch.any(new_bbox < 0) or torch.any(new_bbox > 1):
                        is_valid = False
                    bbox = new_bbox
                if is_valid:
                    new_lines.append((tokens, curve, bbox))
            lines = new_lines

        o = self._photometric_transforms(image=image)
        image = o['image']

        return image, lines
