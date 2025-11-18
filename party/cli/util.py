#
# Copyright 2022 Benjamin Kiessling
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
party.cli.util
~~~~~~~~~~~~~~~~~~~~

Command line driver helpers
"""
import os
import yaml
import glob
import logging
from typing import List, Optional, Tuple

import click
import lightning as L

from lightning.pytorch.callbacks import BaseFinetuning

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike


logging.captureWarnings(True)
logger = logging.getLogger('party')


def _recursive_update(a: dict[str, Any],
                      b: dict[str, Any]) -> dict[str, Any]:
    """Like standard ``dict.update()``, but recursive so sub-dict gets updated.

    Ignore elements present in ``b`` but not in ``a``. Unless ``strict`` is set to
    `True`, in which case a `ValueError` exception will be raised.
    """
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _recursive_update(a[k], v)
        else:
            a[k] = b[k]
    return a


def _load_config(ctx: click.Context,
                 param: click.Parameter,
                 path: 'PathLike') -> None:
    """
    Fetch parameters values from configuration file and sets them as defaults.
    """
    logger.info(f"Load configuration matching {path}")
    if path:
        try:
            conf = yaml.safe_load(path)
            # Update the default_map.
            if ctx.default_map is None:
                ctx.default_map = {}
            ctx.default_map = _recursive_update(ctx.default_map, conf)
        except FileNotFoundError:
            logger.critical(f"No configuration file {path} found.")


def _validate_manifests(ctx, param, value):
    images = []
    for manifest in value:
        for entry in manifest.readlines():
            im_p = entry.rstrip('\r\n')
            if os.path.isfile(im_p):
                images.append(im_p)
            else:
                logger.warning('Invalid entry "{}" in {}'.format(im_p, manifest.name))
    return images


def _expand_gt(ctx, param, value):
    images = []
    for expression in value:
        images.extend([x for x in glob.iglob(expression, recursive=True) if os.path.isfile(x)])
    return images


def message(msg, **styles):
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


def to_ptl_device(device: str) -> Tuple[str, Optional[List[int]]]:
    if device.strip() == 'auto':
        return 'auto', 'auto'
    devices = device.split(',')
    if devices[0] in ['cpu', 'mps']:
        return devices[0], 'auto'
    elif any([devices[0].startswith(x) for x in ['tpu', 'cuda', 'hpu', 'ipu']]):
        devices = [device.split(':') for device in devices]
        devices = [(x[0].strip(), x[1].strip()) for x in devices]
        if len(set(x[0] for x in devices)) > 1:
            raise Exception('Can only use a single type of device at a time.')
        dev, _ = devices[0]
        if dev == 'cuda':
            dev = 'gpu'
        return dev, [int(x[1]) for x in devices]
    raise Exception(f'Invalid device {device} specified')


class FreezeEncoder(BaseFinetuning):
    """
    Callback freezing the encoder for a fixed number of iterations.
    """

    def __init__(self, unfreeze_at_iterations=10):
        super().__init__()
        self.unfreeze_at_iteration = unfreeze_at_iterations

    def freeze_before_training(self, pl_module):
        pass

    def finetune_function(self, pl_module, current_epoch, optimizer):
        pass

    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.freeze(pl_module.nn['encoder'])

    def on_train_batch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch, batch_idx) -> None:
        """
        Called for each training batch.
        """
        if trainer.global_step == self.unfreeze_at_iteration:
            for opt_idx, optimizer in enumerate(trainer.optimizers):
                num_param_groups = len(optimizer.param_groups)
                self.unfreeze_and_add_param_group(modules=pl_module.nn['encoder'],
                                                  optimizer=optimizer,
                                                  train_bn=True,)
                current_param_groups = optimizer.param_groups
                self._store(pl_module, opt_idx, num_param_groups, current_param_groups)

    def on_train_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Called when the epoch begins."""
        pass
