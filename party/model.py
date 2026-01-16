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
Training loop interception helpers
"""
import torch
import logging
import lightning.pytorch as L

from torch import nn
from torch.optim import lr_scheduler
from kraken.models import create_model

from torchmetrics.aggregation import MeanMetric
from typing import Optional, TYPE_CHECKING, Union
from lightning.pytorch.callbacks import EarlyStopping
from torch.distributed import get_world_size, is_initialized
from kraken.train.utils import configure_optimizer_and_lr_scheduler
from torch.utils.data import RandomSampler, DataLoader

from party.tokenizer import OFFSET, LANG_OFFSET
from party.modules import NoisyTeacherForcing
from party.dataset import (collate_null, get_default_transforms,
                           BinnedBaselineDataset, ValidationBaselineDataset,
                           _validation_worker_init_fn)
from party.configs import PartyRecognitionTrainingConfig, PartyRecognitionTrainingDataConfig

if TYPE_CHECKING:
    from os import PathLike
    from kraken.models import BaseModel

logger = logging.getLogger(__name__)


@torch.compile(dynamic=False)
def model_step(model, ntf, criterion, batch):
    tokens = batch['tokens']
    targets = ntf(tokens.clone()[..., 1:])
    # shift the tokens to create targets
    ignore_idxs = torch.full((tokens.shape[0], 1),
                             criterion.ignore_index,
                             dtype=tokens.dtype, device=tokens.device)
    targets = torch.hstack((targets, ignore_idxs)).reshape(-1)

    # our tokens already contain BOS/EOS tokens so we just run it
    # through the model after replacing ignored indices.
    tokens.masked_fill_(tokens == criterion.ignore_index, 0)
    logits = model(tokens=tokens,
                   encoder_input=batch['image'],
                   encoder_curves=batch['curves'],
                   encoder_boxes=batch['boxes'])

    logits = logits.reshape(-1, logits.shape[-1])
    return criterion(logits, targets)


class PartyTextLineDataModule(L.LightningDataModule):
    def __init__(self, data_config: PartyRecognitionTrainingDataConfig):
        super().__init__()

        self.save_hyperparameters()
        self.hparams.data_config.val_batch_size = data_config.batch_size if not data_config.val_batch_size else data_config.val_batch_size

        im_transforms = get_default_transforms(image_size=data_config.image_size)
        if data_config.augment:
            from party.augmentation import Augmenter
            augmentation = Augmenter(image_size=data_config.image_size)
        else:
            augmentation = None

        if data_config.training_data and data_config.evaluation_data:
            self.train_set = BinnedBaselineDataset(data_config.training_data,
                                                   im_transforms=im_transforms,
                                                   augmentation=augmentation,
                                                   batch_size=data_config.batch_size)
            self.val_set = ValidationBaselineDataset(data_config.evaluation_data,
                                                     im_transforms=im_transforms,
                                                     batch_size=data_config.val_batch_size)
            if len(self.train_set) == 0:
                raise ValueError('No valid training data provided. Please add some.')
            if self.val_set.max_seq_len == 0:
                raise ValueError('No valid validation data provided. Please add some.')
            self.train_set.max_seq_len = max(self.train_set.max_seq_len, self.val_set.max_seq_len)
            self.val_set.max_seq_len = self.train_set.max_seq_len
        elif data_config.test_data:
            self.test_set = ValidationBaselineDataset(data_config.test_data,
                                                      im_transforms=im_transforms,
                                                      batch_size=data_config.val_batch_size)
            if len(self.test_set) == 0:
                raise ValueError('No valid test data provided. Please add some.')
        else:
            raise ValueError('Invalid specification of training/evaluation/test data.')

    def train_dataloader(self):
        world_size = get_world_size() if is_initialized() else 1
        sampler = RandomSampler(self.train_set,
                                replacement=True,
                                num_samples=self.train_set.num_batches // world_size)

        return DataLoader(self.train_set,
                          num_workers=self.hparams.data_config.num_workers,
                          batch_size=1,
                          sampler=sampler,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=collate_null)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          collate_fn=collate_null,
                          worker_init_fn=_validation_worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          collate_fn=collate_null,
                          worker_init_fn=_validation_worker_init_fn)


class PartyRecognitionModel(L.LightningModule):
    """
    A LightningModule encapsulating the training setup for a text
    recognition model.
    """
    def __init__(self,
                 config: PartyRecognitionTrainingConfig,
                 model: Optional['BaseModel'] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        if model:
            self.net = model

            if self.net.model_type not in [None, 'recognition']:
                raise ValueError(f'Model {model} is of type {self.net.model_type} while `recognition` is expected.')
        else:
            self.net = None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.val_mean = MeanMetric()

        p_nft = config.noisy_teacher_forcing
        self.noisy_teacher_forcing = nn.Identity() if p_nft == 0. else NoisyTeacherForcing(min_label=OFFSET,
                                                                                           max_label=LANG_OFFSET,
                                                                                           p=p_nft)

        self.model_step = model_step

    def forward(self, x, curves):
        return self.net(encoder_input=x, encoder_curves=curves)

    def training_step(self, batch, batch_idx):
        loss = self.model_step(self.net, self.noisy_teacher_forcing, self.criterion, batch)
        self.log('train_loss',
                 loss,
                 batch_size=batch['tokens'].shape[0],
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens = batch['tokens']
        # shift the tokens to create targets
        ignore_idxs = torch.full((tokens.shape[0], 1),
                                 self.criterion.ignore_index,
                                 dtype=tokens.dtype, device=tokens.device)
        targets = torch.hstack((tokens[..., 1:], ignore_idxs)).reshape(-1)

        # our tokens already contain BOS/EOS tokens so we just run it
        # through the model after replacing ignored indices.
        tokens.masked_fill_(tokens == self.criterion.ignore_index, 0)

        if batch['curves'] is not None:
            logits = self.net(tokens=tokens,
                              encoder_input=batch['image'],
                              encoder_curves=batch['curves'],
                              encoder_boxes=None)

            logits = logits.reshape(-1, logits.shape[-1])
            loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
            self.val_mean.update(loss)

        if batch['boxes'] is not None:
            logits = self.net(tokens=tokens,
                              encoder_input=batch['image'],
                              encoder_curves=None,
                              encoder_boxes=batch['boxes'])

            logits = logits.reshape(-1, logits.shape[-1])
            loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
            self.val_mean.update(loss)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log('val_metric', self.val_mean.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('global_step', self.global_step, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.val_mean.reset()

    def setup(self, stage: Optional[str] = None):
        if stage in [None, 'fit']:
            if self.net is None:
                self.net = create_model('PartyModel',
                                        pretrained=True,
                                        image_size=self.trainer.datamodule.hparams.data_config.image_size)

            if self.hparams.config.freeze_encoder:
                for param in self.net.encoder.parameters():
                    param.requires_grad = False
                for param in self.net.adapter.parameters():
                    param.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        """
        Save hyperparameters a second time so we can set parameters that
        shouldn't be overwritten in on_load_checkpoint.
        """
        checkpoint['_module_config'] = self.hparams.config

    def on_load_checkpoint(self, checkpoint):
        """
        Reconstruct the model from the spec here and not in setup() as
        otherwise the weight loading will fail.
        """
        if not isinstance(checkpoint['_module_config'], PartyRecognitionTrainingConfig):
            raise ValueError('Checkpoint is not a party model.')

        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = create_model('PartyModel',
                                pretrained=True,
                                image_size=data_config.image_size)

    @classmethod
    def load_from_repo(cls,
                       id: str,
                       config: PartyRecognitionTrainingConfig):
        """
        Loads weights from HTRMoPo.
        """
        from htrmopo import get_model

        model_path = get_model(id) / 'model.safetensors'
        return cls.load_from_weights(path=model_path, config=config)

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: PartyRecognitionTrainingConfig) -> 'PartyRecognitionModel':
        """
        Initializes the module from a model weights file.
        """
        from kraken.models import load_models
        models = load_models(path, tasks=['recognition'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} segmentation models in model file.')
        return cls(config=config, model=models[0])

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_metric',
                                           mode='min',
                                           patience=self.hparams.config.lag,
                                           stopping_threshold=0.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return configure_optimizer_and_lr_scheduler(self.hparams.config,
                                                    self.net.parameters(),
                                                    len_train_set=len(self.trainer.datamodule.train_set),
                                                    loss_tracking_mode='min')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.config.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.config.lrate

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.config.warmup or self.trainer.global_step >= self.hparams.config.warmup:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
