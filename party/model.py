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
import math
import timm

from torch import nn
from torch.optim import lr_scheduler

from torchmetrics.aggregation import MeanMetric
from typing import Optional, TYPE_CHECKING, Union, Any
from lightning.pytorch.callbacks import EarlyStopping
from torch.distributed import get_world_size, is_initialized
from torch.utils.data import RandomSampler, DataLoader

from party.tokenizer import OFFSET, LANG_OFFSET
from party.fusion import bytellama_vision_decoder, LocatorReaderConditioner
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
    # shift the tokens to create targets
    ignore_idxs = torch.full((tokens.shape[0], 1),
                             criterion.ignore_index,
                             dtype=tokens.dtype, device=tokens.device)
    targets = torch.hstack((tokens[..., 1:], ignore_idxs)).reshape(-1)

    # apply noisy teacher forcing to decoder inputs, not the targets
    tokens = ntf(tokens.clone())

    # our tokens already contain BOS/EOS tokens so we just run them
    # through the model after replacing ignored indices.
    tokens.masked_fill_(tokens == criterion.ignore_index, 0)
    logits = model(tokens=tokens,
                   encoder_input=batch['image'],
                   encoder_curves=batch['curves'],
                   encoder_boxes=batch['boxes'])

    logits = logits.reshape(-1, logits.shape[-1])
    return criterion(logits, targets)


def get_parameter_groups(model, config) -> list[dict[str, Any]]:
    """Create parameter groups with discriminative learning rates.

    Pretrained group: encoder layers (lower LR)
    Full-LR group: decoder + visual conditioner (base LR)
    """
    base_lr = config.lrate
    encoder_params = []
    full_lr_params = []

    # Encoder params -> reduced LR
    for p in model.nn['encoder'].parameters():
        if p.requires_grad:
            encoder_params.append(p)

    # Visual conditioner params -> base LR
    for p in model.nn['visual_conditioner'].parameters():
        if p.requires_grad:
            full_lr_params.append(p)

    # Decoder params -> base LR (including decoder base layers)
    for _, p in model.nn['decoder'].named_parameters():
        if not p.requires_grad:
            continue
        full_lr_params.append(p)

    param_groups = [{'params': encoder_params,
                     'lr': base_lr * config.lr_pretrained_mult,
                     'name': 'encoder'},
                    {'params': full_lr_params,
                     'lr': base_lr,
                     'name': 'full_lr'}]

    return param_groups


class ConfigurablePartyModel(nn.Module):
    """
    Training-only party model assembled from explicit component selections.
    """
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 visual_conditioner: nn.Module):
        super().__init__()
        self.nn = nn.ModuleDict({'encoder': encoder,
                                 'decoder': decoder,
                                 'visual_conditioner': visual_conditioner})

    def forward(self,
                tokens: torch.Tensor,
                *,
                encoder_input: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_curves: Optional[torch.Tensor] = None,
                encoder_boxes: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if encoder_hidden_states is None and encoder_input is not None:
            encoder_hidden_states = self.nn['visual_conditioner'](
                self.nn['encoder'](encoder_input),
                curves=encoder_curves,
                boxes=encoder_boxes,
            )

        return self.nn['decoder'](tokens=tokens,
                                  mask=mask,
                                  encoder_input=encoder_hidden_states,
                                  encoder_mask=encoder_mask,
                                  input_pos=input_pos)


def _build_party_model_from_components(config: PartyRecognitionTrainingConfig,
                                       image_size: tuple[int, int]) -> ConfigurablePartyModel:
    encoder_name = config.encoder_name
    encoder_out_indices = tuple(int(idx) for idx in config.encoder_out_indices)
    if not encoder_out_indices:
        raise ValueError('encoder_out_indices must not be empty.')

    adapter_num_layers = int(config.adapter_num_layers)
    adapter_num_heads = int(config.adapter_num_heads)
    adapter_ds_factors = list(config.adapter_ds_factors)
    locator_num_tokens = int(config.locator_num_tokens)
    reader_num_tokens = int(config.reader_num_tokens)
    global_num_tokens = int(config.global_num_tokens)
    conditioner_num_rounds = int(config.conditioner_num_rounds)
    prompt_num_layers = int(config.prompt_num_layers)
    prompt_num_heads = int(config.prompt_num_heads)
    conditioner_attn_dropout = float(config.conditioner_attn_dropout)

    decoder_name = config.decoder_name
    fusion_interval = int(config.fusion_interval)

    encoder = timm.create_model(encoder_name,
                                pretrained=True,
                                features_only=True,
                                out_indices=encoder_out_indices)
    encoder_channels = encoder.feature_info.channels()
    encoder_sizes = [(image_size[0] // red, image_size[1] // red)
                     for red in encoder.feature_info.reduction()]

    if len(adapter_ds_factors) != len(encoder_out_indices):
        raise ValueError('adapter_ds_factors must have the same length as encoder_out_indices.')

    encoder_max_seq_len = reader_num_tokens + global_num_tokens
    decoder = bytellama_vision_decoder(pretrained=decoder_name,
                                       encoder_max_seq_len=encoder_max_seq_len,
                                       fusion_interval=fusion_interval)
    decoder_embed_dim = decoder.tok_embeddings.embedding_dim

    visual_conditioner = LocatorReaderConditioner(
        num_layers=adapter_num_layers,
        num_heads=adapter_num_heads,
        encoder_embed_dims=encoder_channels,
        encoder_sizes=encoder_sizes,
        decoder_embed_dim=decoder_embed_dim,
        ds_factors=adapter_ds_factors,
        locator_num_tokens=locator_num_tokens,
        reader_num_tokens=reader_num_tokens,
        global_num_tokens=global_num_tokens,
        num_rounds=conditioner_num_rounds,
        refine_layers=prompt_num_layers,
        refine_num_heads=prompt_num_heads,
        attn_dropout=conditioner_attn_dropout,
    )

    return ConfigurablePartyModel(encoder=encoder,
                                  decoder=decoder,
                                  visual_conditioner=visual_conditioner)


class PartyTextLineDataModule(L.LightningDataModule):
    def __init__(self, data_config: PartyRecognitionTrainingDataConfig):
        super().__init__()

        self.save_hyperparameters()
        self.hparams.data_config.val_batch_size = data_config.batch_size if not data_config.val_batch_size else data_config.val_batch_size

        im_transforms = get_default_transforms(image_size=data_config.image_size)
        augmentation = None
        if data_config.augment:
            from party.augmentation import Augmenter
            augmentation = Augmenter(image_size=data_config.image_size)

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
            logger.info('Training set language statistics:')
            for lang, count in sorted(self.train_set.lang_counts.items(), key=lambda x: -x[1]):
                logger.info(f'  {lang}: {count}')
            logger.info('Validation set language statistics:')
            for lang, count in sorted(self.val_set.lang_counts.items(), key=lambda x: -x[1]):
                logger.info(f'  {lang}: {count}')
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
        dataloader_kwargs = {'num_workers': self.hparams.data_config.num_workers,
                             'batch_size': 1,
                             'sampler': sampler,
                             'pin_memory': True,
                             'shuffle': False,
                             'collate_fn': collate_null}

        if self.hparams.data_config.num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = 4

        return DataLoader(self.train_set, **dataloader_kwargs)

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
            if 'recognition' not in self.net.model_type:
                raise ValueError(f'Model {model} is of type {self.net.model_type} while `recognition` is expected.')
        else:
            self.net = None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.val_mean = MeanMetric()

        p_nft = config.noisy_teacher_forcing
        self._target_ntf_p = p_nft
        self._ntf_warmup_steps = max(0, int(config.noisy_teacher_forcing_warmup or 0))
        initial_ntf_p = 0.0 if self._ntf_warmup_steps > 0 else p_nft
        self.noisy_teacher_forcing = nn.Identity() if p_nft == 0. else NoisyTeacherForcing(min_label=OFFSET,
                                                                                           max_label=LANG_OFFSET,
                                                                                           p=initial_ntf_p,
                                                                                           ignore_index=self.criterion.ignore_index)

        self.model_step = model_step

    def forward(self, x, curves):
        return self.net(encoder_input=x, encoder_curves=curves)

    def _update_noisy_teacher_forcing_p(self):
        if not isinstance(self.noisy_teacher_forcing, NoisyTeacherForcing):
            return
        if self._ntf_warmup_steps <= 0:
            self.noisy_teacher_forcing.p = self._target_ntf_p
            return
        ntf_scale = min(1.0, float(self.trainer.global_step + 1) / self._ntf_warmup_steps)
        self.noisy_teacher_forcing.p = self._target_ntf_p * ntf_scale

    def training_step(self, batch, batch_idx):
        self._update_noisy_teacher_forcing_p()
        loss = self.model_step(self.net, self.noisy_teacher_forcing, self.criterion, batch)
        # Handle NaN/Inf losses in DDP by replacing with zero loss
        # This ensures all processes participate in gradient sync while
        # preventing NaN gradients from corrupting the model
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'NaN/Inf loss detected at batch {batch_idx}, replacing with zero loss')
            # Create zero loss connected to graph via output projection weights
            loss = 0.0 * sum(p.sum() for p in self.net.parameters() if p.requires_grad)
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
                self.net = _build_party_model_from_components(config=self.hparams.config,
                                                              image_size=self.trainer.datamodule.hparams.data_config.image_size)

            if self.hparams.config.freeze_encoder:
                for param in self.net.nn['encoder'].parameters():
                    param.requires_grad = False
                for param in self.net.nn['visual_conditioner'].parameters():
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

        module_config = checkpoint['_module_config']
        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = _build_party_model_from_components(config=module_config,
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
    # scheduler are performed at every step.
    def configure_optimizers(self):
        param_groups = get_parameter_groups(self.net, self.hparams.config)

        # Log parameter groups
        for pg in param_groups:
            n_params = sum(p.numel() for p in pg['params'])
            logger.info(f"Param group '{pg['name']}': {n_params:,} params, lr={pg['lr']:.2e}")

        # Store initial LRs for warmup
        self._initial_lrs = [pg['lr'] for pg in param_groups]

        config = self.hparams.config
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
        world_size = get_world_size() if is_initialized() else 1
        per_rank_batches = self.trainer.datamodule.train_set.num_batches // world_size
        accumulate = max(1, self.hparams.config.accumulate_grad_batches)
        # The train dataloader emits one sampled page-batch per step.
        # Scheduler steps should follow optimizer steps (after grad accumulation).
        steps_per_epoch = max(1, math.ceil(per_rank_batches / accumulate))

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, config.cos_t_max * steps_per_epoch, config.cos_min_lr,
            last_epoch=config.completed_epochs * steps_per_epoch - 1
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate in `warmup` steps.
        # Uses per-group initial LRs for discriminative learning rates.
        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.config.warmup)
            for pg, initial_lr in zip(optimizer.param_groups, self._initial_lrs):
                pg["lr"] = lr_scale * initial_lr

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.config.warmup or self.trainer.global_step >= self.hparams.config.warmup:
            if isinstance(scheduler, (lr_scheduler.OneCycleLR, lr_scheduler.CosineAnnealingLR)):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
