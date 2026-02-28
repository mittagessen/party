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
# or implied. See the License for the specific language governing permissions and
# limitations under the License.
"""
Training loop interception helpers.
"""
import logging
import math
import timm
import torch
import lightning.pytorch as L

from torch import nn
from torch.optim import lr_scheduler

from torchmetrics.aggregation import MeanMetric
from typing import Optional, TYPE_CHECKING, Union, Any
from lightning.pytorch.callbacks import EarlyStopping
from torch.distributed import get_world_size, is_initialized
from torch.utils.data import RandomSampler, DataLoader

from party.tokenizer import OFFSET, LANG_OFFSET
from party.fusion import (
    bytellama_vision_decoder,
    PartyMultiScaleAdapter,
    SingleScaleAdapter,
)
from party.modules import NoisyTeacherForcing, PromptEncoder
from party.dataset import (
    collate_null,
    get_default_transforms,
    BinnedBaselineDataset,
    ValidationBaselineDataset,
    _validation_worker_init_fn,
)
from party.configs import (
    DEFAULT_DECODER_NAME,
    DEFAULT_DECODER_EMBED_DIM,
    DEFAULT_ENCODER_NAME,
    DEFAULT_FUSION_INTERVAL,
    PartyRecognitionTrainingConfig,
    PartyRecognitionTrainingDataConfig,
)

if TYPE_CHECKING:
    from os import PathLike
    from kraken.models import BaseModel

logger = logging.getLogger(__name__)


def _build_decoder_targets(tokens: torch.Tensor, ignore_index: int) -> torch.Tensor:
    ignore_idxs = torch.full(
        (tokens.shape[0], 1),
        ignore_index,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    return torch.hstack((tokens[..., 1:], ignore_idxs)).reshape(-1)


def get_parameter_groups(model, config) -> list[dict[str, Any]]:
    """
    Create parameter groups with discriminative learning rates.
    """
    base_lr = config.lrate
    encoder_params = []
    full_lr_params = []

    for p in model.nn['encoder'].parameters():
        if p.requires_grad:
            encoder_params.append(p)

    for module_name in ('adapter', 'line_embedding'):
        for p in model.nn[module_name].parameters():
            if p.requires_grad:
                full_lr_params.append(p)

    for _, p in model.nn['decoder'].named_parameters():
        if p.requires_grad:
            full_lr_params.append(p)

    return [
        {
            'params': encoder_params,
            'lr': base_lr * config.lr_pretrained_mult,
            'name': 'encoder',
        },
        {
            'params': full_lr_params,
            'lr': base_lr,
            'name': 'full_lr',
        },
    ]


class AdditivePartyModel(nn.Module):
    """
    Training-only additive prompt party model.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        adapter: nn.Module,
        line_embedding: nn.Module,
    ):
        super().__init__()
        self.nn = nn.ModuleDict(
            {
                'encoder': encoder,
                'decoder': decoder,
                'adapter': adapter,
                'line_embedding': line_embedding,
            }
        )

    def forward_encoder_embeddings(self, encoder_input: torch.Tensor) -> torch.Tensor:
        return self.nn['adapter'](self.nn['encoder'](encoder_input))

    def forward_line_features(
        self,
        encoder_hidden_states: torch.Tensor,
        curves: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if curves is not None and boxes is not None:
            raise ValueError('curves and boxes are mutually exclusive.')
        if curves is None and boxes is None:
            raise ValueError('One of curves or boxes must be provided.')

        line_embeddings = self.nn['line_embedding'](curves=curves, boxes=boxes)
        batch_size = line_embeddings.shape[0]

        if encoder_hidden_states.shape[0] == 1 and batch_size != 1:
            encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)
        elif encoder_hidden_states.shape[0] != batch_size:
            raise ValueError(
                f'encoder_hidden_states batch size ({encoder_hidden_states.shape[0]}) '
                f'does not match prompt batch size ({batch_size}).'
            )

        return encoder_hidden_states + line_embeddings.unsqueeze(1)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_curves: Optional[torch.Tensor] = None,
        encoder_boxes: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None and encoder_input is not None:
            encoder_hidden_states = self.forward_encoder_embeddings(encoder_input)
            encoder_hidden_states = self.forward_line_features(
                encoder_hidden_states,
                curves=encoder_curves,
                boxes=encoder_boxes,
            )

        return self.nn['decoder'](
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_hidden_states,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )


def _build_adapter(
    encoder_out_indices: tuple[int, ...],
    adapter_num_layers: int,
    adapter_num_heads: int,
    adapter_ds_factors: list[int],
    encoder_channels: list[int],
    encoder_sizes: list[tuple[int, int]],
    decoder_embed_dim: int,
) -> tuple[nn.Module, int]:
    if len(encoder_out_indices) == 1:
        if adapter_ds_factors != [1]:
            raise ValueError('single-scale encoder features require adapter_ds_factors=[1].')
        return (
            SingleScaleAdapter(
                num_layers=adapter_num_layers,
                num_heads=adapter_num_heads,
                encoder_embed_dim=encoder_channels[0],
                decoder_embed_dim=decoder_embed_dim,
            ),
            encoder_sizes[0][0] * encoder_sizes[0][1],
        )

    if len(adapter_ds_factors) != len(encoder_out_indices):
        raise ValueError('adapter_ds_factors must have the same length as encoder_out_indices.')

    adapter = PartyMultiScaleAdapter(
        num_layers=adapter_num_layers,
        num_heads=adapter_num_heads,
        encoder_embed_dims=encoder_channels,
        encoder_sizes=encoder_sizes,
        decoder_embed_dim=decoder_embed_dim,
        ds_factors=adapter_ds_factors,
    )
    encoder_max_seq_len = sum(
        (size[0] // ds_factor) * (size[1] // ds_factor)
        for size, ds_factor in zip(encoder_sizes, adapter_ds_factors)
    )
    return adapter, encoder_max_seq_len


def _build_party_model(
    config: PartyRecognitionTrainingConfig,
    image_size: tuple[int, int],
) -> AdditivePartyModel:
    encoder_out_indices = tuple(int(idx) for idx in config.encoder_out_indices)
    adapter_num_layers = int(config.adapter_num_layers)
    adapter_num_heads = int(config.adapter_num_heads)
    adapter_ds_factors = list(config.adapter_ds_factors)

    encoder = timm.create_model(
        DEFAULT_ENCODER_NAME,
        pretrained=True,
        features_only=True,
        out_indices=encoder_out_indices,
    )
    encoder_channels = encoder.feature_info.channels()
    encoder_sizes = [
        (image_size[0] // red, image_size[1] // red)
        for red in encoder.feature_info.reduction()
    ]

    adapter, encoder_max_seq_len = _build_adapter(
        encoder_out_indices=encoder_out_indices,
        adapter_num_layers=adapter_num_layers,
        adapter_num_heads=adapter_num_heads,
        adapter_ds_factors=adapter_ds_factors,
        encoder_channels=encoder_channels,
        encoder_sizes=encoder_sizes,
        decoder_embed_dim=DEFAULT_DECODER_EMBED_DIM,
    )

    decoder = bytellama_vision_decoder(
        pretrained=DEFAULT_DECODER_NAME,
        encoder_max_seq_len=encoder_max_seq_len,
        fusion_interval=DEFAULT_FUSION_INTERVAL,
    )
    decoder_embed_dim = decoder.tok_embeddings.embedding_dim
    if decoder_embed_dim != DEFAULT_DECODER_EMBED_DIM:
        raise ValueError(
            f'Unexpected decoder embedding dimension {decoder_embed_dim}; '
            f'expected {DEFAULT_DECODER_EMBED_DIM}.'
        )

    return AdditivePartyModel(
        encoder=encoder,
        decoder=decoder,
        adapter=adapter,
        line_embedding=PromptEncoder(embed_dim=decoder_embed_dim),
    )


class PartyTextLineDataModule(L.LightningDataModule):
    def __init__(self, data_config: PartyRecognitionTrainingDataConfig):
        super().__init__()

        self.save_hyperparameters()
        self.hparams.data_config.val_batch_size = (
            data_config.batch_size if not data_config.val_batch_size else data_config.val_batch_size
        )

        im_transforms = get_default_transforms(image_size=data_config.image_size)
        augmentation = None
        if data_config.augment:
            from party.augmentation import Augmenter
            augmentation = Augmenter(image_size=data_config.image_size)

        if data_config.training_data and data_config.evaluation_data:
            self.train_set = BinnedBaselineDataset(
                data_config.training_data,
                im_transforms=im_transforms,
                augmentation=augmentation,
                prompt_mode=data_config.prompt_mode,
                batch_size=data_config.batch_size,
            )
            self.val_set = ValidationBaselineDataset(
                data_config.evaluation_data,
                im_transforms=im_transforms,
                prompt_mode=data_config.prompt_mode,
                batch_size=data_config.val_batch_size,
            )
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
            self.test_set = ValidationBaselineDataset(
                data_config.test_data,
                im_transforms=im_transforms,
                prompt_mode=data_config.prompt_mode,
                batch_size=data_config.val_batch_size,
            )
            if len(self.test_set) == 0:
                raise ValueError('No valid test data provided. Please add some.')
        else:
            raise ValueError('Invalid specification of training/evaluation/test data.')

    def train_dataloader(self):
        world_size = get_world_size() if is_initialized() else 1
        sampler = RandomSampler(
            self.train_set,
            replacement=True,
            num_samples=self.train_set.num_batches // world_size,
        )
        dataloader_kwargs = {
            'num_workers': self.hparams.data_config.num_workers,
            'batch_size': 1,
            'sampler': sampler,
            'pin_memory': True,
            'shuffle': False,
            'collate_fn': collate_null,
        }

        if self.hparams.data_config.num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = 4

        return DataLoader(self.train_set, **dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=1,
            num_workers=self.hparams.data_config.num_workers,
            pin_memory=True,
            collate_fn=collate_null,
            worker_init_fn=_validation_worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=1,
            num_workers=self.hparams.data_config.num_workers,
            pin_memory=True,
            collate_fn=collate_null,
            worker_init_fn=_validation_worker_init_fn,
        )


class PartyRecognitionModel(L.LightningModule):
    """
    Lightning wrapper for additive prompt recognition training.
    """

    def __init__(
        self,
        config: PartyRecognitionTrainingConfig,
        model: Optional['BaseModel'] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        if model:
            self.net = model
            if 'recognition' not in self.net.model_type:
                raise ValueError(
                    f'Model {model} is of type {self.net.model_type} while `recognition` is expected.'
                )
        else:
            self.net = None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.val_mean = MeanMetric()

        p_ntf = config.noisy_teacher_forcing
        self._target_ntf_p = p_ntf
        self._ntf_warmup_steps = max(0, int(config.noisy_teacher_forcing_warmup or 0))
        initial_ntf_p = 0.0 if self._ntf_warmup_steps > 0 else p_ntf
        self.noisy_teacher_forcing = (
            nn.Identity()
            if p_ntf == 0.0
            else NoisyTeacherForcing(
                min_label=OFFSET,
                max_label=LANG_OFFSET,
                p=initial_ntf_p,
                ignore_index=self.criterion.ignore_index,
            )
        )

    def forward(self, x, curves=None, boxes=None):
        return self.net(encoder_input=x, encoder_curves=curves, encoder_boxes=boxes)

    def _update_noisy_teacher_forcing_p(self):
        if not isinstance(self.noisy_teacher_forcing, NoisyTeacherForcing):
            return
        if self._ntf_warmup_steps <= 0:
            self.noisy_teacher_forcing.p = self._target_ntf_p
            return
        ntf_scale = min(1.0, float(self.trainer.global_step + 1) / self._ntf_warmup_steps)
        self.noisy_teacher_forcing.p = self._target_ntf_p * ntf_scale

    def _compute_loss(self, batch, apply_ntf: bool) -> torch.Tensor:
        tokens = batch['tokens']
        targets = _build_decoder_targets(tokens, self.criterion.ignore_index)

        decoder_tokens = tokens.clone()
        if apply_ntf:
            decoder_tokens = self.noisy_teacher_forcing(decoder_tokens)
        decoder_tokens.masked_fill_(decoder_tokens == self.criterion.ignore_index, 0)

        encoder_hidden_states = self.net.forward_encoder_embeddings(batch['image'])
        encoder_hidden_states = self.net.forward_line_features(
            encoder_hidden_states,
            curves=batch['curves'],
            boxes=batch['boxes'],
        )

        logits = self.net(tokens=decoder_tokens, encoder_hidden_states=encoder_hidden_states)
        return self.criterion(logits.reshape(-1, logits.shape[-1]), targets)

    def training_step(self, batch, batch_idx):
        self._update_noisy_teacher_forcing_p()
        loss = self._compute_loss(batch, apply_ntf=True)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'NaN/Inf loss detected at batch {batch_idx}, replacing with zero loss')
            loss = 0.0 * sum(p.sum() for p in self.net.parameters() if p.requires_grad)

        self.log(
            'train_loss',
            loss,
            batch_size=batch['tokens'].shape[0],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        loss = None

        if batch['curves'] is not None:
            curve_loss = self._compute_loss(
                {
                    'image': batch['image'],
                    'tokens': batch['tokens'],
                    'curves': batch['curves'],
                    'boxes': None,
                },
                apply_ntf=False,
            )
            self.val_mean.update(curve_loss)
            loss = curve_loss

        if batch['boxes'] is not None:
            box_loss = self._compute_loss(
                {
                    'image': batch['image'],
                    'tokens': batch['tokens'],
                    'curves': None,
                    'boxes': batch['boxes'],
                },
                apply_ntf=False,
            )
            self.val_mean.update(box_loss)
            loss = box_loss

        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log(
                'val_metric',
                self.val_mean.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                'global_step',
                self.global_step,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        self.val_mean.reset()

    def setup(self, stage: Optional[str] = None):
        if stage in [None, 'fit'] and self.net is None:
            self.net = _build_party_model(
                config=self.hparams.config,
                image_size=self.trainer.datamodule.hparams.data_config.image_size,
            )

        if stage in [None, 'fit'] and self.hparams.config.freeze_encoder:
            for param in self.net.nn['encoder'].parameters():
                param.requires_grad = False
            for param in self.net.nn['adapter'].parameters():
                param.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        checkpoint['_module_config'] = self.hparams.config

    def on_load_checkpoint(self, checkpoint):
        if not isinstance(checkpoint['_module_config'], PartyRecognitionTrainingConfig):
            raise ValueError('Checkpoint is not a party model.')

        module_config = checkpoint['_module_config']
        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = _build_party_model(
            config=module_config,
            image_size=data_config.image_size,
        )

    @classmethod
    def load_from_repo(cls, id: str, config: PartyRecognitionTrainingConfig):
        from htrmopo import get_model

        model_path = get_model(id) / 'model.safetensors'
        return cls.load_from_weights(path=model_path, config=config)

    @classmethod
    def load_from_weights(
        cls,
        path: Union[str, 'PathLike'],
        config: PartyRecognitionTrainingConfig,
    ) -> 'PartyRecognitionModel':
        from kraken.models import load_models

        models = load_models(path, tasks=['recognition'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} segmentation models in model file.')
        return cls(config=config, model=models[0])

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(
                EarlyStopping(
                    monitor='val_metric',
                    mode='min',
                    patience=self.hparams.config.lag,
                    stopping_threshold=0.0,
                )
            )
        return callbacks

    def configure_optimizers(self):
        param_groups = get_parameter_groups(self.net, self.hparams.config)

        for pg in param_groups:
            n_params = sum(p.numel() for p in pg['params'])
            logger.info(f"Param group '{pg['name']}': {n_params:,} params, lr={pg['lr']:.2e}")

        self._initial_lrs = [pg['lr'] for pg in param_groups]

        config = self.hparams.config
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
        world_size = get_world_size() if is_initialized() else 1
        per_rank_batches = self.trainer.datamodule.train_set.num_batches // world_size
        accumulate = max(1, self.hparams.config.accumulate_grad_batches)
        steps_per_epoch = max(1, math.ceil(per_rank_batches / accumulate))

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            config.cos_t_max * steps_per_epoch,
            config.cos_min_lr,
            last_epoch=config.completed_epochs * steps_per_epoch - 1,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        del epoch, batch_idx
        optimizer.step(closure=optimizer_closure)

        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.config.warmup)
            for pg, initial_lr in zip(optimizer.param_groups, self._initial_lrs):
                pg['lr'] = lr_scale * initial_lr

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.config.warmup or self.trainer.global_step >= self.hparams.config.warmup:
            if isinstance(scheduler, (lr_scheduler.OneCycleLR, lr_scheduler.CosineAnnealingLR)):
                scheduler.step()
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
