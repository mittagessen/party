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
import torch.nn.functional as F

from collections import defaultdict
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from torchmetrics.aggregation import MeanMetric
from typing import Optional, TYPE_CHECKING, Union, Any
from lightning.pytorch.callbacks import EarlyStopping
from torch.distributed import ReduceOp, all_reduce, get_world_size, is_initialized
from torch.utils.data import RandomSampler, DataLoader

from party.party import PartyModel
from party.dataset import (collate_null, get_default_transforms,
                           BinnedBaselineDataset, TestBaselineDataset,
                           ValidationBaselineDataset,
                           _validation_worker_init_fn)
from party.report import compute_script_cer_from_algn, global_align
from party.configs import PartyRecognitionTrainingConfig, PartyRecognitionTrainingDataConfig
from torchmetrics.text import CharErrorRate, WordErrorRate
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from os import PathLike
    from kraken.models import BaseModel

logger = logging.getLogger(__name__)


def _shift_targets(tokens: torch.Tensor, ignore_index: int) -> torch.Tensor:
    # shift the tokens to create targets
    ignore_idxs = torch.full((tokens.shape[0], 1),
                             ignore_index,
                             dtype=tokens.dtype, device=tokens.device)
    return torch.hstack((tokens[..., 1:], ignore_idxs)).reshape(-1)


@torch.compile(dynamic=False)
def model_step(model, tokens, image, curves, boxes, ignore_index, label_smoothing: float = 0.0):
    targets = _shift_targets(tokens, ignore_index)

    # our tokens already contain BOS/EOS tokens so we just run them
    # through the model after replacing ignored indices.
    tokens = tokens.masked_fill(tokens == ignore_index, 0)
    logits = model(tokens=tokens,
                   encoder_input=image,
                   encoder_curves=curves,
                   encoder_boxes=boxes)

    logits = logits.reshape(-1, logits.shape[-1])
    token_loss = F.cross_entropy(logits,
                                 targets,
                                 ignore_index=ignore_index,
                                 reduction='none',
                                 label_smoothing=label_smoothing)
    valid_mask = targets != ignore_index
    loss_sum = token_loss.masked_fill(~valid_mask, 0).sum()
    valid_count = valid_mask.sum()
    loss = loss_sum / valid_count.clamp_min(1)
    return loss, loss_sum, valid_count


def _update_loss_metric(metric: MeanMetric,
                        loss: torch.Tensor,
                        valid_count: torch.Tensor) -> None:
    metric.update(loss.detach(), weight=valid_count.detach().to(loss.dtype))


def _compute_global_mean(metric: MeanMetric) -> torch.Tensor:
    loss_sum = metric.mean_value.detach().clone()
    weight = metric.weight.detach().clone()
    if is_initialized():
        all_reduce(loss_sum, op=ReduceOp.SUM)
        all_reduce(weight, op=ReduceOp.SUM)
    return loss_sum / weight.clamp_min(1)


def _compute_combined_mean(*metrics: MeanMetric) -> torch.Tensor:
    if not metrics:
        raise ValueError('No metrics to combine.')
    loss_sum = metrics[0].mean_value.detach().clone()
    weight = metrics[0].weight.detach().clone()
    for m in metrics[1:]:
        loss_sum = loss_sum + m.mean_value.detach()
        weight = weight + m.weight.detach()
    if is_initialized():
        all_reduce(loss_sum, op=ReduceOp.SUM)
        all_reduce(weight, op=ReduceOp.SUM)
    return loss_sum / weight.clamp_min(1)


class MuonWithAuxAdam(Optimizer):
    """
    Lightning-compatible wrapper that applies Muon to hidden linear 2D
    weights and AdamW everywhere else.

    This mirrors the MuonWithAuxAdam split used in Keller Jordan's reference
    implementation, but delegates the actual optimization to
    ``torch.optim.Muon`` and ``torch.optim.AdamW``.
    """
    def __init__(self,
                 param_groups: list[dict[str, Any]],
                 *,
                 lr: float,
                 weight_decay: float,
                 momentum: float):
        if not hasattr(torch.optim, 'Muon'):
            raise RuntimeError('party requires torch.optim.Muon, available in torch 2.10+.')

        normalized_groups = []
        muon_group_count = 0
        for group in param_groups:
            group = dict(group)
            if 'use_muon' not in group:
                raise ValueError('MuonWithAuxAdam parameter groups must define a `use_muon` flag.')
            group.setdefault('lr', lr)
            group.setdefault('weight_decay', weight_decay)
            if group['use_muon']:
                group.setdefault('momentum', momentum)
                muon_group_count += 1
            normalized_groups.append(group)

        super().__init__(normalized_groups, {})
        self._muon_group_count = muon_group_count
        self._muon_optimizer = None
        self._adamw_optimizer = None

        muon_groups = self.param_groups[:self._muon_group_count]
        adamw_groups = self.param_groups[self._muon_group_count:]
        if muon_groups:
            self._muon_optimizer = torch.optim.Muon(muon_groups,
                                                   adjust_lr_fn='match_rms_adamw')
        if adamw_groups:
            self._adamw_optimizer = torch.optim.AdamW(adamw_groups)

        # Lightning moves optimizer state via optimizer.state, so both child
        # optimizers need to share the wrapper's state mapping.
        self.state = defaultdict(dict)
        self._sync_child_optimizers()

    def _sync_child_optimizers(self):
        if self._muon_optimizer is not None:
            self._muon_optimizer.param_groups = self.param_groups[:self._muon_group_count]
            self._muon_optimizer.state = self.state
        if self._adamw_optimizer is not None:
            self._adamw_optimizer.param_groups = self.param_groups[self._muon_group_count:]
            self._adamw_optimizer.state = self.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if self._muon_optimizer is not None:
            loss = self._muon_optimizer.step(closure=closure)
            closure = None
        if self._adamw_optimizer is not None:
            adamw_loss = self._adamw_optimizer.step(closure=closure)
            if loss is None:
                loss = adamw_loss
        return loss

    def load_state_dict(self, state_dict):
        saved_groups = state_dict.get('param_groups', [])
        if len(saved_groups) != len(self.param_groups) or any('use_muon' not in group for group in saved_groups):
            logger.warning('Ignoring incompatible optimizer state from checkpoint and reinitializing MuonWithAuxAdam.')
            return
        super().load_state_dict(state_dict)
        self._sync_child_optimizers()


def _get_parameter_owners(model: nn.Module) -> dict[str, nn.Module]:
    owners: dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        prefix = f'{module_name}.' if module_name else ''
        for param_name, _ in module.named_parameters(recurse=False):
            owners[f'{prefix}{param_name}'] = module
    return owners


def _should_use_muon(name: str,
                     parameter: nn.Parameter,
                     owner: Optional[nn.Module]) -> bool:
    # Follow the MuonWithAuxAdam split: use Muon for hidden linear matrices,
    # while embeddings, the decoder output head, and non-linear 2D parameters
    # stay on AdamW.
    if parameter.ndim != 2:
        return False
    if not isinstance(owner, nn.Linear):
        return False
    if name.startswith('nn.decoder.output.'):
        return False
    return True


def get_parameter_groups(model) -> list[dict[str, Any]]:
    owners = _get_parameter_owners(model)
    muon_params = []
    muon_param_names = []
    adamw_excluded_2d_params = []
    adamw_excluded_2d_param_names = []
    adamw_other_params = []
    adamw_other_param_names = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        owner = owners.get(name)
        if _should_use_muon(name, parameter, owner):
            muon_params.append(parameter)
            muon_param_names.append(name)
        elif parameter.ndim == 2:
            adamw_excluded_2d_params.append(parameter)
            adamw_excluded_2d_param_names.append(name)
        else:
            adamw_other_params.append(parameter)
            adamw_other_param_names.append(name)

    param_groups = []
    if muon_params:
        param_groups.append({'params': muon_params,
                             'param_names': muon_param_names,
                             'name': 'muon_hidden_linear_2d',
                             'use_muon': True})
    if adamw_excluded_2d_params:
        param_groups.append({'params': adamw_excluded_2d_params,
                             'param_names': adamw_excluded_2d_param_names,
                             'name': 'adamw_embeddings_heads_other_2d',
                             'use_muon': False})
    if adamw_other_params:
        param_groups.append({'params': adamw_other_params,
                             'param_names': adamw_other_param_names,
                             'name': 'adamw_non_2d',
                             'use_muon': False})
    return param_groups


class PartyTextLineDataModule(L.LightningDataModule):
    def __init__(self, data_config: PartyRecognitionTrainingDataConfig):
        super().__init__()

        self.save_hyperparameters()
        self.hparams.data_config.val_batch_size = data_config.batch_size if not data_config.val_batch_size else data_config.val_batch_size

        if not (data_config.training_data and data_config.evaluation_data) and not data_config.test_data:
            raise ValueError('Invalid specification of training/evaluation/test data.')

        self._build_fit_sets = bool(data_config.training_data and data_config.evaluation_data)
        self._build_test_set = bool(data_config.test_data)

        # Eagerly build fit datasets so callers can inspect language statistics
        # before training (see cli/train.py); test set is built lazily in
        # setup('test') to avoid loading test data during fit.
        if self._build_fit_sets:
            self._init_fit_sets()

    def _init_fit_sets(self):
        data_config = self.hparams.data_config
        train_transforms = get_default_transforms(image_size=data_config.image_size)
        eval_transforms = get_default_transforms(image_size=data_config.image_size)
        augmentation = None
        if data_config.augment:
            from party.augmentation import Augmenter
            augmentation = Augmenter(image_size=data_config.image_size)

        self.train_set = BinnedBaselineDataset(data_config.training_data,
                                               im_transforms=train_transforms,
                                               augmentation=augmentation,
                                               prompt_mode=data_config.prompt_mode,
                                               batch_size=data_config.batch_size)
        self.val_set = ValidationBaselineDataset(data_config.evaluation_data,
                                                 im_transforms=eval_transforms,
                                                 prompt_mode=data_config.prompt_mode,
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

    def setup(self, stage: Optional[str] = None):
        if stage == 'test' and self._build_test_set and not hasattr(self, 'test_set'):
            data_config = self.hparams.data_config
            # TestBaselineDataset is generative (returns (PIL.Image, Segmentation)
            # tuples); 'both' is meaningless here, default to curves.
            prompt_mode = data_config.prompt_mode if data_config.prompt_mode in ('curves', 'boxes') else 'curves'
            self.test_set = TestBaselineDataset(files=data_config.test_data,
                                                prompt_mode=prompt_mode)
            if len(self.test_set) == 0:
                raise ValueError('No valid test data provided. Please add some.')

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
            dataloader_kwargs['prefetch_factor'] = 1

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
        # One page per step: TestBaselineDataset yields (PIL.Image, Segmentation)
        # tuples and the model's test_step handles batched line generation
        # internally via PartyModel.predict_string.
        return DataLoader(self.test_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.data_config.num_workers,
                          collate_fn=lambda x: x[0])


@dataclass
class PartyTestMetrics:
    """Aggregated test metrics for a party model.

    Mirrors the keyword arguments of :func:`party.report.render_report`.
    """
    micro_cer: float
    micro_wer: float
    page_macro_cer: float
    page_macro_wer: float
    per_lang_cer: dict[str, float] = field(default_factory=dict)
    per_lang_wer: dict[str, float] = field(default_factory=dict)
    per_lang_page_macro_cer: dict[str, float] = field(default_factory=dict)
    per_lang_page_macro_wer: dict[str, float] = field(default_factory=dict)
    per_script_cer: dict[str, float] = field(default_factory=dict)
    per_script_page_macro_cer: dict[str, float] = field(default_factory=dict)


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

        self.criterion = nn.CrossEntropyLoss()
        self.train_curves_mean = MeanMetric()
        self.train_boxes_mean = MeanMetric()
        self.val_curves_mean = MeanMetric()
        self.val_boxes_mean = MeanMetric()

        self.model_step = model_step

    def forward(self, x, curves):
        return self.net(encoder_input=x, encoder_curves=curves)

    def training_step(self, batch, batch_idx):
        loss, _, valid_count = self.model_step(self.net,
                                               batch['tokens'],
                                               batch['image'],
                                               batch['curves'],
                                               batch['boxes'],
                                               self.criterion.ignore_index,
                                               self.hparams.config.label_smoothing)
        finite_loss = torch.isfinite(loss.detach())
        if is_initialized():
            finite_flag = finite_loss.to(dtype=torch.int32)
            all_reduce(finite_flag, op=ReduceOp.MIN)
            all_ranks_finite = bool(finite_flag.item())
        else:
            all_ranks_finite = bool(finite_loss.item())

        if not all_ranks_finite:
            sample_index = batch.get('index', 'unknown')
            if not bool(finite_loss.item()):
                logger.warning(f'NaN/Inf loss detected at batch {batch_idx}, sample {sample_index}, rank {self.global_rank}; skipping optimizer signal for this batch')
            zero_loss = loss.new_zeros(())
            for parameter in self.net.parameters():
                if parameter.requires_grad:
                    zero_loss = zero_loss + parameter.sum() * 0.0
            loss = zero_loss
        else:
            if batch['curves'] is not None:
                _update_loss_metric(self.train_curves_mean, loss, valid_count)
            else:
                _update_loss_metric(self.train_boxes_mean, loss, valid_count)

        self.log('train_loss_step',
                 loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_curves_loss',
                 _compute_global_mean(self.train_curves_mean),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True)
        self.log('train_boxes_loss',
                 _compute_global_mean(self.train_boxes_mean),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True)
        self.log('train_loss',
                 _compute_combined_mean(self.train_curves_mean, self.train_boxes_mean),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.train_curves_mean.reset()
        self.train_boxes_mean.reset()

    def validation_step(self, batch, batch_idx):
        if batch['curves'] is not None:
            loss, _, valid_count = self.model_step(self.net,
                                                   batch['tokens'],
                                                   batch['image'],
                                                   batch['curves'],
                                                   None,
                                                   self.criterion.ignore_index)
            _update_loss_metric(self.val_curves_mean, loss, valid_count)

        if batch['boxes'] is not None:
            loss, _, valid_count = self.model_step(self.net,
                                                   batch['tokens'],
                                                   batch['image'],
                                                   None,
                                                   batch['boxes'],
                                                   self.criterion.ignore_index)
            _update_loss_metric(self.val_boxes_mean, loss, valid_count)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log('val_curves_metric',
                     _compute_global_mean(self.val_curves_mean),
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True)
            self.log('val_boxes_metric',
                     _compute_global_mean(self.val_boxes_mean),
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True)
            self.log('val_metric',
                     _compute_combined_mean(self.val_curves_mean, self.val_boxes_mean),
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            self.log('global_step', self.global_step, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.val_curves_mean.reset()
        self.val_boxes_mean.reset()

    def on_test_epoch_start(self):
        data_config = self.trainer.datamodule.hparams.data_config
        config = self.hparams.config

        self._test_batch_size = data_config.val_batch_size or data_config.batch_size
        self._test_add_lang_token = bool(getattr(config, 'add_lang_token', False))
        self._test_im_transforms = get_default_transforms(image_size=data_config.image_size,
                                                          dtype=next(self.net.parameters()).dtype)
        self._test_prompt_mode = self.trainer.datamodule.test_set.prompt_mode

        self.net.prepare_for_generation(batch_size=self._test_batch_size,
                                        max_generated_tokens=config.max_generated_tokens)

        # Per-epoch metric state
        self._test_micro_cer = CharErrorRate()
        self._test_micro_wer = WordErrorRate()
        self._test_page_macro_cer = MeanMetric()
        self._test_page_macro_wer = MeanMetric()
        self._test_per_lang_micro_cer = defaultdict(CharErrorRate)
        self._test_per_lang_micro_wer = defaultdict(WordErrorRate)
        self._test_per_lang_page_macro_cer = defaultdict(MeanMetric)
        self._test_per_lang_page_macro_wer = defaultdict(MeanMetric)
        self._test_per_script_page_macro_cer = defaultdict(MeanMetric)
        self._test_algn_gt: list[str] = []
        self._test_algn_pred: list[str] = []

    def test_step(self, batch, batch_idx):
        im, bounds = batch
        try:
            device = next(self.net.parameters()).device
    
            image_input = self._test_im_transforms(im).unsqueeze(0).to(device)
            lines = torch.tensor([line.bbox if line.type == 'bbox' else line.baseline
                                  for line in bounds.lines]).view(-1, 4, 2).to(device)
    
            prompt_mode = self._test_prompt_mode
            languages = bounds.language if self._test_add_lang_token else None
    
            page_cer = CharErrorRate().to(device)
            page_wer = WordErrorRate().to(device)
            lang_page_cer: dict[str, CharErrorRate] = defaultdict(CharErrorRate)
            lang_page_wer: dict[str, WordErrorRate] = defaultdict(WordErrorRate)
            page_algn_gt: list[str] = []
            page_algn_pred: list[str] = []
    
            preds = self.net.predict_string(encoder_input=image_input,
                                            curves=lines if prompt_mode == 'curves' else None,
                                            boxes=lines if prompt_mode == 'boxes' else None,
                                            languages=languages)
    
            for (pred_text, _, _), line in zip(preds, bounds.lines):
                target = line.text
                logger.info(f'pred: {pred_text}\ngt: {target}')
                algn_s_gt, algn_s_pred = global_align(pred_text, target)
                page_algn_gt.extend(algn_s_gt)
                page_algn_pred.extend(algn_s_pred)
                self._test_micro_cer.update(pred_text, target)
                self._test_micro_wer.update(pred_text, target)
                page_cer.update(pred_text, target)
                page_wer.update(pred_text, target)
                if bounds.language:
                    for lang in bounds.language:
                        self._test_per_lang_micro_cer[lang].update(pred_text, target)
                        self._test_per_lang_micro_wer[lang].update(pred_text, target)
                        lang_page_cer[lang].update(pred_text, target)
                        lang_page_wer[lang].update(pred_text, target)
    
            self._test_algn_gt.extend(page_algn_gt)
            self._test_algn_pred.extend(page_algn_pred)
    
            for k, v in compute_script_cer_from_algn(page_algn_gt, page_algn_pred).items():
                self._test_per_script_page_macro_cer[k].update(v)
    
            for lang in lang_page_cer:
                self._test_per_lang_page_macro_cer[lang].update(lang_page_cer[lang].compute())
                self._test_per_lang_page_macro_wer[lang].update(lang_page_wer[lang].compute())
            self._test_page_macro_cer.update(page_cer.compute())
            self._test_page_macro_wer.update(page_wer.compute())
        except Exception:
            logger.warning(f'Sample {batch_idx} failed to process.', exc_info=True)

    def on_test_epoch_end(self):
        per_lang_cer = {k: float(v.compute()) for k, v in self._test_per_lang_micro_cer.items()}
        per_lang_wer = {k: float(v.compute()) for k, v in self._test_per_lang_micro_wer.items()}
        per_lang_page_macro_cer = {k: float(v.compute()) for k, v in self._test_per_lang_page_macro_cer.items()}
        per_lang_page_macro_wer = {k: float(v.compute()) for k, v in self._test_per_lang_page_macro_wer.items()}
        per_script_page_macro_cer = {k: float(v.compute()) for k, v in self._test_per_script_page_macro_cer.items()}
        per_script_cer = compute_script_cer_from_algn(self._test_algn_gt, self._test_algn_pred)

        self.test_metrics = PartyTestMetrics(
            micro_cer=float(self._test_micro_cer.compute()),
            micro_wer=float(self._test_micro_wer.compute()),
            page_macro_cer=float(self._test_page_macro_cer.compute()),
            page_macro_wer=float(self._test_page_macro_wer.compute()),
            per_lang_cer=per_lang_cer,
            per_lang_wer=per_lang_wer,
            per_lang_page_macro_cer=per_lang_page_macro_cer,
            per_lang_page_macro_wer=per_lang_page_macro_wer,
            per_script_cer=per_script_cer,
            per_script_page_macro_cer=per_script_page_macro_cer,
        )

        for k in ('micro_cer', 'micro_wer', 'page_macro_cer', 'page_macro_wer'):
            self.log(f'test_{k}', getattr(self.test_metrics, k), on_epoch=True, logger=True)

    def setup(self, stage: Optional[str] = None):
        if stage in [None, 'fit']:
            if self.net is None:
                self.net = PartyModel(model_variant=self.hparams.config.model_variant,
                                      image_size=self.trainer.datamodule.hparams.data_config.image_size)

            if self.hparams.config.freeze_encoder:
                for param in self.net.nn['encoder'].parameters():
                    param.requires_grad = False
                for param in self.net.nn['adapter'].parameters():
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
        self.net = PartyModel(model_variant=module_config.model_variant,
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
        param_groups = get_parameter_groups(self.net)

        # Log parameter groups
        for pg in param_groups:
            n_params = sum(p.numel() for p in pg['params'])
            logger.info(f"Param group '{pg['name']}': {n_params:,} params")

        # Store initial LRs for warmup
        config = self.hparams.config
        self._initial_lrs = [config.lrate for _ in param_groups]
        optimizer = MuonWithAuxAdam(param_groups,
                                    lr=config.lrate,
                                    weight_decay=config.weight_decay,
                                    momentum=config.momentum)
        world_size = get_world_size() if is_initialized() else 1
        per_rank_batches = self.trainer.datamodule.train_set.num_batches // world_size
        accumulate = max(1, self.hparams.config.accumulate_grad_batches)
        # The train dataloader emits one sampled page-batch per step.
        # Scheduler steps should follow optimizer steps (after grad accumulation).
        steps_per_epoch = max(1, math.ceil(per_rank_batches / accumulate))

        cosine_steps = max(1, config.cos_t_max * steps_per_epoch - (config.warmup or 0))
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, cosine_steps, config.cos_min_lr,
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
