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

import timm
import torch
import logging
import lightning.pytorch as L

from torch import nn
from itertools import chain
from lightning.pytorch.callbacks import EarlyStopping
from torch.optim import lr_scheduler

from typing import Literal, Tuple

from torchmetrics.aggregation import MeanMetric

from party.pred import validation_pred
from party.fusion import bytellama_vision_decoder, PartyModel

logger = logging.getLogger(__name__)


class RecognitionModel(L.LightningModule):
    """
    A LightningModule encapsulating the training setup for a text
    recognition model.
    """
    def __init__(self,
                 quit: Literal['fixed', 'early'] = 'fixed',
                 lag: int = 10,
                 optimizer: str = 'Mars',
                 lr: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-3,
                 label_smoothing: float = 0.0,
                 schedule: Literal['cosine', 'exponential', 'step', 'reduceonplateau', 'constant'] = 'cosine',
                 step_size: int = 10,
                 gamma: float = 0.1,
                 rop_factor: float = 0.1,
                 rop_patience: int = 5,
                 cos_t_max: float = 30,
                 cos_min_lr: float = 1e-4,
                 warmup: int = 15000,
                 layer_lr_decay: float = 0.0,
                 encoder: str = 'swin_base_patch4_window12_384.ms_in22k',
                 encoder_input_size: Tuple[int, int] = (2560, 1920),
                 decoder: str = 'mittagessen/bytellama_oscar',
                 decoder_attn_dropout: float = 0.0,
                 pretrained: bool = True,
                 freeze_encoder: bool = False,
                 **kwargs):
        super().__init__()

        self.best_epoch = -1
        self.best_metric = 0.0
        self.best_model = None

        self.save_hyperparameters()

        # enable fused attn in encoder
        timm.layers.use_fused_attn(experimental=True)

        encoder_model = timm.create_model(encoder,
                                          pretrained=pretrained,
                                          num_classes=0,
                                          img_size=encoder_input_size,
                                          global_pool='')

        l_idx = encoder_model.prune_intermediate_layers(indices=(-2,), prune_head=True, prune_norm=True)[0]
        l_red = encoder_model.feature_info[l_idx]['reduction']

        decoder_model = bytellama_vision_decoder(pretrained=decoder if pretrained else None,
                                                 encoder_max_seq_len=encoder_input_size[0] // l_red * encoder_input_size[1] // l_red,
                                                 attn_dropout=decoder_attn_dropout)

        self.model = PartyModel(encoder=encoder_model,
                                decoder=decoder_model,
                                encoder_embed_dim=encoder_model.feature_info[l_idx]['num_chs'],
                                decoder_embed_dim=decoder_model.tok_embeddings.embedding_dim)

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        self.model = torch.compile(self.model)
        self.model.train()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.val_mean = MeanMetric()

    def forward(self, x, curves):
        return self.model(encoder_input=x,
                          encoder_curves=curves)

    def training_step(self, batch, batch_idx):
        tokens = batch['tokens']
        # shift the tokens to create targets
        ignore_idxs = torch.full((tokens.shape[0], 1),
                                 self.criterion.ignore_index,
                                 dtype=tokens.dtype, device=tokens.device)
        targets = torch.hstack((tokens[..., 1:], ignore_idxs)).reshape(-1)

        # our tokens already contain BOS/EOS tokens so we just run it
        # through the model after replacing ignored indices.
        tokens.masked_fill_(tokens == self.criterion.ignore_index, 0)
        logits = self.model(tokens=tokens,
                            encoder_input=batch['image'],
                            encoder_curves=batch['curves'],
                            encoder_boxes=batch['boxes'])

        logits = logits.reshape(-1, logits.shape[-1])
        loss = self.criterion(logits, targets)
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
            logits = self.model(tokens=tokens,
                                encoder_input=batch['image'],
                                encoder_curves=batch['curves'],
                                encoder_boxes=None)

            logits = logits.reshape(-1, logits.shape[-1])
            loss = nn.CrossEntropyLoss()(logits, targets)
            self.val_mean.update(loss)

        if batch['boxes'] is not None:
            logits = self.model(tokens=tokens,
                                encoder_input=batch['image'],
                                encoder_curves=None,
                                encoder_boxes=batch['boxes'])

            logits = logits.reshape(-1, logits.shape[-1])
            loss = nn.CrossEntropyLoss()(logits, targets)
            self.val_mean.update(loss)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log('val_metric', self.val_mean.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('global_step', self.global_step, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.val_mean.reset()

    def save_checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)

    @classmethod
    def load_from_repo(cls, id=None, *args, **kwargs):
        """
        Loads weights from a huggingface hub repository.
        """
        from htrmopo import get_model

        module = cls(*args, **kwargs, pretrained=False)

        model_path = get_model(id) / 'model.safetensors'

        module.model = PartyModel.from_safetensors(model_path)
        module.model = torch.compile(module.model)
        module.model.train()
        return module

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_metric',
                                           mode='min',
                                           patience=self.hparams.lag,
                                           stopping_threshold=1.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams,
                                                     self.model,
                                                     loss_tracking_mode='min')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lr` in `warmup`
        # steps.
        if self.hparams.warmup and self.trainer.global_step < self.hparams.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.warmup)
            for pg in optimizer.param_groups:
                if 'lr_scale' in pg:
                    lr_scale = pg['lr_scale'] * lr_scale
                if self.hparams.optimizer not in ['Adam8bit', 'Adam4bit', 'AdamW8bit', 'AdamW4bit', 'AdamWFp8']:
                    pg['lr'] = lr_scale * self.hparams.lr
                else:
                    pg['lr'].fill_(lr_scale * self.hparams.lr)

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.warmup or self.trainer.global_step >= self.hparams.warmup:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)


def _configure_optimizer_and_lr_scheduler(hparams, model, loss_tracking_mode='min'):
    optimizer = hparams.get("optimizer")
    lr = hparams.get("lr")
    momentum = hparams.get("momentum")
    weight_decay = hparams.get("weight_decay")
    schedule = hparams.get("schedule")
    gamma = hparams.get("gamma")
    cos_t_max = hparams.get("cos_t_max")
    cos_min_lr = hparams.get("cos_min_lr")
    step_size = hparams.get("step_size")
    rop_factor = hparams.get("rop_factor")
    rop_patience = hparams.get("rop_patience")
    completed_epochs = hparams.get("completed_epochs")
    layer_lr_decay = hparams.get('layer_lr_decay')

    if layer_lr_decay:
        param_groups = group_layers(model, weight_decay=weight_decay)
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    # XXX: Warmup is not configured here because it needs to be manually done in optimizer_step()
    logger.debug(f'Constructing {optimizer} optimizer (lr: {lr}, momentum: {momentum})')
    if optimizer in ['Adam', 'AdamW']:
        optim = getattr(torch.optim, optimizer)(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer in ['Adam8bit', 'Adam4bit', 'AdamW8bit', 'AdamW4bit', 'AdamWFp8']:
        import torchao.prototype.low_bit_optim
        optim = getattr(torchao.prototype.low_bit_optim, optimizer)(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'Mars':
        from timm.optim import Mars
        optim = Mars(param_groups, lr=lr, weight_decay=weight_decay, caution=True)
    else:
        optim = getattr(torch.optim, optimizer)(param_groups,
                                                lr=lr,
                                                momentum=momentum,
                                                weight_decay=weight_decay)
    lr_sched = {}
    if schedule == 'exponential':
        lr_sched = {'scheduler': lr_scheduler.ExponentialLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'cosine':
        lr_sched = {'scheduler': lr_scheduler.CosineAnnealingLR(optim,
                                                                cos_t_max,
                                                                cos_min_lr,
                                                                last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'step':
        lr_sched = {'scheduler': lr_scheduler.StepLR(optim, step_size, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'reduceonplateau':
        lr_sched = {'scheduler': lr_scheduler.ReduceLROnPlateau(optim,
                                                                mode=loss_tracking_mode,
                                                                factor=rop_factor,
                                                                patience=rop_patience),
                    'interval': 'step'}
    elif schedule != 'constant':
        raise ValueError(f'Unsupported learning rate scheduler {schedule}.')

    ret = {'optimizer': optim}
    if lr_sched:
        ret['lr_scheduler'] = lr_sched

    if schedule == 'reduceonplateau':
        lr_sched['monitor'] = 'val_accuracy'
        lr_sched['strict'] = False
        lr_sched['reduce_on_plateau'] = True

    return ret


def group_layers(model,
                 layer_decay: float = 0.9,
                 weight_decay: float = 0.05):
    layers = list(chain(model.encoder.layers, model.adapter, model.decoder.layers))
    num_layers = len(layers)
    layer_scales = list(layer_decay ** (num_layers - i - 1) for i in range(num_layers))
    pgs = []
    for scale, layer in zip(layer_scales, layers):
        # filter out non-trainable parameters
        if (params := list(filter(lambda p: p.requires_grad, layer.parameters()))):
            pgs.append({'params': params,
                        'lr_scale': scale,
                        'weight_decay': weight_decay})
    return pgs
