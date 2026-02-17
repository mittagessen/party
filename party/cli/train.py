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
party.cli.train
~~~~~~~~~~~~~~~~~~

Command line driver for recognition training.
"""
import click
import logging
import importlib

from pathlib import Path
from party.tokenizer import ISO_TO_LANG
from threadpoolctl import threadpool_limits
from kraken.registry import OPTIMIZERS, SCHEDULERS, STOPPERS

from .util import _expand_gt, _validate_manifests, message

logging.captureWarnings(True)
logger = logging.getLogger('party')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('compile')
@click.pass_context
@click.option('-o', '--output', type=click.Path(), default='dataset.arrow', help='Output dataset file')
@click.option('-F', '--files', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data.')
@click.option('-u', '--normalization', type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              help='Normalizes unicode whitespace')
@click.option('-r', '--resize', nargs=2, type=int, default=None,
              help='Resize images to fixed (height, width)')
@click.option('--allow-textless/--no-allow-textless', default=False,
              help='Include lines without text in the dataset')
@click.argument('ground_truth', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def compile(ctx, **params):
    """
    Precompiles a binary dataset from a collection of XML files.
    """
    params = ctx.params.copy()
    params.update(ctx.meta)

    files = params.pop('files', [])
    ground_truth = list(params.pop('ground_truth', []))

    from .util import message

    if files:
        ground_truth.extend(files)

    if not ground_truth:
        raise click.UsageError('No training data was provided to the compile command. Use the `ground_truth` argument.')

    from party import dataset
    from rich.progress import Progress, TimeElapsedColumn, MofNCompleteColumn

    with Progress(*Progress.get_default_columns(),
                  TimeElapsedColumn(),
                  MofNCompleteColumn()) as progress:
        extract_task = progress.add_task('Compiling dataset', total=0, start=False, visible=True if not ctx.meta['verbose'] else False)

        def _update_bar(advance, total):
            if not progress.tasks[0].started:
                progress.start_task(extract_task)
            progress.update(extract_task, total=total, advance=advance)

        lang_counts = dataset.compile(ground_truth,
                                        params['output'],
                                        normalization=params['normalization'],
                                        normalize_whitespace=params['normalize_whitespace'],
                                        resize=tuple(params['resize']) if params['resize'] else None,
                                        allow_textless=params['allow_textless'],
                                        callback=_update_bar)

    message(f'Output file written to {params["output"]}')
    if lang_counts:
        message('Language statistics:')
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            message(f'  {lang}: {count}')


@click.command('train')
@click.pass_context
@click.option('-B', '--batch-size', type=int, help='batch sample size')
@click.option('--val-batch-size', type=int, help='validation batch sample size')
@click.option('-o', '--output', 'checkpoint_path', type=click.Path(), default='model', help='Output checkpoint path')
@click.option('--weights-format', default='safetensors', help='Output weights format.')
@click.option('-i', '--load', type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
@click.option('--resume', type=click.Path(exists=True, readable=True), help='Load a checkpoint to continue training')
@click.option('--train-from-scratch', is_flag=True, help='Train model from scratch')
@click.option('-F', '--freq', type=float,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              type=click.Choice(STOPPERS),
              help='Stop condition for training. Set to `early` for early stopping or `fixed` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              type=int,
              help='Number of epochs to train for')
@click.option('--min-epochs',
              type=int,
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              type=int,
              help='Number of evaluations (--report frequency) to wait before stopping training without improvement')
@click.option('--min-delta',
              type=float,
              help='Minimum improvement between epochs to reset early stopping. By default it scales the delta by the best loss')
@click.option('--optimizer',
              type=click.Choice(OPTIMIZERS),
              help='Select optimizer')
@click.option('-r',
              '--lrate',
              type=float,
              help='Learning rate')
@click.option('--lr-pretrained-mult',
              type=float,
              help='Learning rate multiplier for pretrained components (encoder + decoder base layers)')
@click.option('-m',
              '--momentum',
              type=float,
              help='Momentum')
@click.option('-w',
              '--weight-decay',
              type=float,
              help='Weight decay')
@click.option('--gradient-clip-val',
              type=float,
              help='Gradient clip value')
@click.option('--accumulate-grad-batches',
              type=int,
              help='Number of batches to accumulate gradient across.')
@click.option('--warmup',
              type=int,
              help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              type=click.Choice(SCHEDULERS),
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.')
@click.option('-g',
              '--gamma',
              type=float,
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              type=float,
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              'rop_patience',
              type=int,
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              'cos_t_max',
              type=int,
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              type=float,
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('--gradient-clip-val', help='Gradient clip value', type=float)
@click.option('--freeze-encoder/--no-freeze-encoder', help='Switch to freeze the encoder')
@click.option('--warmup', type=int, help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--augment/--no-augment', help='Enable image augmentation')
@click.option('--noisy-teacher-forcing', type=click.FloatRange(0.0, 1.0), help='Probability that each individual target token is altered for NTF.')
@click.option('--label-smoothing', type=click.FloatRange(0.0, 1.0), help='Amount of label smoothing')
@click.option('--accumulate-grad-batches', type=int, help='Number of batches to accumulate gradient across.')
@click.option('-t', '--training-files', 'training_data', multiple=True, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', 'evaluation_data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--prompt-mode',
              type=click.Choice(['boxes',
                                 'curves',
                                 'both']),
              help='Sets line prompt sampling mode: `boxes` for boxes only, '
              '`curves` for curves only, and `both` for randomly switching '
              'between boxes and curves.')
@click.option('--prompt-num-samples',
              type=click.IntRange(1),
              help='Number of filtered prompt tokens produced by PromptCrossAttention.')
@click.option('--logger',
              'pl_logger',
              type=click.Choice(['tensorboard', 'wandb']),
              default=None,
              help='Logger to use for training.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, **kwargs):
    """
    Trains a model from image-text pairs.
    """
    params = ctx.params.copy()
    params.update(ctx.meta)
    resume = params.pop('resume', None)
    load = params.pop('load', None)
    training_data = params.pop('training_data', [])
    ground_truth = list(params.pop('ground_truth', []))

    train_from_scratch = params.pop('train_from_scratch', None)

    for training_file in training_data:
        manifest_contents = _validate_manifests(ctx, None, [training_file])
        ground_truth.extend(manifest_contents)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    params['training_data'] = ground_truth

    if not (0 <= params.get('freq') <= 1) and params.get('freq') % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if sum(map(bool, [resume, load, train_from_scratch])) > 1:
        raise click.BadOptionsUsage('load', 'load/resume/train_from_scratch options are mutually exclusive.')
    elif resume is None and load is None and train_from_scratch is False:
        load = '10.5281/zenodo.15075344'

    import torch

    from party.configs import PartyRecognitionTrainingConfig, PartyRecognitionTrainingDataConfig
    from party.model import PartyRecognitionModel, PartyTextLineDataModule

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint, RichProgressBar
    from kraken.train.utils import KrakenOnExceptionCheckpoint

    torch.set_float32_matmul_precision('high')

    if params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(params['freq'])}
    else:
        val_check_interval = {'val_check_interval': params['freq']}

    cbs = [RichModelSummary(max_depth=3)]
    checkpoint_path = params.pop('checkpoint_path')

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                          save_top_k=10,
                                          monitor='val_metric',
                                          mode='min',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')
    abort_checkpoint_callback = KrakenOnExceptionCheckpoint(dirpath=checkpoint_path,
                                                            filename='checkpoint_abort')

    cbs.append(checkpoint_callback)
    cbs.append(abort_checkpoint_callback)

    dm_config = PartyRecognitionTrainingDataConfig(**params)
    m_config = PartyRecognitionTrainingConfig(**params)

    if resume:
        data_module = PartyTextLineDataModule.load_from_checkpoint(resume)
    else:
        data_module = PartyTextLineDataModule(dm_config)

    if params.get('pl_logger') == 'tensorboard':
        try:
            import tensorboard  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    if params.get('pl_logger') == 'wandb':
        try:
            import wandb  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'wandb logger needs the `wandb` package installed.')

    pl_logger = None
    if params.get('pl_logger') == 'tensorboard':
        from lightning.pytorch.loggers import TensorBoardLogger
        pl_logger = TensorBoardLogger(save_dir=checkpoint_path)
    elif params.get('pl_logger') == 'wandb':
        from lightning.pytorch.loggers import WandbLogger
        pl_logger = WandbLogger(project='party',
                                save_dir=checkpoint_path,
                                log_model=False)

    if not params['verbose']:
        cbs.append(RichProgressBar(leave=True))

    trainer = Trainer(accelerator=ctx.meta['accelerator'],
                      devices=ctx.meta['devices'],
                      precision=ctx.meta['precision'],
                      max_epochs=params['epochs'] if params['quit'] == 'fixed' else -1,
                      min_epochs=params['min_epochs'],
                      enable_progress_bar=True if not ctx.meta['verbose'] else False,
                      deterministic=ctx.meta['deterministic'],
                      enable_model_summary=True,
                      accumulate_grad_batches=params['accumulate_grad_batches'],
                      callbacks=cbs,
                      gradient_clip_val=params['gradient_clip_val'],
                      num_sanity_val_steps=0,
                      logger=pl_logger if pl_logger else False,
                      **val_check_interval)

    if trainer.is_global_zero:
        from rich.table import Table
        from rich.console import Console

        all_langs = set(data_module.train_set.lang_counts.keys()) | set(data_module.val_set.lang_counts.keys())

        table = Table(title='Language Statistics')
        table.add_column('Language', style='cyan')
        table.add_column('ISO', style='dim')
        table.add_column('Training', justify='right')
        table.add_column('Validation', justify='right')

        sorted_langs = sorted(all_langs,
                              key=lambda x: -(data_module.train_set.lang_counts.get(x, 0) +
                                              data_module.val_set.lang_counts.get(x, 0)))

        for lang in sorted_langs:
            train_count = data_module.train_set.lang_counts.get(lang, 0)
            val_count = data_module.val_set.lang_counts.get(lang, 0)
            lang_name = ISO_TO_LANG.get(lang, lang).replace('_', ' ').title()
            table.add_row(lang_name, lang, str(train_count), str(val_count))

        console = Console()
        console.print(table)

    with trainer.init_module(empty_init=False if train_from_scratch else True):
        if train_from_scratch:
            message('Initializing new model.')
            model = PartyRecognitionModel(config=m_config)
        elif load:
            message(f'Loading from checkpoint {load}.')
            if load.endswith('safetensors'):
                model = PartyRecognitionModel.load_from_weights(load, config=m_config)
            elif load.endswith('ckpt'):
                model = PartyRecognitionModel.load_from_checkpoint(load, config=m_config)
            else:
                message(f'Loading from zenodo repository {load}.')
                model = PartyRecognitionModel.load_from_repo(load, config=m_config)
        elif resume:
            message(f'Resuming from checkpoint {resume}.')
            model = PartyRecognitionModel.load_from_checkpoint(resume)

    try:
        (entry_point,) = importlib.metadata.entry_points(group='kraken.writers', name=params['weights_format'])
        writer = entry_point.load()
    except ValueError:
        raise click.UsageError('weights_format', 'Unknown format `{params.get("weights_format")}` for weights.')

    with threadpool_limits(limits=ctx.meta['num_threads']):
        if resume:
            trainer.fit(model, data_module, ckpt_path=resume)
        else:
            trainer.fit(model, data_module)

    score = checkpoint_callback.best_model_score.item()
    weight_path = Path(checkpoint_callback.best_model_path).with_name(f'best_{score:.4f}.{params.get("weights_format")}')
    model = PartyRecognitionModel.load_from_checkpoint(checkpoint_callback.best_model_path, config=m_config)
    opath = writer([model.net], weight_path)
    message(f'Converting best model {checkpoint_callback.best_model_path} (score: {score:.4f}) to weights {opath}')
