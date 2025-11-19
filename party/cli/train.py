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

from itertools import repeat
from party.tokenizer import ISO_TO_LANG
from threadpoolctl import threadpool_limits

from .util import _expand_gt, _validate_manifests, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('party')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('convert')
@click.pass_context
@click.option('-o', '--output', type=click.Path(), default='model.safetensors', help='Output model file')
@click.option('-i', '--model-card', default=None, type=click.File(mode='r', lazy=True),
              help='Markdown file containing the model card.')
@click.argument('checkpoint_path', nargs=1, type=click.Path(exists=True, dir_okay=False))
def convert(ctx, output, model_card, checkpoint_path):
    """
    Converts a checkpoint into the new safetensors-based kraken serialization
    format.
    """
    from .util import message

    from party.util import checkpoint_to_kraken

    if model_card:
        model_card = model_card.read()

    checkpoint_to_kraken(checkpoint_path,
                         filename=output,
                         model_card=model_card)

    message(f'Output file written to {output}')


@click.command('compile',
               epilog=f"""
                       Language tags are determined by traversing the path of each XML file
                       upwards until a component is found that matched one of the following
                       identifiers:
                       {', '.join(ISO_TO_LANG.values())}
               """)
@click.pass_context
@click.option('-o', '--output', type=click.Path(), default='dataset.arrow', help='Output dataset file')
@click.option('-F', '--files', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data.')
@click.option('-u', '--normalization', type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              help='Normalizes unicode whitespace')
@click.argument('ground_truth', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def compile(ctx, output, files, normalization, normalize_whitespace,
            ground_truth):
    """
    Precompiles a binary dataset from a collection of XML files.
    """
    from .util import message

    ground_truth = list(ground_truth)

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

        dataset.compile(ground_truth,
                        output,
                        normalization=normalization,
                        normalize_whitespace=normalize_whitespace,
                        callback=_update_bar)

    message(f'Output file written to {output}')


@click.command('train')
@click.pass_context
@click.option('--load', default=None, type=click.Path(exists=True), help='Path to checkpoint/safetensors/zenodo DOI to load')
@click.option('--train-from-scratch', is_flag=True, default=False, help='Train model from scratch')
@click.option('--resume', default=None, type=click.Path(exists=True), help='Path to checkpoint to resume from')
@click.option('-o', '--output', type=click.Path(file_okay=False, dir_okay=True), default='checkpoints', help='Output directory for checkpoints.')
@click.option('-t', '--training-files', default=None, multiple=True,
              type=click.File(mode='r', lazy=True), help='Manifest file(s) with '
              ' additional paths to training data')
@click.option('-e', '--evaluation-files', default=None, multiple=True,
              type=click.File(mode='r', lazy=True), help='Manifest file(s) with '
              ' paths to evaluation data.')
@click.option('-B', '--batch-size', type=int, help='batch sample size')
@click.option('--val-batch-size', type=int, help='validation batch sample size')
@click.option('-F', '--freq',
              type=float,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q', '--quit',
              type=click.Choice(['early',
                                 'fixed']),
              help='Stop condition for training. Set to `early` for early stooping or `fixed` for fixed number of epochs')
@click.option('-N', '--epochs', type=int,
              help='Number of epochs to train for')
@click.option('--min-epochs', type=int,
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--freeze-encoder/--no-freeze-encoder', help='Switch to freeze the encoder')
@click.option('--lag', type=int,
              help='Number of evaluations (--report frequency) to wait before stopping training without improvement')
@click.option('--optimizer',
              type=click.Choice(['Adam',
                                 'AdamW',
                                 'SGD',
                                 'Mars',
                                 'Adam8bit',
                                 'AdamW8bit',
                                 'Adam4bit',
                                 'AdamW4bit']),
              help='Select optimizer')
@click.option('-r', '--lrate', help='Learning rate', type=float)
@click.option('-m', '--momentum', help='Optimizer momentum', type=float)
@click.option('-w', '--weight-decay', help='Weight decay', type=float)
@click.option('--gradient-clip-val', help='Gradient clip value', type=float)
@click.option('--warmup', type=int, help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-g', '--gamma', type=float,
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss', '--step-size', type=int,
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience', type=int,
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max', type=int,
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr', type=float,
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('--augment/--no-augment', help='Enable image augmentation')
@click.option('--prompt-mode',
              type=click.Choice(['boxes',
                                 'curves',
                                 'both']),
              help='Sets line prompt sampling mode: `boxes` for boxes only, '
              '`curves` for curves only, and `both` for randomly switching '
              'between boxes and curves.')
@click.option('--accumulate-grad-batches', type=int, help='Number of batches to accumulate gradient across.')
@click.option('--validate-before-train/--no-validate-before-train', default=True, help='Enables validation run before first training run.')
@click.option('--sampling-weights', type=click.UNPROCESSED, default=repeat(1), hidden=True)
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, **kwargs):
    """
    Trains a model from image-text pairs.
    """
    params = ctx.params
    resume = params.pop('resume', None)
    load = params.pop('load', None)
    train_from_scratch = params.pop('train_from_scratch', None)
    training_files = params.pop('training_files', [])
    evaluation_files = params.pop('evaluation_files', [])
    ground_truth = list(params.pop('ground_truth', []))
    sampling_weights = iter(params.pop('sampling_weights', repeat(1)))

    weights = len(ground_truth) * [1,]
    for training_file in training_files:
        manifest_contents = _validate_manifests(ctx, None, [training_file])
        weights.extend([next(sampling_weights),] * len(manifest_contents))
        ground_truth.extend(manifest_contents)

    ev_files = []
    for evaluation_file in evaluation_files:
        ev_files.extend(_validate_manifests(ctx, None, [evaluation_file]))
    evaluation_files = ev_files

    if not (0 <= params.get('freq') <= 1) and params.get('freq') % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if sum(map(bool, [resume, load, train_from_scratch])) > 1:
        raise click.BadOptionsUsage('load', 'load/resume/train_from_scratch options are mutually exclusive.')
    elif resume is None and load is None and train_from_scratch is False:
        load = '10.5281/zenodo.15075344'

    if params.get('augment'):
        try:
            import albumentations  # NOQA
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    import torch

    from party.dataset import TextLineDataModule
    from party.model import RecognitionModel

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint, RichProgressBar

    torch.set_float32_matmul_precision('high')

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(params['freq'])}
    else:
        val_check_interval = {'val_check_interval': params['freq']}

    cbs = [RichModelSummary(max_depth=2)]

    checkpoint_callback = ModelCheckpoint(dirpath=params.pop('output'),
                                          save_top_k=10,
                                          monitor='global_step',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')

    cbs.append(checkpoint_callback)
    if not ctx.meta['verbose']:
        cbs.append(RichProgressBar(leave=True))

    if resume:
        data_module = TextLineDataModule.load_from_checkpoint(resume)
    else:
        data_module = TextLineDataModule(training_data=ground_truth,
                                         evaluation_data=evaluation_files,
                                         num_workers=ctx.meta['workers'],
                                         sampling_weights=weights,
                                         **params)

    trainer = Trainer(accelerator=accelerator,
                      devices=device,
                      precision=ctx.meta['precision'],
                      max_epochs=params['epochs'] if params['quit'] == 'fixed' else -1,
                      min_epochs=params['min_epochs'],
                      enable_progress_bar=True if not ctx.meta['verbose'] else False,
                      deterministic=ctx.meta['deterministic'],
                      enable_model_summary=False,
                      accumulate_grad_batches=params['accumulate_grad_batches'],
                      callbacks=cbs,
                      gradient_clip_val=params['gradient_clip_val'],
                      num_sanity_val_steps=0,
                      use_distributed_sampler=False,
                      **val_check_interval)

    with trainer.init_module(empty_init=False if train_from_scratch else True):
        if train_from_scratch:
            message('Initializing new model.')
            model = RecognitionModel(**params)
        elif load:
            message(f'Loading from checkpoint {load}.')
            if load.endswith('safetensors'):
                model = RecognitionModel.load_from_safetensors(load, **params)
            elif load.endswith('ckpt'):
                model = RecognitionModel.load_from_checkpoint(load, **params)
            else:
                message(f'Loading from zenodo repository {load}.')
                model = RecognitionModel.load_from_repo(load, **params)
        elif resume:
            message(f'Resuming from checkpoint {resume}.')
            model = RecognitionModel.load_from_checkpoint(resume)

    with threadpool_limits(limits=ctx.meta['threads']):
        if resume:
            trainer.fit(model, data_module, ckpt_path=resume)
        else:
            if params.get('validate_before_train'):
                trainer.validate(model, data_module)
            trainer.fit(model, data_module)

    if not model.current_epoch:
        logger.warning('Training aborted before end of first epoch.')
        ctx.exit(1)
