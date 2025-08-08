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
import logging

import click
from threadpoolctl import threadpool_limits

from party.default_specs import RECOGNITION_HYPER_PARAMS
from party.tokenizer import ISO_TO_LANG

from .util import _expand_gt, _validate_manifests, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('party')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('convert')
@click.pass_context
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model.safetensors', help='Output model file')
@click.option('-i', '--model-card', show_default=True, default=None, type=click.File(mode='r', lazy=True),
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
@click.option('-o', '--output', show_default=True, type=click.Path(), default='dataset.arrow', help='Output dataset file')
@click.option('-F', '--files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=RECOGNITION_HYPER_PARAMS['normalization'], help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace', show_default=True,
              default=RECOGNITION_HYPER_PARAMS['normalize_whitespace'], help='Normalizes unicode whitespace')
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
@click.option('--load-from-checkpoint', default=None, type=click.Path(exists=True), help='Path to checkpoint to load')
@click.option('--load-from-safetensors', default=None, type=click.Path(exists=True), help='Path to safetensors file to load')
@click.option('--load-from-repo', default=None, help='Identifier of model on huggingface hub, .e.g `10.5281/zenodo.14616981`')
@click.option('--train-from-scratch', is_flag=True, show_default=True, default=False, help='Train model from scratch')
@click.option('--resume-from-checkpoint', default=None, type=click.Path(exists=True), help='Path to checkpoint to resume from')
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='batch sample size')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-F', '--freq', show_default=True, default=RECOGNITION_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['quit'],
              type=click.Choice(['early',
                                 'fixed']),
              help='Stop condition for training. Set to `early` for early stooping or `fixed` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--freeze-encoder/--no-freeze-encoder', show_default=True,
              default=False, help='Switch to freeze the encoder')
@click.option('--lag',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequency) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. Default is scales the delta by the best loss')
@click.option('--optimizer',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['optimizer'],
              type=click.Choice(['Adam',
                                 'AdamW',
                                 'SGD',
                                 'Mars',
                                 'Adam8bit',
                                 'AdamW8bit',
                                 'Adam4bit',
                                 'AdamW4bit']),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=RECOGNITION_HYPER_PARAMS['lr'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=RECOGNITION_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True, type=float,
              default=RECOGNITION_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--gradient-clip-val', show_default=True, default=RECOGNITION_HYPER_PARAMS['gradient_clip_val'], help='Gradient clip value')
@click.option('--warmup', show_default=True, type=int,
              default=RECOGNITION_HYPER_PARAMS['warmup'], help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              default=RECOGNITION_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['cos_min_lr'],
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--workers', show_default=True, default=1, type=click.IntRange(1), help='Number of worker processes.')
@click.option('--threads', show_default=True, default=1, type=click.IntRange(1), help='Maximum size of OpenMP/BLAS thread pool.')
@click.option('--augment/--no-augment',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['augment'],
              help='Enable image augmentation')
@click.option('--prompt-mode',
              show_default=True,
              type=click.Choice(['boxes',
                                 'curves',
                                 'both']),
              default='both',
              help='Sets line prompt sampling mode: `boxes` for boxes only, '
              '`curves` for curves only, and `both` for randomly switching '
              'between boxes and curves.')
@click.option('--accumulate-grad-batches',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['accumulate_grad_batches'],
              help='Number of batches to accumulate gradient across.')
@click.option('--validate-before-train/--no-validate-before-train', show_default=True, default=True, help='Enables validation run before first training run.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, load_from_checkpoint, load_from_safetensors, load_from_repo,
          train_from_scratch, resume_from_checkpoint, batch_size, output, freq,
          quit, epochs, min_epochs, freeze_encoder, lag, min_delta, optimizer,
          lrate, momentum, weight_decay, gradient_clip_val, warmup, schedule,
          gamma, step_size, sched_patience, cos_max, cos_min_lr,
          training_files, evaluation_files, workers, threads, augment,
          prompt_mode, accumulate_grad_batches, validate_before_train,
          ground_truth):
    """
    Trains a model from image-text pairs.
    """
    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if sum(map(bool, [load_from_checkpoint, load_from_repo, load_from_safetensors, resume_from_checkpoint])) > 1:
        raise click.BadOptionsUsage('load_from_checkpoint', 'load_from_* options are mutually exclusive.')
    elif load_from_checkpoint is None and load_from_repo is None and load_from_safetensors is None and resume_from_checkpoint is None:
        load_from_repo = '10.5281/zenodo.15075344'

    if augment:
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

    hyper_params = RECOGNITION_HYPER_PARAMS.copy()
    hyper_params.update({'freq': freq,
                         'batch_size': batch_size,
                         'quit': quit,
                         'epochs': epochs,
                         'min_epochs': min_epochs,
                         'freeze_encoder': freeze_encoder,
                         'lag': lag,
                         'min_delta': min_delta,
                         'optimizer': optimizer,
                         'lr': lrate,
                         'momentum': momentum,
                         'weight_decay': weight_decay,
                         'warmup': warmup,
                         'schedule': schedule,
                         'gamma': gamma,
                         'step_size': step_size,
                         'rop_patience': sched_patience,
                         'cos_t_max': cos_max,
                         'cos_min_lr': cos_min_lr,
                         'augment': augment,
                         'accumulate_grad_batches': accumulate_grad_batches,
                         'gradient_clip_val': gradient_clip_val,
                         })

    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    if resume_from_checkpoint:
        data_module = TextLineDataModule.load_from_checkpoint(resume_from_checkpoint)
    else:
        data_module = TextLineDataModule(training_data=ground_truth,
                                         evaluation_data=evaluation_files,
                                         prompt_mode=prompt_mode,
                                         augmentation=augment,
                                         batch_size=batch_size,
                                         num_workers=workers)

    cbs = [RichModelSummary(max_depth=2)]

    checkpoint_callback = ModelCheckpoint(dirpath=output,
                                          save_top_k=10,
                                          monitor='global_step',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')

    cbs.append(checkpoint_callback)
    if not ctx.meta['verbose']:
        cbs.append(RichProgressBar(leave=True))

    trainer = Trainer(accelerator=accelerator,
                      devices=device,
                      precision=ctx.meta['precision'],
                      max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'fixed' else -1,
                      min_epochs=hyper_params['min_epochs'],
                      enable_progress_bar=True if not ctx.meta['verbose'] else False,
                      deterministic=ctx.meta['deterministic'],
                      enable_model_summary=False,
                      accumulate_grad_batches=hyper_params['accumulate_grad_batches'],
                      callbacks=cbs,
                      gradient_clip_val=hyper_params['gradient_clip_val'],
                      num_sanity_val_steps=0,
                      **val_check_interval)

    with trainer.init_module(empty_init=True):
        if train_from_scratch:
            message('Initializing new model.')
            model = RecognitionModel(**hyper_params)
        elif load_from_checkpoint:
            message(f'Loading from checkpoint {load_from_checkpoint}.')
            model = RecognitionModel.load_from_checkpoint(load_from_checkpoint,
                                                          **hyper_params)
        elif load_from_safetensors:
            message(f'Loading from safetensors {load_from_safetensors}.')
            model = RecognitionModel.load_from_safetensors(load_from_safetensors,
                                                          **hyper_params)
        elif resume_from_checkpoint:
            message(f'Resuming from checkpoint {resume_from_checkpoint}.')
            model = RecognitionModel.load_from_checkpoint(resume_from_checkpoint)
        elif load_from_repo:
            message(f'Loading from huggingface hub {load_from_repo}.')
            model = RecognitionModel.load_from_repo(load_from_repo,
                                                    **hyper_params)

    with threadpool_limits(limits=threads):
        if resume_from_checkpoint:
            trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint)
        else:
            if validate_before_train:
                trainer.validate(model, data_module)
            trainer.fit(model, data_module)

    if not model.current_epoch:
        logger.warning('Training aborted before end of first epoch.')
        ctx.exit(1)
