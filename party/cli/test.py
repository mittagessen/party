#
# Copyright 2026 Benjamin Kiessling
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
party.cli.test
~~~~~~~~~~~~~~

Command line driver for recognition testing.
"""
import click
import logging

from dataclasses import asdict

from .util import _expand_gt, _validate_manifests, message
from party.default_specs import RECOGNITION_HYPER_PARAMS

logging.captureWarnings(True)
logger = logging.getLogger('party')

logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('test')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'],
              help='Batch sample size for parallel line generation per page.')
@click.option('-m', '--load-from-repo',
              default=None,
              show_default=True,
              help="HTRMoPo identifier of the party model to evaluate")
@click.option('-i', '--load-from-file',
              default=None,
              show_default=True,
              help="Path to the party model to evaluate (safetensors or .ckpt)")
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('--workers', show_default=True, default=1,
              type=click.IntRange(0),
              help='Number of worker processes when running on CPU.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=None, help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('--curves/--boxes', help='Encode line prompts as bounding boxes or curves', default=None, show_default=True)
@click.option('--compile/--no-compile', help='Switch to enable/disable torch.compile() on model', default=True, show_default=True)
@click.option('--quantize/--no-quantize', help='Switch to enable/disable PTQ', default=False, show_default=True)
@click.option('--add-lang-token/--no-lang-token', help='Switch to enable language tokens.', default=False, show_default=True)
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, batch_size, load_from_repo, load_from_file, evaluation_files,
         workers, normalization, normalize_whitespace, curves,
         compile, quantize, add_lang_token, test_set):
    """
    Tests a model on a compiled dataset.
    """
    if load_from_file and load_from_repo:
        raise click.BadOptionUsage('load_from_file', 'load_from_* options are mutually exclusive.')
    elif load_from_file is None and load_from_repo is None:
        load_from_repo = '10.5281/zenodo.14616981'

    import torch

    from htrmopo import get_model
    from threadpoolctl import threadpool_limits
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import RichProgressBar

    try:
        from kraken.lib.progress import KrakenDownloadProgressBar
    except ImportError:
        raise click.UsageError('Inference requires the kraken package')

    from party.configs import (PartyRecognitionTrainingConfig,
                               PartyRecognitionTrainingDataConfig)
    from party.model import PartyRecognitionModel, PartyTextLineDataModule
    from party.report import render_report

    torch.set_float32_matmul_precision('medium')

    if load_from_repo:
        with KrakenDownloadProgressBar() as progress:
            download_task = progress.add_task(f'Downloading {load_from_repo}', total=0, visible=True)
            p = get_model(load_from_repo,
                          callback=lambda total, advance: progress.update(download_task, total=total, advance=advance))
            load_from_file = next(p.glob('*.safetensors'))

    if curves is True:
        prompt_mode = 'curves'
    elif curves is False:
        prompt_mode = 'boxes'
    else:
        prompt_mode = 'curves'

    # torchao expects bf16 weights
    if quantize:
        ctx.meta['precision'] = 'bf16-true'

    logger.info('Building test set from {} line images'.format(len(test_set) + len(evaluation_files)))

    test_set = list(test_set)
    if evaluation_files:
        test_set.extend(evaluation_files)

    if not test_set:
        raise click.UsageError('No test data provided. Use the `-e` option or positional arguments.')

    dm_config = PartyRecognitionTrainingDataConfig(test_data=test_set,
                                                   batch_size=batch_size,
                                                   val_batch_size=batch_size,
                                                   prompt_mode=prompt_mode,
                                                   num_workers=workers,
                                                   normalization=normalization,
                                                   normalize_whitespace=normalize_whitespace)
    m_config = PartyRecognitionTrainingConfig(add_lang_token=add_lang_token)

    trainer = Trainer(accelerator=ctx.meta['accelerator'],
                      devices=ctx.meta['devices'],
                      precision=ctx.meta['precision'],
                      enable_progress_bar=not ctx.meta['verbose'],
                      enable_model_summary=False,
                      deterministic=ctx.meta['deterministic'],
                      logger=False,
                      callbacks=[RichProgressBar(leave=True)] if not ctx.meta['verbose'] else [],
                      num_sanity_val_steps=0)

    with trainer.init_module(empty_init=False):
        message(f'Loading from {load_from_file}.')
        model = PartyRecognitionModel.load_from_weights(load_from_file, config=m_config)

        if compile:
            click.echo('Compiling model ', nl=False)
            try:
                model.net = torch.compile(model.net, mode='max-autotune')
                click.secho('✓', fg='green')
            except Exception:
                click.secho('✗', fg='red')

        if quantize:
            click.echo('Quantizing model ', nl=False)
            import torchao
            torchao.quantization.utils.recommended_inductor_config_setter()
            click.secho('✓', fg='green')

    data_module = PartyTextLineDataModule(dm_config)

    with threadpool_limits(limits=ctx.meta['num_threads']):
        trainer.test(model, data_module)

    render_report(load_from_file, **asdict(model.test_metrics))
