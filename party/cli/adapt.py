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
party.cli.adapt
~~~~~~~~~~~~~~~

Adapt a trained prototype model's vocabulary/prototypes to a new dataset.
"""
import click
import logging
import importlib

from .util import _validate_manifests, message

logging.captureWarnings(True)
logger = logging.getLogger('party')


def _load_model(load):
    """Returns the PartyModel for a .ckpt or weights file."""
    from party.model import PartyRecognitionModel
    if str(load).endswith('.ckpt'):
        return PartyRecognitionModel.load_from_checkpoint(load,
                                                          map_location='cpu',
                                                          weights_only=False).net
    from kraken.models import load_models
    return load_models(load)[0]


@click.command('adapt')
@click.pass_context
@click.option('-i', '--load', required=True, type=click.Path(exists=True, readable=True),
              help='Trained prototype model (.ckpt or weights file) to adapt.')
@click.option('-o', '--output', default='adapted.safetensors', type=click.Path(),
              help='Output weights path.')
@click.option('--weights-format', default='safetensors', help='Output weights format.')
@click.option('-t', '--training-files', 'training_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='Manifest(s) of target dataset shards to adapt towards.')
@click.option('--resize', type=click.Choice(['union', 'new']), default='union',
              help='`union` keeps all existing prototypes and appends new code '
                   'points; `new` restricts the inventory to the target dataset.')
@click.option('--refine-steps', type=int, default=0,
              help='Prototype-table refinement steps (0 = training-free).')
@click.option('--recalibrate-temperature/--no-recalibrate-temperature', default=True,
              help='Refit the head temperature on the target data.')
@click.option('--prompt-mode', type=click.Choice(['boxes', 'curves']), default='curves',
              help='Line prompt geometry to use for the support set.')
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False, readable=True))
def adapt(ctx, **params):
    """
    Resizes and re-seeds a prototype model's character inventory toward a new
    dataset, with the backbone frozen (training-free, or lightly refined via
    --refine-steps). Full fine-tuning is done with `party train --resize`.
    """
    params.update(ctx.meta)
    load = params['load']
    training_data = list(params.pop('training_data', [])) + list(params.pop('files', []))
    if not training_data:
        raise click.UsageError('No target data provided. Pass arrow file(s) as '
                               'arguments and/or manifest(s) via `-t`.')

    from lightning.fabric import Fabric

    from party.adapt import adapt_model, resolve_target_vocab
    from party.dataset import (ValidationBaselineDataset, get_default_transforms,
                               collate_null, _validation_worker_init_fn)
    from torch.utils.data import DataLoader

    message(f'Loading model from {load}.')
    net = _load_model(load)
    net.eval()

    fabric = Fabric(accelerator=ctx.meta['accelerator'],
                    devices=ctx.meta['devices'],
                    precision=ctx.meta['precision'])
    fabric.launch()
    net = fabric._precision.convert_module(net)
    net = fabric.to_device(net)
    dtype = next(net.parameters()).dtype
    image_size = net.user_metadata.get('image_size', (2560, 1920))

    # Deterministic dataset: iterates every page in order and yields all lines,
    # so a single pass seeds every code point in the support inventory.
    transforms = get_default_transforms(image_size=image_size, dtype=dtype)
    support_set = ValidationBaselineDataset(training_data,
                                            im_transforms=transforms,
                                            prompt_mode=params['prompt_mode'])
    target_codepoints = support_set.tokenizer.codepoints

    # Align the support tokenizer with the post-adaptation vocabulary so its
    # token IDs match the rebuilt prototype table.
    plan = resolve_target_vocab(net.tokenizer, target_codepoints, params['resize'])
    support_set.tokenizer = plan.new_tokenizer
    support_loader = DataLoader(support_set,
                                batch_size=1,
                                num_workers=ctx.meta['num_workers'] or 0,
                                collate_fn=collate_null,
                                worker_init_fn=_validation_worker_init_fn)

    report = adapt_model(net,
                         target_codepoints,
                         resize=params['resize'],
                         support_loader=support_loader,
                         recalibrate=params['recalibrate_temperature'],
                         refine_steps=params['refine_steps'],
                         fabric=fabric)

    message(f'Vocabulary: {report.vocab_before} -> {report.vocab_after} '
            f'(+{len(report.added)} added, -{len(report.dropped)} dropped).')
    message(f'Seeded {len(report.seeded_support)} new prototype(s) from support.')
    message(f'Temperature: {report.temperature_before:.3f} -> {report.temperature_after:.3f}.')

    try:
        (entry_point,) = importlib.metadata.entry_points(group='kraken.writers',
                                                         name=params['weights_format'])
        writer = entry_point.load()
    except ValueError:
        raise click.UsageError(f'Unknown weights format `{params["weights_format"]}`.')

    net.cpu()
    opath = writer([net], params['output'])
    message(f'Adapted model written to {opath}.')
