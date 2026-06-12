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
party.cli.ocr
~~~~~~~~~~~~~

Command line driver for recognition inference. Like ``kraken ocr`` but with the
additional options of ``PartyRecognitionInferenceConfig``.

party is a recognition-only model: it transcribes lines from a pre-existing
segmentation and cannot segment images itself. Inputs must therefore be
ALTO or PageXML files carrying the line segmentation.
"""
import logging
import uuid
from pathlib import Path

import click

from .util import message

logging.captureWarnings(True)
logger = logging.getLogger('party')

logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)

# default party model on HTRMoPo
DEFAULT_MODEL = '10.5281/zenodo.14616980'


def _recognize(ctx, model, config, output_mode, output_template, steps,
               input, output):
    """
    Runs recognition on a single input/output pair and serializes the result.
    """
    import dataclasses

    from kraken.lib.xml import XMLPage
    from kraken.lib.util import open_image

    doc = XMLPage(input)
    base_image = doc.imagename
    bounds = doc.to_container()

    if not bounds.lines:
        raise click.UsageError(f'No lines in segmentation of {input}. party '
                               'requires a pre-segmented ALTO/PageXML input.')

    try:
        im = open_image(base_image)
    except IOError as e:
        raise click.BadParameter(str(e))

    it = model.predict(im=im, segmentation=bounds, config=config)

    preds = []
    from kraken.lib.progress import KrakenProgressBar
    with KrakenProgressBar() as progress:
        pred_task = progress.add_task('Processing',
                                      total=len(bounds.lines),
                                      visible=True if not ctx.meta['verbose'] else False)
        for pred in it:
            preds.append(pred)
            progress.update(pred_task, advance=1)

    results = dataclasses.replace(bounds, lines=preds, imagename=base_image)

    with click.open_file(output, 'w', encoding='utf-8') as fp:
        message(f'Writing recognition results for {input}\t', nl=False)
        logger.info('Serializing as {} into {}'.format(output_mode, output))
        if output_mode != 'native':
            from kraken import serialization
            fp.write(serialization.serialize(results=results,
                                             image_size=im.size,
                                             scripts=None,
                                             template=output_template,
                                             template_source='custom' if output_mode == 'template' else 'native',
                                             processing_steps=steps))
        else:
            fp.write('\n'.join(s.prediction for s in preds))
        message('✓', fg='green')


@click.command('ocr')
@click.pass_context
@click.option('-m', '--load-from-repo', default=None, show_default=True,
              help='HTRMoPo identifier of the party model to use for recognition.')
@click.option('-l', '--load-from-file', default=None, show_default=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Path to a party model (safetensors or .ckpt) to use for recognition.')
@click.option('-i', '--input',
              type=(click.Path(exists=True, dir_okay=False, path_type=Path),
                    click.Path(writable=True, dir_okay=False, path_type=Path)),
              multiple=True,
              help='Input-output file pairs. Each input file (first argument) is '
                   'an ALTO/PageXML segmentation mapped to one output file '
                   '(second argument), e.g. `-i input.xml output.xml`')
@click.option('-I', '--batch-input', multiple=True,
              help='Glob expression to add multiple files at once.')
@click.option('-o', '--suffix', default='.xml',
              help='Suffix for output files from batch inputs.')
@click.option('-h', '--hocr', 'serializer',
              help='Switch between hOCR, ALTO, abbyyXML, PageXML or "native" '
                   'output. Native is plain text transcription output.',
              flag_value='hocr')
@click.option('-a', '--alto', 'serializer', flag_value='alto', default=True)
@click.option('-y', '--abbyy', 'serializer', flag_value='abbyyxml')
@click.option('-x', '--pagexml', 'serializer', flag_value='pagexml')
@click.option('-n', '--native', 'serializer', flag_value='native')
@click.option('-t', '--template', type=click.Path(exists=True, dir_okay=False),
              help='Explicit template for output serialization.')
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              help='Number of lines per forward pass batch.')
@click.option('--prompt-mode', type=click.Choice(['curves', 'boxes']),
              help='How to embed line positional prompts. If unset it is '
                   'derived from the segmentation type (baselines -> curves, '
                   'bbox -> boxes).')
@click.option('--max-generated-tokens', type=click.IntRange(1),
              show_default=True, help='Maximum number of tokens to generate per line.')
@click.option('--add-lang-token/--no-lang-token', default=False, show_default=True,
              help='Prepend language tokens from the segmentation to condition '
                   'the decoder on the input language.')
@click.option('-r', '--raise-on-error/--no-raise-on-error', default=False,
              help='Raises the exception that caused processing to fail in the case of an error.')
def ocr(ctx, load_from_repo, load_from_file, input, batch_input, suffix,
        serializer, template, batch_size, prompt_mode, max_generated_tokens,
        add_lang_token, raise_on_error):
    """
    Recognizes text in pre-segmented ALTO/PageXML page images.
    """
    if load_from_file and load_from_repo:
        raise click.BadOptionUsage('load_from_file', 'load_from_* options are mutually exclusive.')
    elif load_from_file is None and load_from_repo is None:
        load_from_repo = DEFAULT_MODEL

    import glob
    import torch

    from threadpoolctl import threadpool_limits

    from kraken.containers import ProcessingStep
    from kraken.tasks import RecognitionTaskModel
    from party.configs import PartyRecognitionInferenceConfig

    torch.set_float32_matmul_precision('medium')

    # output serialization mode
    if template:
        output_mode = 'template'
        output_template = template
    else:
        output_mode = serializer
        output_template = serializer

    # assemble input/output pairs
    io_pairs = list(input)
    if batch_input:
        for batch_expr in batch_input:
            for in_file in glob.glob(str(Path(batch_expr).expanduser()), recursive=True):
                io_pairs.append((Path(in_file), Path(in_file).with_suffix(suffix)))

    if not io_pairs:
        raise click.UsageError('No inputs given. Add input/output pairs with `-i` or `-I`/`-o`.')

    # download model from repo if requested
    if load_from_repo:
        from htrmopo import get_model
        from kraken.lib.progress import KrakenDownloadProgressBar
        with KrakenDownloadProgressBar() as progress:
            download_task = progress.add_task(f'Downloading {load_from_repo}', total=0, visible=True)
            p = get_model(load_from_repo,
                          callback=lambda total, advance: progress.update(download_task, total=total, advance=advance))
            load_from_file = next(p.glob('*.safetensors'))

    config = PartyRecognitionInferenceConfig(accelerator=ctx.meta['accelerator'],
                                             device=ctx.meta['devices'],
                                             precision=ctx.meta['precision'] or '32-true',
                                             num_threads=ctx.meta['num_threads'] or 1,
                                             batch_size=batch_size if batch_size else 1,
                                             prompt_mode=prompt_mode,
                                             max_generated_tokens=max_generated_tokens,
                                             add_lang_token=add_lang_token,
                                             raise_on_error=raise_on_error)

    steps = [ProcessingStep(id=f'_{uuid.uuid4()}',
                            category='processing',
                            description='Text line recognition',
                            settings={'models': load_from_repo or str(load_from_file),
                                      'prompt_mode': prompt_mode,
                                      'max_generated_tokens': max_generated_tokens,
                                      'add_lang_token': add_lang_token})]

    message(f'Loading model from {load_from_file}.')
    model = RecognitionTaskModel.load_model(load_from_file)

    for (in_file, out_file) in io_pairs:
        try:
            with threadpool_limits(limits=ctx.meta['num_threads']):
                _recognize(ctx, model, config, output_mode, output_template,
                           steps, in_file, out_file)
        except Exception as e:
            logger.error(f'Failed processing {in_file}: {str(e)}')
            if raise_on_error:
                raise
