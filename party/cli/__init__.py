#! /usr/bin/env python
import logging

import click
from PIL import Image
from rich.logging import RichHandler
from rich.traceback import install

from .pred import ocr
from .test import test
from .train import train, compile, convert

from .util import _load_config

from party.default_specs import RECOGNITION_HYPER_PARAMS


def set_logger(logger=None, level=logging.ERROR):
    logger.addHandler(RichHandler(rich_tracebacks=True))
    logger.setLevel(level)


# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2

logging.captureWarnings(True)
logger = logging.getLogger()

APP_NAME = 'party'

logging.captureWarnings(True)
logger = logging.getLogger(APP_NAME)

# install rich traceback handler
install(suppress=[click])

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.group(context_settings=dict(show_default=True,
                                   default_map={'precision': 'bf16-mixed',
                                                'deterministic': False,
                                                'device': 'cpu',
                                                'workers': 1,
                                                'threads': 1,
                                                'train': RECOGNITION_HYPER_PARAMS,
                                                'compile': RECOGNITION_HYPER_PARAMS}))
@click.version_option()
@click.pass_context
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('-r', '--deterministic/--no-deterministic',
              help="Enables deterministic training. If no seed is given and enabled the seed will be set to 42.")
@click.option('-d', '--device', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--precision',
              type=click.Choice(['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']),
              help='Numerical precision to use for training. Default is 32-bit single-point precision.')
@click.option('--workers', type=click.IntRange(0),
              help='Size of worker pool for data loading')
@click.option('--threads', type=click.IntRange(1),
              help='Size of thread pools for intra-op parallelization')
@click.option('--config',
              type=click.File(mode='r', lazy=True),
              help="Path to configuration file.",
              callback=_load_config,
              is_eager=True,
              expose_value=False,
              required=False)
def cli(ctx, verbose, seed, deterministic, device, precision, workers, threads):
    ctx.meta['deterministic'] = False if not deterministic else 'warn'
    if seed:
        from lightning.pytorch import seed_everything
        seed_everything(seed, workers=True)
    elif deterministic:
        from lightning.pytorch import seed_everything
        seed_everything(42, workers=True)

    if precision not in ['bf16-mixed', '32-true']:
        logger.warning(f'Selected float precision {precision} is not in '
                       '[bf16-mixed, 32-true]. party training is known to be '
                       'unstable in bf16-true mode. Proceed with caution.')
    ctx.meta['verbose'] = verbose
    ctx.meta['device'] = device
    ctx.meta['precision'] = precision
    ctx.meta['workers'] = workers
    ctx.meta['threads'] = threads
    set_logger(logger, level=30 - min(10 * verbose, 20))


cli.add_command(compile)
cli.add_command(convert)
cli.add_command(train)
cli.add_command(test)
cli.add_command(ocr)

if __name__ == '__main__':
    cli()
