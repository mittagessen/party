#!/usr/bin/env python
import click
import uuid

from party.cli.util import _validate_manifests, to_ptl_device
from party.default_specs import RECOGNITION_HYPER_PARAMS

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna


def train_model(trial: 'optuna.trial.Trial',
                checkpoint,
                accelerator,
                device,
                num_workers,
                training_data,
                evaluation_data,
                prompt_mode,
                epochs) -> float:

    from party.dataset import TextLineDataModule
    from party.model import RecognitionModel
    from party.lightning_utils import PyTorchLightningPruningCallback

    from lightning.pytorch import Trainer
    from threadpoolctl import threadpool_limits

    hyper_params = RECOGNITION_HYPER_PARAMS.copy()
    hyper_params['epochs'] = epochs
    hyper_params['cos_t_max'] = epochs
    hyper_params['optimizer'] = 'AdamW8bit'
    hyper_params['freeze_encoder'] = True

    hyper_params['batch_size'] = 40
    hyper_params['augment'] = trial.suggest_categorical('augment', [True, False])

    hyper_params['accumulate_grad_batches'] = trial.suggest_int('accumulate_grad_batches', 1, 8, step=2)
    hyper_params['label_smoothing'] = trial.suggest_loguniform('label_smoothing', 1e-2, 0.2)
    hyper_params['decoder_attn_dropout'] = trial.suggest_categorical('decoder_attn_dropout', [0.0, 0.1, 0.2])

    hyper_params['warmup'] = trial.suggest_int('warmup', 50, 500, log=True)
    hyper_params['lr'] = trial.suggest_loguniform('lr', 1e-8, 1e-3)
    hyper_params['cos_min_lr'] = hyper_params['lr']/10
    hyper_params['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

    data_module = TextLineDataModule(training_data=training_data,
                                     evaluation_data=evaluation_data,
                                     batch_size=hyper_params['batch_size'],
                                     augmentation=True,
                                     prompt_mode=prompt_mode,
                                     num_workers=num_workers)

    model = RecognitionModel.load_from_checkpoint(checkpoint, **hyper_params)

    trainer = Trainer(accelerator=accelerator,
                      devices=device,
                      precision='bf16-true',
                      max_epochs=hyper_params['epochs'],
                      min_epochs=hyper_params['min_epochs'],
                      enable_progress_bar=True,
                      enable_model_summary=True,
                      enable_checkpointing=False,
                      callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_metric")])

    with threadpool_limits(limits=1):
        trainer.fit(model, data_module)

    return trainer.callback_metrics["val_metric"].item()


@click.command()
@click.version_option()
@click.pass_context
@click.option('-d', '--device', default='cpu', show_default=True,
              help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('--database', show_default=True, default='sqlite:///cocr.db', help='optuna SQL database location')
@click.option('-n', '--name', show_default=True, default=str(uuid.uuid4()), help='trial identifier')
@click.option('-N', '--epochs', show_default=True, default=RECOGNITION_HYPER_PARAMS['epochs'], help='Number of epochs to train for')
@click.option('-S', '--samples', show_default=True, default=100, help='Number of samples')
@click.option('-w', '--workers', show_default=True, default=16, help='Number of dataloader workers')
@click.option('--pruning/--no-pruning', show_default=True, default=True, help='Enables/disables trial pruning')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--load-from-checkpoint', default=None, type=click.Path(exists=True), help='Path to checkpoint to load')
@click.option('--prompt-mode',
              show_default=True,
              type=click.Choice(['boxes',
                                 'curves',
                                 'both']),
              default='both',
              help='Sets line prompt sampling mode: `boxes` for boxes only, '
              '`curves` for curves only, and `both` for randomly switching '
              'between boxes and curves.')
def cli(ctx, device, seed, database, name, epochs, samples, workers, pruning,
        load_from_checkpoint, training_files, evaluation_files, prompt_mode):

    try:
        accelerator, device = to_ptl_device(device)
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    from functools import partial

    import torch
    import optuna
    from optuna.samplers import TPESampler

    torch.set_float32_matmul_precision('medium')

    if seed:
        from lightning.pytorch import seed_everything
        seed_everything(seed, workers=True)

    ground_truth = []

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    objective = partial(train_model,
                        checkpoint=load_from_checkpoint,
                        accelerator=accelerator,
                        device=device,
                        training_data=ground_truth,
                        evaluation_data=evaluation_files,
                        num_workers=workers,
                        prompt_mode=prompt_mode,
                        epochs=epochs)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10,
                                         n_warmup_steps=10) if pruning else optuna.pruners.NopPruner()

    print(f'database: {database} trial: {name}')

    study = optuna.create_study(direction="minimize",
                                pruner=pruner,
                                study_name=name,
                                storage=database,
                                load_if_exists=True,
                                sampler=TPESampler(constant_liar=True))

    study.optimize(objective, n_trials=samples)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    print(study.best_trial)


if __name__ == '__main__':
    cli()
