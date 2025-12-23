from kraken.configs import TrainingConfig, RecognitionTrainingDataConfig


class PartyRecognitionTrainingConfig(TrainingConfig):
    """
    Base configuration for training a party model.

    Args:
    """
    def __init__(self, **kwargs):
        self.freeze_encoder = kwargs.pop('freeze_encoder', False)
        self.train_from_scratch = kwargs.pop('train_from_scratch', False)
        self.noisy_teacher_forcing = kwargs.pop('noisy_teacher_forcing', 0.1)
        self.label_smoothing = kwargs.pop('label_smoothing', 0.2)

        kwargs.setdefault('quit', 'fixed')
        kwargs.setdefault('epochs', 12)
        kwargs.setdefault('lrate', 5e-4)
        kwargs.setdefault('weight_decay', 1e-5)
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('cos_t_max', 12)
        kwargs.setdefault('cos_min_lr', 5e-6)
        kwargs.setdefault('warmup', 1000)
        kwargs.setdefault('accumulate_grad_batches', 4)
        kwargs.setdefault('augment', True)
        super().__init__(**kwargs)


class PartyRecognitionTrainingDataConfig(RecognitionTrainingDataConfig):
    """
    Base data configuration for a Party recognition model.
    """
    def __init__(self, **kwargs):
        self.val_batch_size = kwargs.pop('val_batch_size', None)
        self.image_size = kwargs.pop('image_size', (2560, 1920))
        self.prompt_mode = kwargs.pop('prompt_mode', 'both')

        kwargs.setdefault('batch_size', 16)

        super().__init__(**kwargs)
