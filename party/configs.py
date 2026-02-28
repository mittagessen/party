from kraken.configs import TrainingConfig, RecognitionTrainingDataConfig, RecognitionInferenceConfig

from party.fusion import default_adapter_ds_factors

DEFAULT_ENCODER_NAME = 'convnextv2_base.fcmae_ft_in22k_in1k'
DEFAULT_DECODER_NAME = 'mittagessen/bytellama-40m-oscar'
DEFAULT_DECODER_EMBED_DIM = 576
DEFAULT_FUSION_INTERVAL = 3


class PartyRecognitionInferenceConfig(RecognitionInferenceConfig):
    """
    Configuration for party model inference.
    """

    def __init__(self, **kwargs):
        self.prompt_mode = kwargs.pop('prompt_mode', None)
        self.max_generated_tokens = kwargs.pop('max_generated_tokens', 512)
        self.add_lang_token = kwargs.pop('add_lang_token', True)
        super().__init__(**kwargs)


class PartyRecognitionTrainingConfig(TrainingConfig):
    """
    Training configuration for the additive prompt party model.
    """

    def __init__(self, **kwargs):
        self.encoder_out_indices = tuple(kwargs.pop('encoder_out_indices', (2,)))
        if not self.encoder_out_indices:
            raise ValueError('encoder_out_indices must not be empty.')
        if tuple(sorted(self.encoder_out_indices)) != self.encoder_out_indices:
            raise ValueError('encoder_out_indices must be sorted in ascending order.')

        adapter_ds_factors = kwargs.pop('adapter_ds_factors', None)
        if adapter_ds_factors is None:
            adapter_ds_factors = default_adapter_ds_factors(len(self.encoder_out_indices))
        self.adapter_ds_factors = list(adapter_ds_factors)
        if len(self.adapter_ds_factors) != len(self.encoder_out_indices):
            raise ValueError('adapter_ds_factors must have the same length as encoder_out_indices.')
        if len(self.encoder_out_indices) == 1 and self.adapter_ds_factors != [1]:
            raise ValueError('single-scale encoder features require adapter_ds_factors=[1].')

        default_adapter_layers = 4 if len(self.encoder_out_indices) == 1 else 1
        self.adapter_num_layers = kwargs.pop('adapter_num_layers', default_adapter_layers)
        self.adapter_num_heads = kwargs.pop('adapter_num_heads', 8)

        self.freeze_encoder = kwargs.pop('freeze_encoder', False)
        self.train_from_scratch = kwargs.pop('train_from_scratch', False)
        self.noisy_teacher_forcing = kwargs.pop('noisy_teacher_forcing', 0.02)
        self.noisy_teacher_forcing_warmup = kwargs.pop('noisy_teacher_forcing_warmup', None)
        self.label_smoothing = kwargs.pop('label_smoothing', 0.0)
        self.lr_pretrained_mult = kwargs.pop('lr_pretrained_mult', 0.2)

        kwargs.setdefault('quit', 'fixed')
        kwargs.setdefault('epochs', 12)
        kwargs.setdefault('lrate', 5e-4)
        kwargs.setdefault('weight_decay', 1e-5)
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('cos_t_max', 12)
        kwargs.setdefault('cos_min_lr', 5e-6)
        kwargs.setdefault('warmup', 1000)
        if self.noisy_teacher_forcing_warmup is None:
            self.noisy_teacher_forcing_warmup = kwargs['warmup']
        kwargs.setdefault('accumulate_grad_batches', 4)
        kwargs.setdefault('augment', True)
        super().__init__(**kwargs)


class PartyRecognitionTrainingDataConfig(RecognitionTrainingDataConfig):
    """
    Base data configuration for a party recognition model.
    """

    def __init__(self, **kwargs):
        self.val_batch_size = kwargs.pop('val_batch_size', None)
        self.image_size = kwargs.pop('image_size', (2560, 1920))
        self.prompt_mode = kwargs.pop('prompt_mode', 'both')
        self.normalization = kwargs.pop('normalization', None)
        self.normalize_whitespace = kwargs.pop('normalize_whitespace', True)

        kwargs.setdefault('batch_size', 16)

        super().__init__(**kwargs)
