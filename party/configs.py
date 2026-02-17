from kraken.configs import TrainingConfig, RecognitionTrainingDataConfig, RecognitionInferenceConfig


class PartyRecognitionInferenceConfig(RecognitionInferenceConfig):
    """
    Configuration for party model inference.

    Args:
        prompt_mode (one of 'curves', 'boxes', or None, defaults to None):
            How to embed line positional prompts. If None, the prompt mode is
            determined from the input segmentation type ('baselines' ->
            'curves', 'bbox' -> 'boxes'). If set explicitly, lines will be
            converted to the requested format when possible. Setting 'curves'
            with a bounding box segmentation will raise a ValueError.
        max_generated_tokens (int, defaults to 512):
            Maximum number of tokens to generate per line.
        add_lang_token (bool, defaults to True):
            Prepend language tokens from the segmentation's language field to
            condition the decoder on the input language.
    """
    def __init__(self, **kwargs):
        self.prompt_mode = kwargs.pop('prompt_mode', None)
        self.max_generated_tokens = kwargs.pop('max_generated_tokens', 512)
        self.add_lang_token = kwargs.pop('add_lang_token', True)
        super().__init__(**kwargs)


class PartyRecognitionTrainingConfig(TrainingConfig):
    """
    Base configuration for training a party model.

    Args:
    """
    def __init__(self, **kwargs):
        self.freeze_encoder = kwargs.pop('freeze_encoder', False)
        self.train_from_scratch = kwargs.pop('train_from_scratch', False)
        self.prompt_num_samples = kwargs.pop('prompt_num_samples', 384)
        self.noisy_teacher_forcing = kwargs.pop('noisy_teacher_forcing', 0.05)
        self.label_smoothing = kwargs.pop('label_smoothing', 0.0)
        # LR multiplier for pretrained components (encoder + decoder base layers)
        self.lr_pretrained_mult = kwargs.pop('lr_pretrained_mult', 0.1)

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
        self.normalization = kwargs.pop('normalization', None)
        self.normalize_whitespace = kwargs.pop('normalize_whitespace', True)

        kwargs.setdefault('batch_size', 16)

        super().__init__(**kwargs)
