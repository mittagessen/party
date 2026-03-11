from kraken.configs import TrainingConfig, RecognitionTrainingDataConfig, RecognitionInferenceConfig

MODEL_VARIANTS = {
    # Tiny: ~10M LM params, ConvNeXt v2 Nano encoder (~15.6M)
    'tiny': {
        'encoder': {
            'name': 'convnextv2_nano.fcmae_ft_in22k_in1k',
            'out_indices': (1, 2, 3),
        },
        'decoder': {
            'name': 'mittagessen/bytellama-7m-cc',
            'embed_dim': 192,
            'num_heads': 6,
            'num_kv_heads': 2,
            'num_layers': 18,
            'intermediate_dim': 512,
        },
        'adapter': {
            'output_ds_factors': (2, 4, 4),
            'hidden_dim': 128,
            'num_heads': 4,
            'num_encoder_layers': 1,
            'dim_feedforward': 512,
            'fusion_depth': 2,
        },
        'fusion_interval': 3,
    },
    # Small: ~21M LM params, ConvNeXt v2 Tiny encoder (~28.6M)
    'small': {
        'encoder': {
            'name': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
            'out_indices': (1, 2, 3),
        },
        'decoder': {
            'name': 'mittagessen/bytellama-16m-cc',
            'embed_dim': 288,
            'num_heads': 9,
            'num_kv_heads': 3,
            'num_layers': 18,
            'intermediate_dim': 768,
        },
        'adapter': {
            'output_ds_factors': (2, 4, 4),
            'hidden_dim': 192,
            'num_heads': 8,
            'num_encoder_layers': 1,
            'dim_feedforward': 768,
            'fusion_depth': 2,
        },
        'fusion_interval': 3,
    },
    # Base: ~40M LM params (pretrained), ConvNeXt v2 Base encoder (~89M)
    'base': {
        'encoder': {
            'name': 'convnextv2_base.fcmae_ft_in22k_in1k',
            'out_indices': (1, 2, 3),
        },
        'decoder': {
            'name': 'mittagessen/bytellama-43m-cc',
            'embed_dim': 576,
            'num_heads': 9,
            'num_kv_heads': 3,
            'num_layers': 12,
            'intermediate_dim': 1536,
        },
        'adapter': {
            'output_ds_factors': (2, 4, 4),
            'hidden_dim': 256,
            'num_heads': 8,
            'num_encoder_layers': 1,
            'dim_feedforward': 1024,
            'fusion_depth': 2,
        },
        'fusion_interval': 3,
    },
    # Large: ~164M LM params, ConvNeXt v2 Large encoder (~198M)
    'large': {
        'encoder': {
            'name': 'convnextv2_large.fcmae_ft_in22k_in1k',
            'out_indices': (1, 2, 3),
        },
        'decoder': {
            'name': None,
            'embed_dim': 576,
            'num_heads': 9,
            'num_kv_heads': 3,
            'num_layers': 30,
            'intermediate_dim': 1536,
        },
        'adapter': {
            'output_ds_factors': (2, 4, 4),
            'hidden_dim': 256,
            'num_heads': 8,
            'num_encoder_layers': 1,
            'dim_feedforward': 1024,
            'fusion_depth': 2,
        },
        'fusion_interval': 3,
    },
}


def get_model_variant(name: str) -> dict:
    if name not in MODEL_VARIANTS:
        raise ValueError(f'Unknown model variant {name!r}. Available: {list(MODEL_VARIANTS)}')
    return MODEL_VARIANTS[name]


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
        self.model_variant = kwargs.pop('model_variant', 'base')

        self.freeze_encoder = kwargs.pop('freeze_encoder', False)
        self.train_from_scratch = kwargs.pop('train_from_scratch', False)
        self.noisy_teacher_forcing = kwargs.pop('noisy_teacher_forcing', 0.02)
        self.noisy_teacher_forcing_warmup = kwargs.pop('noisy_teacher_forcing_warmup', None)
        self.label_smoothing = kwargs.pop('label_smoothing', 0.0)
        # LR multiplier for pretrained encoder components.
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
    Base data configuration for a Party recognition model.
    """
    def __init__(self, **kwargs):
        self.val_batch_size = kwargs.pop('val_batch_size', None)
        self.image_size = kwargs.pop('image_size', (2560, 1920))
        self.prompt_mode = kwargs.pop('prompt_mode', 'both')
        self.normalization = kwargs.pop('normalization', None)
        self.normalize_whitespace = kwargs.pop('normalize_whitespace', True)
        self.prompt_corruption = kwargs.pop('prompt_corruption', False)

        kwargs.setdefault('batch_size', 16)

        super().__init__(**kwargs)
