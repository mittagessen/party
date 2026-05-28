from kraken.configs import TrainingConfig, RecognitionTrainingDataConfig, RecognitionInferenceConfig

MODEL_VARIANTS = {
    # Tiny: 7M decoder with a 2-layer adapter and Swin Tiny encoder (~28M).
    'tiny': {
        'encoder': {
            'name': 'swin_tiny_patch4_window7_224.ms_in1k',
            'out_indices': (2,),
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
            'num_layers': 2,
            'num_heads': 8,
        },
        'fusion_interval': 3,
    },
    # Small: 16M decoder with a 2-layer adapter and Swin Small encoder (~49M).
    'small': {
        'encoder': {
            'name': 'swin_small_patch4_window7_224.ms_in1k',
            'out_indices': (2,),
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
            'num_layers': 2,
            'num_heads': 8,
        },
        'fusion_interval': 3,
    },
    # Base: 43M decoder with a 4-layer adapter and Swin Base encoder (~88M).
    'base': {
        'encoder': {
            'name': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            'out_indices': (2,),
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
            'num_layers': 4,
            'num_heads': 8,
        },
        'fusion_interval': 3,
    },
    # Large: ~164M decoder with a 4-layer adapter and Swin Large encoder (~197M).
    'large': {
        'encoder': {
            'name': 'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
            'out_indices': (2,),
        },
        'decoder': {
            'name': None,
            'embed_dim': 768,
            'num_heads': 12,
            'num_kv_heads': 4,
            'num_layers': 24,
            'intermediate_dim': 2048,
        },
        'adapter': {
            'num_layers': 4,
            'num_heads': 8,
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
        self.label_smoothing = kwargs.pop('label_smoothing', 0.0)
        self.proto_margin = kwargs.pop('proto_margin', 0.0)
        self.proto_temperature_init = kwargs.pop('proto_temperature_init', 10.0)
        # When set, load weights from a "conventional" party safetensors file
        # into the prototype model after construction, reinitializing the
        # prototype-specific layers (tok_embeddings, output head). Set by
        # cli/train.py when --load points at a non-prototype safetensors file.
        self.pretrained_weights_path = kwargs.pop('pretrained_weights_path', None)
        # Only consulted by the Lightning test loop: prepend language tokens
        # from the segmentation's language field when generating predictions.
        self.add_lang_token = kwargs.pop('add_lang_token', False)
        self.max_generated_tokens = kwargs.pop('max_generated_tokens', 512)

        kwargs.setdefault('quit', 'fixed')
        kwargs.setdefault('epochs', 32)
        kwargs.setdefault('lrate', 5e-4)
        kwargs.setdefault('momentum', 0.95)
        kwargs.setdefault('weight_decay', 1e-5)
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('cos_t_max', 32)
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
        self.normalization = kwargs.pop('normalization', 'NFC')
        self.normalize_whitespace = kwargs.pop('normalize_whitespace', True)

        kwargs.setdefault('batch_size', 16)
        kwargs.setdefault('augment', True)

        super().__init__(**kwargs)
