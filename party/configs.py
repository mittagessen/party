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


def _popped(kwargs: dict, key: str, default):
    """
    ``dict.pop`` treating an explicit None as "not provided". The train CLI
    builds its config from ``ctx.params``, where unset options are present and
    None, so a plain pop would hand back None instead of the default.
    """
    value = kwargs.pop(key, None)
    return default if value is None else value


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
            Canvas width in tokens, clamped to the decoder's maximum sequence
            length. Each line is decoded on a canvas of this many slots and
            terminated by the first predicted EOS. Decoding cost scales with it,
            so it is worth setting from the data.
        add_lang_token (bool, defaults to True):
            Prepend language tokens from the segmentation's language field to
            condition the decoder on the input language.
        diffusion_steps (int, defaults to 4):
            Number of denoising passes over the canvas. Each predicts every
            position in parallel and commits the most confident ``1/remaining``
            of what is still masked, so the canvas fills in this many passes
            whatever the line length. 1 decodes in a single pass.
        confidence_threshold (float or None, defaults to None):
            Optionally also commit any position above this probability, on top
            of the per-pass quota. Off by default, as the prototype head's
            logit scale is learned and a fixed threshold would need
            recalibrating per fine-tune.
        truncate_at_eos (bool, defaults to True):
            Drop the canvas tail from later passes once every line in the batch
            has committed an EOS.
    """
    def __init__(self, **kwargs):
        self.prompt_mode = kwargs.pop('prompt_mode', None)
        self.max_generated_tokens = kwargs.pop('max_generated_tokens', 512)
        self.add_lang_token = kwargs.pop('add_lang_token', True)
        self.diffusion_steps = kwargs.pop('diffusion_steps', 4)
        self.confidence_threshold = kwargs.pop('confidence_threshold', None)
        self.truncate_at_eos = kwargs.pop('truncate_at_eos', True)
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
        self.resize = kwargs.pop('resize', 'union')
        self.label_smoothing = kwargs.pop('label_smoothing', 0.0)
        self.noisy_teacher_forcing = kwargs.pop('noisy_teacher_forcing', 0.0)
        self.noisy_teacher_forcing_warmup = kwargs.pop('noisy_teacher_forcing_warmup', 0)
        self.proto_margin = kwargs.pop('proto_margin', 0.0)
        self.proto_temperature_init = kwargs.pop('proto_temperature_init', 10.0)
        self.pretrained_weights_path = kwargs.pop('pretrained_weights_path', None)
        self.add_lang_token = kwargs.pop('add_lang_token', False)
        self.max_generated_tokens = kwargs.pop('max_generated_tokens', 512)

        # masked diffusion objective, see party/diffusion.py. The mask rate
        # ceiling ramps from `mask_rate_max_start` to `mask_rate_max` over
        # `mask_rate_warmup` steps, starting near the teacher-forced regime.
        self.mask_rate_min = _popped(kwargs, 'mask_rate_min', 0.0)
        self.mask_rate_max = _popped(kwargs, 'mask_rate_max', 1.0)
        self.mask_rate_max_start = _popped(kwargs, 'mask_rate_max_start', 0.15)
        self.mask_rate_warmup = _popped(kwargs, 'mask_rate_warmup', 4000)
        self.stratified_t = _popped(kwargs, 'stratified_t', True)
        # the window over which each query may see future positions grows from
        # causal to bidirectional. 0 goes straight to bidirectional.
        self.attn_anneal_steps = _popped(kwargs, 'attn_anneal_steps', 3000)
        # most slots of a full canvas are PAD, which would otherwise dominate
        self.pad_loss_weight = _popped(kwargs, 'pad_loss_weight', 0.05)
        # probability of keeping a line's language tokens as prompt; the rest
        # are masked and supervised, keeping language prediction available
        self.lang_prompt_p = _popped(kwargs, 'lang_prompt_p', 0.5)
        # fraction of steps trained on the old autoregressive objective, which
        # keeps a fallback decoder alive in the same weights
        self.ar_consistency_p = _popped(kwargs, 'ar_consistency_p', 0.05)
        # cycled deterministically; random rates would make val_metric a
        # high-variance estimator and checkpoint selection noisy
        self.val_mask_rates = _popped(kwargs, 'val_mask_rates', (0.25, 0.5, 0.75))
        # decoding knobs, used by `party test` to sweep the step budget
        self.diffusion_steps = _popped(kwargs, 'diffusion_steps', 4)
        self.confidence_threshold = kwargs.pop('confidence_threshold', None)
        self.truncate_at_eos = _popped(kwargs, 'truncate_at_eos', True)

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
