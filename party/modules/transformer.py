# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class TransformerSelfAttentionLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): this parameter is ignored in this layer.
            decoder_max_seq_len (int): maximum cache sequence length.
        """
        self.attn.setup_cache(batch_size, dtype, max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup on ``self.attn``.
        See :func:~party.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches on ``self.attn`` are enabled.
        See :func:~party.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.attn.cache_enabled

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[torch.Tensor]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


class DeformableCrossAttention(nn.Module):
    """
    Multi-scale deformable cross-attention operating on flattened encoder
    tokens with known per-level spatial shapes.

    Args:
        embed_dim (int): model embedding dimension.
        num_heads (int): number of query heads.
        num_levels (int): number of encoder feature levels.
        num_points (int): sampled points per level and head.
        num_reference_points (int): geometry reference points per sample.
        offset_scale (float): maximum offset magnitude in pixels.
    """
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int = 20,
        num_reference_points: int = 4,
        offset_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        if num_points < 1:
            raise ValueError('num_points must be >= 1')
        if num_reference_points < 1:
            raise ValueError('num_reference_points must be >= 1')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_reference_points = num_reference_points
        self.offset_scale = float(offset_scale)
        self.head_dim = embed_dim // num_heads

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.reference_weights = nn.Linear(embed_dim, num_heads * num_reference_points)
        self.default_reference_points = nn.Parameter(torch.zeros(num_reference_points, 2))
        nn.init.uniform_(self.default_reference_points, 0.1, 0.9)

    @staticmethod
    def _as_spatial_shapes(
        spatial_shapes: Union[torch.Tensor, List[Tuple[int, int]], Tuple[Tuple[int, int], ...]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(spatial_shapes, torch.Tensor):
            shapes = spatial_shapes.to(device=device, dtype=torch.long)
        else:
            shapes = torch.tensor(spatial_shapes, device=device, dtype=torch.long)
        if shapes.ndim != 2 or shapes.shape[1] != 2:
            raise ValueError('encoder_spatial_shapes must have shape [num_levels, 2].')
        return shapes

    def _prepare_reference_points(
        self,
        x: torch.Tensor,
        reference_points: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if reference_points is None:
            refs = self.default_reference_points.sigmoid().view(1, 1, self.num_reference_points, 2)
            refs = refs.expand(bsz, seq_len, -1, -1)
        else:
            if reference_points.ndim == 3:
                refs = reference_points[:, None, :, :]
            elif reference_points.ndim == 4:
                refs = reference_points
            else:
                raise ValueError('encoder_reference_points must have shape [b, r, 2] or [b, s, r, 2].')
            if refs.shape[0] != bsz:
                raise ValueError('encoder_reference_points batch size must match query batch size.')
            if refs.shape[1] == 1 and seq_len > 1:
                refs = refs.expand(-1, seq_len, -1, -1)
            elif refs.shape[1] != seq_len:
                raise ValueError('encoder_reference_points sequence dimension must be 1 or match query length.')
            if refs.shape[2] != self.num_reference_points:
                if refs.shape[2] > self.num_reference_points:
                    refs = refs[:, :, :self.num_reference_points, :]
                else:
                    pad = refs[:, :, -1:, :].expand(-1, -1, self.num_reference_points - refs.shape[2], -1)
                    refs = torch.cat([refs, pad], dim=2)
            refs = refs.clamp(0.0, 1.0)
        return refs.to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        encoder_input: torch.Tensor,
        *,
        encoder_spatial_shapes: Union[torch.Tensor, List[Tuple[int, int]], Tuple[Tuple[int, int], ...]],
        encoder_reference_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        shapes = self._as_spatial_shapes(encoder_spatial_shapes, device=x.device)

        values = self.v_proj(encoder_input).view(bsz, -1, self.num_heads, self.head_dim)

        refs = self._prepare_reference_points(x, encoder_reference_points)
        ref_mix = self.reference_weights(x).view(bsz, seq_len, self.num_heads, self.num_reference_points)
        ref_mix = torch.softmax(ref_mix, dim=-1)
        anchors = (ref_mix[..., None] * refs[:, :, None, :, :]).sum(dim=-2)

        offsets = self.sampling_offsets(x).view(
            bsz, seq_len, self.num_heads, self.num_levels, self.num_points, 2
        )
        attn_weights = self.attention_weights(x).view(
            bsz, seq_len, self.num_heads, self.num_levels * self.num_points
        )
        attn_weights = torch.softmax(attn_weights, dim=-1).view(
            bsz, seq_len, self.num_heads, self.num_levels, self.num_points
        )

        out = torch.zeros((bsz, seq_len, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        start = 0
        for lvl_idx, (height, width) in enumerate(shapes.tolist()):
            level_tokens = height * width
            level_vals = values[:, start:start + level_tokens]
            start += level_tokens

            level_vals = level_vals.view(bsz, height, width, self.num_heads, self.head_dim)
            level_vals = level_vals.permute(0, 3, 4, 1, 2).reshape(bsz * self.num_heads, self.head_dim, height, width)

            lvl_offsets = offsets[:, :, :, lvl_idx]
            normalizer = torch.tensor([max(width, 1), max(height, 1)],
                                      device=x.device,
                                      dtype=x.dtype).view(1, 1, 1, 1, 2)
            lvl_locs = anchors[:, :, :, None, :] + torch.tanh(lvl_offsets) * (self.offset_scale / normalizer)
            lvl_grid = lvl_locs.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            lvl_grid = lvl_grid.permute(0, 2, 1, 3, 4).reshape(bsz * self.num_heads, seq_len, self.num_points, 2)

            sampled = F.grid_sample(
                level_vals,
                lvl_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
            sampled = sampled.view(bsz, self.num_heads, self.head_dim, seq_len, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            weights = attn_weights[:, :, :, lvl_idx, :].unsqueeze(-1)
            out = out + (sampled * weights).sum(dim=3)

        out = out.view(bsz, seq_len, self.embed_dim)
        return self.out_proj(out)


class TransformerDeformableCrossAttentionLayer(nn.Module):
    """
    Cross-attention transformer layer using multi-scale deformable attention.
    """
    def __init__(
        self,
        attn: DeformableCrossAttention,
        mlp: nn.Module,
        *,
        ca_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        ca_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.ca_norm = ca_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.ca_scale = ca_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()
        self._cache_setup = False
        self._cache_enabled = False
        self._cached_encoder_input: Optional[torch.Tensor] = None
        self._cached_spatial_shapes: Optional[Union[torch.Tensor, List[Tuple[int, int]], Tuple[Tuple[int, int], ...]]] = None
        self._cached_reference_points: Optional[torch.Tensor] = None

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        del batch_size, dtype, encoder_max_seq_len, decoder_max_seq_len
        self._cache_setup = True
        self._cache_enabled = True
        self._cached_encoder_input = None
        self._cached_spatial_shapes = None
        self._cached_reference_points = None

    def caches_are_setup(self) -> bool:
        return self._cache_setup

    def caches_are_enabled(self) -> bool:
        return self._cache_enabled

    def reset_cache(self):
        self._cached_encoder_input = None
        self._cached_spatial_shapes = None
        self._cached_reference_points = None

    def _skip_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        return torch.all(mask, dim=-1, keepdim=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        encoder_spatial_shapes: Optional[Union[torch.Tensor, List[Tuple[int, int]], Tuple[Tuple[int, int], ...]]] = None,
        encoder_reference_points: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        del kwargs
        if encoder_input is None:
            if not self._cache_enabled or self._cached_encoder_input is None:
                return x
            encoder_input = self._cached_encoder_input
            encoder_spatial_shapes = self._cached_spatial_shapes
            if encoder_reference_points is None:
                encoder_reference_points = self._cached_reference_points
        elif self._cache_enabled:
            self._cached_encoder_input = encoder_input
            self._cached_spatial_shapes = encoder_spatial_shapes
            self._cached_reference_points = encoder_reference_points

        if encoder_spatial_shapes is None:
            raise ValueError('encoder_spatial_shapes is required for deformable cross-attention.')

        skip_mask = self._skip_mask(encoder_mask)
        attn_out = self.attn(
            self.ca_norm(x),
            encoder_input,
            encoder_spatial_shapes=encoder_spatial_shapes,
            encoder_reference_points=encoder_reference_points,
        )
        if skip_mask is not None:
            attn_out = attn_out.masked_fill(skip_mask, 0)

        h = self.ca_scale(attn_out) + x
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out = mlp_out.masked_fill(skip_mask, 0)

        return h + self.mlp_scale(mlp_out)


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, List[nn.Module], nn.ModuleList]): A single transformer Decoder layer, an
            nn.ModuleList of layers or a list of layers. It is recommended to use an nn.ModuleList.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~party.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~party.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~party.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (Union[nn.Linear, Callable]): Callable that applies a linear transformation to the output of
            the decoder.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[List[int]]): List of layers (indices) to include in the output

    Raises:
        AssertionError: num_layers is set and layer is a list
        AssertionError: num_layers is not set and layer is an nn.Module

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module], nn.ModuleList],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if isinstance(layers, nn.ModuleList):
            pass
        elif isinstance(layers, list):
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            layers = _get_clones(layers, num_layers)

        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.num_output_chunks = 0

        # attributes for KV caches during inference
        self.encoder_max_cache_seq_len = None
        self.decoder_max_cache_seq_len = None

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.num_output_chunks = num_output_chunks

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """
        Sets up key-value attention caches for inference. For each layer in ``self.layers``:
            - :class:`~party.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
            - :class:`~party.modules.TransformerDeformableCrossAttentionLayer` will cache encoder features.
            - :class:`~party.modules.FusionLayer` will use ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (Optional[int]): maximum encoder cache sequence length.
            decoder_max_seq_len (Optional[int]): maximum decoder cache sequence length.
        """

        has_encoder_layers = any(
            isinstance(m, TransformerDeformableCrossAttentionLayer) for m in self.modules()
        )
        has_decoder_layers = any(
            isinstance(m, TransformerSelfAttentionLayer) for m in self.modules()
        )

        if has_encoder_layers:
            if encoder_max_seq_len is not None:
                self.encoder_max_cache_seq_len = encoder_max_seq_len
            else:
                self.encoder_max_cache_seq_len = self.max_seq_len

        if has_decoder_layers:
            if decoder_max_seq_len is not None:
                self.decoder_max_cache_seq_len = decoder_max_seq_len
            else:
                self.decoder_max_cache_seq_len = self.max_seq_len

        for layer in self.layers:
            layer.setup_caches(
                batch_size,
                dtype,
                encoder_max_seq_len=self.encoder_max_cache_seq_len,
                decoder_max_seq_len=self.decoder_max_cache_seq_len,
            )

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.layers[0].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.layers[0].caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.

        Raises:
            RuntimeError: if KV-caches are not setup. Use :func:`~party.modules.TransformerDecoder.setup_caches` to
                setup caches first.
        """
        if not self.caches_are_enabled():
            raise RuntimeError(
                "Key value caches are not setup. Call model.setup_caches first."
            )

        for layer in self.layers:
            layer.reset_cache()

    @torch.compiler.disable
    def chunked_output(self, last_hidden_state: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply output projection in chunks. This should be applied in conjunction with
        :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss` as upcasting to fp32 is done there.

        To use this method, you should first call
        :func:`~party.modules.TransformerDecoder.set_num_output_chunks`.

        Args:
            last_hidden_state (torch.Tensor): last hidden state of the decoder, having shape
                [b, seq_len, embed_dim].

        Returns:
            List[torch.Tensor]: List of num_chunks output tensors, each with shape
                [b, seq_len/num_chunks, out_dim], where out_dim is usually the vocab size.
        """
        return [
            self.output(chunk)
            for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)
        ]

    def _validate_inputs(
        self,
        seq_len: int,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        """
        Validates inputs for ``forward``.
        Args:
            seq_len (int): Input tensor sequence length.
            mask (Optional[torch.Tensor]): Attention mask used for inference and for sequence packing.
            encoder_input (Optional[torch.Tensor]): Encoder input for cross-attention.
            encoder_mask (Optional[torch.Tensor]): Encoder attention mask for cross-embedding attention.
            input_pos (Optional[torch.Tensor]): Input tensor position IDs.

        Raises:
            ValueError: if seq_len of x is bigger than max_seq_len
            ValueError: if the model has caches which have been setup with self-attention layers and ``mask`` is not provided.
            ValueError: if the model has caches which have been setup with encoder layers and ``encoder_mask`` is not provided.
            ValueError: if the model has caches which have been setup ``input_pos`` is not provided.
        """

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        if self.caches_are_enabled():
            if mask is None:
                raise ValueError(
                    "KV-caches for self-attention layers are setup for inference mode, causal masks must be provided!"
                    " Use the `mask` arg to provide a causal mask."
                )

            if encoder_input is not None and encoder_mask is None:
                raise ValueError(
                    "KV-caches for cross-attention/fusion layers are setup for inference mode and you seem to be using"
                    " encoder_input, causal masks must be provided! Use the `encoder_mask` arg to provide a causal mask."
                )

            if input_pos is None:
                raise ValueError(
                    "KV-caches are setup for inference mode, input positions must be provided!"
                )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        encoder_spatial_shapes: Optional[Union[torch.Tensor, List[Tuple[int, int]], Tuple[Tuple[int, int], ...]]] = None,
        encoder_reference_points: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Used to mask the scores after the query-key multiplication
                and before the softmax. This parameter is required during inference if caches have been setup.
                Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                This parameter is required during inference if caches have been setup. Default is None.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` or a list of layer
                output tensors defined by ``output_hidden_states`` with the
                final output tensor appended to the list.

        Note:
            At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` should contain the positions of all of the tokens in the prompt.
            For a single-batch prompt, or a batch of prompts with identical lengths, this
            will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
            shorter prompts are left-padded and position ids are correspondingly right-shifted,
            thus positional ids should be of shape ``[b, padded_prompt_length]``.
            This is because we will need to retrieve the positional embeddings for each input id.
            In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
            the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
            ``input_pos`` will contain all the position ids up to the current token.

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        seq_len = tokens.shape[1]

        self._validate_inputs(
            seq_len,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                encoder_spatial_shapes=encoder_spatial_shapes,
                encoder_reference_points=encoder_reference_points,
                input_pos=input_pos,
            )

        # shape: [b, s, d]
        h = self.norm(h)

        if self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        # Output list if hidden states are requested, otherwise just the output
        # TODO: always output a list to have a consistent output type
        output = output if not hidden else [*hidden, output]
        return output
