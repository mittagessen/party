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
Prototype/vocabulary adaptation of trained party models toward new datasets.
"""
import logging

import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Optional, TypeAlias

from party.tokenizer import CodePointTokenizer, CODEPOINT_OFFSET

if TYPE_CHECKING:
    from lightning.fabric import Fabric
    from party.party import PartyModel

logger = logging.getLogger(__name__)

__all__ = ['RowPlan', 'AdaptationReport', 'resolve_target_vocab',
           'rebuild_prototype_table', 'assert_table_consistency',
           'collect_support_prototypes', 'recalibrate_temperature',
           'refine_prototypes', 'adapt_model']

ResizePolicy: TypeAlias = Literal['union', 'new']
IGNORE_INDEX = -100
TEMPERATURE_CANDIDATES = (1., 2., 4., 6., 8., 10., 15., 20., 30., 50., 75., 100.)


@dataclass
class RowPlan:
    """
    Resolved layout of the new code-point region of the prototype table.

    ``actions`` has one entry per code-point row of ``new_tokenizer`` (in token
    order, starting at ``CODEPOINT_OFFSET``); each is either ``('copy',
    base_token_id)`` to carry a base prototype over or ``('seed', codepoint)``
    for a row that must be initialized by the caller.
    """
    actions: list[tuple[str, int]]
    new_tokenizer: CodePointTokenizer

    @property
    def seeded_codepoints(self) -> list[int]:
        return [v for kind, v in self.actions if kind == 'seed']


def resolve_target_vocab(base_tokenizer: CodePointTokenizer,
                         target_codepoints: Sequence[int],
                         resize: ResizePolicy) -> RowPlan:
    """
    Recomputes the code-point table for the ``resize`` policy.

    - ``union``: keep every base code point in its existing order, append
      target code points the base model lacks.
    - ``new``: keep only target code points (in target order); base prototypes
      are carried over where the code point recurs, the rest are dropped.
    """
    target = list(dict.fromkeys(int(c) for c in target_codepoints))

    if resize == 'union':
        ordered = list(base_tokenizer.codepoints)
        ordered += [c for c in target if base_tokenizer.token_for_codepoint(c) is None]
    elif resize == 'new':
        ordered = target
    else:
        raise ValueError(f'Invalid resize value {resize}.')

    new_tokenizer = CodePointTokenizer(codepoints=ordered, frozen=True)
    actions: list[tuple[str, int]] = []
    for cp in ordered:
        base_tid = base_tokenizer.token_for_codepoint(cp)
        if base_tid is not None:
            actions.append(('copy', base_tid))
        else:
            actions.append(('seed', cp))
    return RowPlan(actions=actions, new_tokenizer=new_tokenizer)


@torch.no_grad()
def rebuild_prototype_table(model: 'PartyModel',
                            plan: RowPlan,
                            seed_vectors: dict[int, torch.Tensor]) -> None:
    """
    Resizes the tied prototype/embedding table to match ``plan`` in place.

    Special and language-token rows (``< CODEPOINT_OFFSET``) are invariant. The
    resized table is re-tied into both the decoder embedding and the prototype
    head, and the tokenizer/metadata are updated.

    Args:
        model: The model whose prototype head and decoder embedding are resized
            in place. Its ``tokenizer`` and ``user_metadata['tokenizer']`` are
            replaced with ``plan.new_tokenizer`` on success.
        plan: The resolved code-point layout (from :func:`resolve_target_vocab`)
            describing, per new code-point row, whether to copy an existing base
            prototype (``'copy'``) or seed a fresh one (``'seed'``).
        seed_vectors: Initial prototype vectors for the seeded code points,
            keyed by code point. Must contain an entry for every code point in
            ``plan.seeded_codepoints``; a missing entry raises ``ValueError``.
            Vectors are cast to the table's device/dtype before assignment.
    """
    head = model._prototype_head()
    old = head.prototypes.weight
    device, dtype, dim = old.device, old.dtype, old.shape[1]
    new_vocab = plan.new_tokenizer.vocab_size

    new_weight = torch.empty(new_vocab, dim, device=device, dtype=dtype)
    new_weight[:CODEPOINT_OFFSET] = old[:CODEPOINT_OFFSET]
    for i, (kind, val) in enumerate(plan.actions):
        row = CODEPOINT_OFFSET + i
        if kind == 'copy':
            new_weight[row] = old[val]
        elif kind == 'seed':
            if val not in seed_vectors:
                raise ValueError(f'Missing seed vector for code point U+{val:04X}.')
            new_weight[row] = seed_vectors[val].to(device=device, dtype=dtype)
        else:
            raise ValueError(f'Unknown row action {kind!r}.')

    resized = nn.Embedding(new_vocab, dim, device=device, dtype=dtype)
    resized.weight = nn.Parameter(new_weight)
    head.tie_embeddings(resized)
    model.nn['decoder'].tok_embeddings = resized
    model.tokenizer = plan.new_tokenizer
    model.user_metadata['tokenizer'] = model.tokenizer.save()


def _shift_targets(tokens: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Left-shifts tokens to teacher-forcing targets, flattened to [b*s]."""
    pad = torch.full((tokens.shape[0], 1), ignore_index,
                     dtype=tokens.dtype, device=tokens.device)
    return torch.hstack((tokens[..., 1:], pad)).reshape(-1)


def _accumulate_support(hidden: torch.Tensor,
                        targets: torch.Tensor,
                        wanted: set[int],
                        sums: dict[int, torch.Tensor],
                        counts: dict[int, int]) -> None:
    """Adds per-target-token hidden vectors into running sum/count maps."""
    for tid in wanted:
        mask = targets == tid
        c = int(mask.sum())
        if c == 0:
            continue
        s = hidden[mask].sum(dim=0)
        sums[tid] = s if tid not in sums else sums[tid] + s
        counts[tid] = counts.get(tid, 0) + c


def _best_temperature(cos: torch.Tensor,
                      targets: torch.Tensor,
                      candidates: Sequence[float]) -> float:
    """Grid-searches the temperature minimizing cross-entropy on cos-logits."""
    best_t, best_loss = float(candidates[0]), float('inf')
    for t in candidates:
        loss = float(F.cross_entropy(t * cos, targets))
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def _move(batch, fabric):
    """Moves a support batch onto the fabric device, unpacking model inputs."""
    batch = fabric.to_device(batch)
    return batch['tokens'], batch['image'], batch['curves'], batch['boxes']


@torch.inference_mode()
def collect_support_prototypes(model: 'PartyModel',
                               support_loader,
                               target_token_ids: set[int],
                               fabric: 'Fabric',
                               *,
                               ignore_index: int = IGNORE_INDEX):
    """
    Means of prototype-space hidden states over the entire support set, bucketed
    by the target token they predict. Returns ``(means, counts)`` keyed by token
    id.
    """
    head = model._prototype_head()
    dim = head.prototypes.embedding_dim
    wanted = {int(t) for t in target_token_ids}
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    captured: dict[str, torch.Tensor] = {}

    handle = head.register_forward_pre_hook(lambda m, a: captured.__setitem__('h', a[0]))
    was_training = model.training
    model.eval()
    try:
        for batch in support_loader:
            tokens, image, curves, boxes = _move(batch, fabric)
            targets = _shift_targets(tokens, ignore_index)
            tokens_in = tokens.masked_fill(tokens == ignore_index, 0)
            captured.clear()
            with fabric.autocast():
                model(tokens=tokens_in, encoder_input=image,
                      encoder_curves=curves, encoder_boxes=boxes)
            hidden = captured['h'].reshape(-1, dim).float()
            _accumulate_support(hidden, targets, wanted, sums, counts)
    finally:
        handle.remove()
        if was_training:
            model.train()

    means = {t: sums[t] / counts[t] for t in sums}
    return means, counts


@torch.inference_mode()
def recalibrate_temperature(model: 'PartyModel',
                            support_loader,
                            fabric: 'Fabric',
                            *,
                            max_positions: int = 4096,
                            candidates: Sequence[float] = TEMPERATURE_CANDIDATES,
                            ignore_index: int = IGNORE_INDEX) -> float:
    """
    Sets the head temperature to the value minimizing support cross-entropy and
    returns it. No-op return of the current value if no support is available.

    Args:
        model: The model whose prototype head temperature is recalibrated in
            place.
        support_loader: Loader over the support set; batches must carry
            ``tokens``, ``image``, ``curves`` and ``boxes``.
        fabric: Fabric providing the device and autocast precision the support
            passes run under.
        max_positions: Upper bound on the number of (non-ignored) target
            positions gathered for the grid search; iteration stops once it is
            reached.
        candidates: Temperature values to search over; the one minimizing
            cross-entropy on the cosine logits is selected.
        ignore_index: Target value marking padding/ignored positions, excluded
            from the search.

    Returns:
        The selected temperature, or the current head temperature unchanged if
        the support set yielded no valid positions.
    """
    head = model._prototype_head()
    dim = head.prototypes.embedding_dim
    captured: dict[str, torch.Tensor] = {}
    handle = head.register_forward_pre_hook(lambda m, a: captured.__setitem__('h', a[0]))
    hs: list[torch.Tensor] = []
    tgs: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        total = 0
        for batch in support_loader:
            if total >= max_positions:
                break
            tokens, image, curves, boxes = _move(batch, fabric)
            targets = _shift_targets(tokens, ignore_index)
            tokens_in = tokens.masked_fill(tokens == ignore_index, 0)
            captured.clear()
            with fabric.autocast():
                model(tokens=tokens_in, encoder_input=image,
                      encoder_curves=curves, encoder_boxes=boxes)
            hidden = captured['h'].reshape(-1, dim).float()
            valid = targets != ignore_index
            hs.append(hidden[valid])
            tgs.append(targets[valid])
            total += int(valid.sum())
    finally:
        handle.remove()
        if was_training:
            model.train()

    if not hs:
        return float(head.temperature.item())
    hidden = torch.cat(hs)[:max_positions]
    targets = torch.cat(tgs)[:max_positions]
    proto = F.normalize(head.prototypes.weight.float(), dim=-1)
    cos = F.normalize(hidden, dim=-1) @ proto.T
    best_t = _best_temperature(cos, targets, candidates)
    head.temperature.data.fill_(best_t)
    return best_t


def refine_prototypes(model: 'PartyModel',
                      support_loader,
                      fabric: 'Fabric',
                      *,
                      steps: int,
                      lr: float = 1e-3,
                      ignore_index: int = IGNORE_INDEX) -> int:
    """
    Backbone-frozen refinement of the tied prototype table + temperature for up
    to ``steps`` optimizer steps. Returns the number of steps taken.

    Only the tied prototype/embedding table and the head temperature are left
    trainable; every other parameter's ``requires_grad`` is forced off for the
    duration and restored on return.

    Args:
        model: The model whose prototype table and temperature are refined in
            place.
        support_loader: Loader over the support set; batches must carry
            ``tokens``, ``image``, ``curves`` and ``boxes``. Cycled as many
            times as needed to reach ``steps``.
        fabric: Fabric providing the device, autocast precision and optimizer
            setup for the refinement loop.
        steps: Maximum number of optimizer steps to run.
        lr: AdamW learning rate for the table and temperature.
        ignore_index: Target value marking padding/ignored positions, excluded
            from the cross-entropy loss.

    Returns:
        The number of optimizer steps actually taken (normally ``steps``; fewer
        only if the support loader yields no batches at all).
    """
    head = model._prototype_head()
    table = model.nn['decoder'].tok_embeddings.weight
    trainable = {id(table), id(head.temperature)}

    saved = {}
    for name, p in model.named_parameters():
        saved[name] = p.requires_grad
        p.requires_grad = id(p) in trainable

    optim = torch.optim.AdamW([table, head.temperature], lr=lr)
    optim = fabric.setup_optimizers(optim)
    was_training = model.training
    model.train()
    done = 0
    try:
        while done < steps:
            for batch in support_loader:
                if done >= steps:
                    break
                tokens, image, curves, boxes = _move(batch, fabric)
                targets = _shift_targets(tokens, ignore_index)
                tokens_in = tokens.masked_fill(tokens == ignore_index, 0)
                with fabric.autocast():
                    logits = model(tokens=tokens_in, encoder_input=image,
                                   encoder_curves=curves, encoder_boxes=boxes)
                    logits = logits.reshape(-1, logits.shape[-1])
                    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
                optim.zero_grad()
                fabric.backward(loss)
                optim.step()
                done += 1
    finally:
        for name, p in model.named_parameters():
            p.requires_grad = saved[name]
        if not was_training:
            model.eval()
    return done


@dataclass
class AdaptationReport:
    """
    Summary of a single :func:`adapt_model` run.

    Attributes:
        vocab_before: Tokenizer vocabulary size before adaptation.
        vocab_after: Tokenizer vocabulary size after the table was rebuilt.
        added: Code points that were newly seeded into the table.
        dropped: Base code points discarded by the ``new`` resize policy.
        seeded_support: For each newly seeded code point that the support set
            covered, the number of support positions averaged into its
            prototype.
        temperature_before: Head temperature before adaptation.
        temperature_after: Head temperature after recalibration/refinement
            (equal to ``temperature_before`` when neither ran).
        refine_steps: Number of prototype-refinement optimizer steps taken.
    """
    vocab_before: int
    vocab_after: int
    added: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    seeded_support: dict[int, int] = field(default_factory=dict)
    temperature_before: float = 0.0
    temperature_after: float = 0.0
    refine_steps: int = 0


def _codepoint_centroid(weight: torch.Tensor) -> torch.Tensor:
    """Transient placeholder seed: mean of existing code-point prototype rows."""
    rows = weight[CODEPOINT_OFFSET:] if weight.shape[0] > CODEPOINT_OFFSET else weight
    return rows.mean(dim=0).detach()


@torch.no_grad()
def adapt_model(model: 'PartyModel',
                target_codepoints: Sequence[int],
                *,
                resize: ResizePolicy,
                support_loader: 'DataLoader' = None,
                recalibrate: bool = True,
                refine_steps: int = 0,
                refine_lr: float = 1e-3,
                fabric: Optional['Fabric'] = None) -> AdaptationReport:
    """
    Grows/reindexes the prototype table for ``target_codepoints``.

    With a ``support_loader`` (``party adapt``), every new prototype is the mean
    of its support-set hidden states; temperature is recalibrated and the table
    optionally refined for ``refine_steps`` backbone-frozen steps. The support
    passes run on the ``fabric``'s device and precision, so a ``fabric`` is
    required whenever a ``support_loader`` is given. Without one (``train
    --resize`` growth), new rows keep the neutral placeholder and are optimized
    by the subsequent training run.

    Args:
        model: The model adapted in place (table, tokenizer, head temperature).
        target_codepoints: Code points the adapted model should cover;
            duplicates and ordering are handled by :func:`resolve_target_vocab`.
        resize: Vocabulary policy. ``union`` keeps all base code points and
            appends the missing targets; ``new`` keeps only the targets,
            carrying base prototypes over where a code point recurs.
        support_loader: Optional loader over the support set used to seed new
            prototypes from real hidden states and to recalibrate/refine. When
            ``None``, new rows keep the neutral placeholder.
        recalibrate: Whether to recalibrate the head temperature on the support
            set (ignored without a ``support_loader``).
        refine_steps: Number of backbone-frozen refinement steps to run on the
            support set; ``0`` disables refinement.
        refine_lr: Learning rate for prototype refinement.
        fabric: Fabric object defining how the refinement is run. Required
            whenever ``support_loader`` is given.

    Returns:
        An :class:`AdaptationReport` summarizing the vocabulary changes,
        support coverage of seeded prototypes, temperature and refinement.

    Raises:
        ValueError: If a ``support_loader`` is given without a ``fabric``, or if
            the support set produced no examples for a code point it was
            expected to cover.
    """
    if support_loader is not None and fabric is None:
        raise ValueError('A fabric is required to run the support set.')
    base_tok = model.tokenizer
    head = model._prototype_head()
    temp_before = float(head.temperature.item())

    plan = resolve_target_vocab(base_tok, target_codepoints, resize)
    seeded_cps = plan.seeded_codepoints
    dropped = [cp for cp in base_tok.codepoints
               if plan.new_tokenizer.token_for_codepoint(cp) is None]

    # The table must exist (new rows present) before the support pass can run;
    # fill new rows with a neutral placeholder that the support means overwrite.
    placeholder = _codepoint_centroid(head.prototypes.weight)
    rebuild_prototype_table(model, plan, {cp: placeholder.clone() for cp in seeded_cps})

    report = AdaptationReport(vocab_before=base_tok.vocab_size,
                              vocab_after=model.tokenizer.vocab_size,
                              added=list(seeded_cps),
                              dropped=dropped,
                              temperature_before=temp_before,
                              temperature_after=temp_before)

    if seeded_cps and support_loader is not None:
        new_tok = model.tokenizer
        tid_to_cp = {new_tok.token_for_codepoint(cp): cp for cp in seeded_cps}
        means, counts = collect_support_prototypes(model, support_loader,
                                                   set(tid_to_cp), fabric)
        proto_w = head.prototypes.weight
        uncovered = []
        for tid, cp in tid_to_cp.items():
            if counts.get(tid, 0) > 0:
                proto_w.data[tid] = means[tid].to(proto_w.dtype)
                report.seeded_support[cp] = counts[tid]
            else:
                uncovered.append(cp)
        if uncovered:
            raise ValueError(
                f'Support set produced no examples for {len(uncovered)} code '
                f'point(s) {"".join(chr(c) for c in uncovered)} that it should '
                f'contain. Check the support data covers every line.')

    if recalibrate and support_loader is not None:
        report.temperature_after = recalibrate_temperature(model, support_loader, fabric)

    if refine_steps > 0 and support_loader is not None:
        report.refine_steps = refine_prototypes(model, support_loader, fabric,
                                                steps=refine_steps, lr=refine_lr)
        report.temperature_after = float(head.temperature.item())

    return report
