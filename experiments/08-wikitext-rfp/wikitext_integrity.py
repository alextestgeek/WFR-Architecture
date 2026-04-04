"""
Проверки протокола Exp 08: без «подкрутки» val, воспроизводимость holdout, отпечаток батчей.

Используются в ``run_wikitext_train.py`` и в тестах.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from wikitext_loader import WikiTextCharCorpus


def tensor_batches_fingerprint(batches: list[torch.Tensor]) -> str:
    """SHA256 по байтам тензоров батчей (для фиксации val holdout в манифесте)."""
    h = hashlib.sha256()
    for b in batches:
        h.update(b.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_val_holdout_reproducible(
    corpus: "WikiTextCharCorpus",
    *,
    num_val_batches: int,
    batch_size: int,
    seq_len: int,
    val_seed: int,
    device: torch.device,
) -> str:
    """Дважды строим val из одного ``val_seed`` — отпечатки должны совпадать."""
    a = corpus.make_val_batches(num_val_batches, batch_size, seq_len, val_seed, device)
    b = corpus.make_val_batches(num_val_batches, batch_size, seq_len, val_seed, device)
    fa = tensor_batches_fingerprint(a)
    fb = tensor_batches_fingerprint(b)
    if fa != fb:
        raise AssertionError("val holdout not reproducible: fingerprint mismatch")
    return fa


def assert_train_windows_change_across_epochs(
    corpus: "WikiTextCharCorpus",
    *,
    num_train_batches: int,
    batch_size: int,
    seq_len: int,
    train_seed: int,
    device: torch.device,
) -> None:
    """Эпоха 0 и эпоха 1 должны давать разные train-батчи (fresh windows)."""
    t0 = corpus.make_train_batches_for_epoch(
        num_train_batches, batch_size, seq_len, 0, train_seed, device
    )
    t1 = corpus.make_train_batches_for_epoch(
        num_train_batches, batch_size, seq_len, 1, train_seed, device
    )
    if tensor_batches_fingerprint(t0) == tensor_batches_fingerprint(t1):
        raise AssertionError(
            "train batches for epoch 0 and 1 are identical — check make_train_batches_for_epoch seeding"
        )
