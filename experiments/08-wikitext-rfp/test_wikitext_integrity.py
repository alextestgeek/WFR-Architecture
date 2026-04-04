"""Проверки протокола: val детерминирован, train окна меняются между эпохами."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(ROOT))

from wikitext_integrity import (  # noqa: E402
    assert_train_windows_change_across_epochs,
    assert_val_holdout_reproducible,
    tensor_batches_fingerprint,
)
from wikitext_loader import DEFAULT_DATASET_DIR, WikiTextCharCorpus  # noqa: E402

requires_wikitext = pytest.mark.skipif(
    not DEFAULT_DATASET_DIR.is_dir(),
    reason=f"No dataset: {DEFAULT_DATASET_DIR}",
)


@requires_wikitext
def test_val_holdout_same_seed_same_tensors() -> None:
    c = WikiTextCharCorpus.from_hf_disk(max_vocab=64)
    dev = torch.device("cpu")
    fp = assert_val_holdout_reproducible(
        c, num_val_batches=2, batch_size=4, seq_len=32, val_seed=12345, device=dev
    )
    assert len(fp) == 64


@requires_wikitext
def test_train_epochs_differ() -> None:
    c = WikiTextCharCorpus.from_hf_disk(max_vocab=64)
    dev = torch.device("cpu")
    assert_train_windows_change_across_epochs(
        c, num_train_batches=2, batch_size=4, seq_len=32, train_seed=42, device=dev
    )


@requires_wikitext
def test_fingerprint_stable() -> None:
    c = WikiTextCharCorpus.from_hf_disk(max_vocab=64)
    dev = torch.device("cpu")
    a = c.make_val_batches(3, 4, 48, seed=999, device=dev)
    b = c.make_val_batches(3, 4, 48, seed=999, device=dev)
    assert tensor_batches_fingerprint(a) == tensor_batches_fingerprint(b)
