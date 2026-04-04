"""
Смоук-тесты Exp 08: корпус, один шаг forward, короткий градиент.

Без скачанного WikiText тесты пропускаются (``pytest -m "not slow"`` или просто pytest).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(ROOT / "experiments" / "06-rfp-v0"))

from phase0_best_config import PHASE0_FREQ_BALANCED  # noqa: E402
from wfr.core import WFRNetwork  # noqa: E402
from wfr_lm import WFRLM  # noqa: E402

from wikitext_loader import DEFAULT_DATASET_DIR, WikiTextCharCorpus  # noqa: E402

DATASET_OK = DEFAULT_DATASET_DIR.is_dir()
requires_wikitext = pytest.mark.skipif(
    not DATASET_OK,
    reason=f"Dataset missing: {DEFAULT_DATASET_DIR} — run: python data/hf/download_wikitext2.py",
)


@requires_wikitext
def test_corpus_vocab_and_windows() -> None:
    c = WikiTextCharCorpus.from_hf_disk(max_vocab=128)
    assert c.vocab_size >= 8
    assert len(c.train_text) > len(c.val_text) or len(c.train_text) > 1000
    g = torch.Generator()
    g.manual_seed(0)
    dev = torch.device("cpu")
    b = c.sample_train_batch(4, 32, g, dev)
    assert b.shape == (4, 32)
    vb = c.make_val_batches(2, 4, 32, seed=99, device=dev)
    assert len(vb) == 2


@requires_wikitext
def test_one_optimizer_step_cuda_if_available() -> None:
    from run_wikitext_train import train_wikitext_run  # noqa: E402

    corpus = WikiTextCharCorpus.from_hf_disk(max_vocab=64)
    r = train_wikitext_run(
        corpus,
        epochs=1,
        seq_len=32,
        batch_size=4,
        num_train_batches=2,
        num_val_batches=2,
        use_rfp=False,
        rfp_interval=8,
        rfp_version="v0",
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        save_png=True,
        plot_path=_EXP / "outputs" / "_smoke_wikitext_baseline.png",
        epoch_json_path=_EXP / "outputs" / "_smoke_epoch_log.json",
    )
    assert r.metrics.best_val_ce < float("inf")
    assert len(r.history_train_ce) == 1
    assert (_EXP / "outputs" / "_smoke_epoch_log.json").is_file()
