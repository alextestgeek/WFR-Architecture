"""
Unit-тесты для tier_checks и детерминизма схемы сидов (без обязательного GPU).

Запуск с корня репозитория:
  python -m pytest experiments/06-rfp-protocol-tests/test_protocol.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(ROOT / "experiments" / "06-rfp-v0"))
sys.path.insert(0, str(PT))

from tier_checks import check_tier_a_engineering, check_tier_b_field, check_tier_c_task_signal  # noqa: E402


def test_tier_c_fails_on_flat_val_ce() -> None:
    flat = [3.5] * 10
    r = check_tier_c_task_signal(flat, min_std=1e-6, min_range=1e-5)
    assert r.get("pass") is False
    assert r.get("val_ce_range") == 0.0


def test_tier_c_passes_on_varying_series() -> None:
    s = [3.5, 3.48, 3.52, 3.49, 3.51]
    r = check_tier_c_task_signal(s, min_std=1e-8, min_range=1e-7)
    assert r.get("pass") is True


def test_tier_b_rc_threshold() -> None:
    assert check_tier_b_field(0.9, 0.0)["pass"] is True
    assert check_tier_b_field(0.1, 0.0, min_rc=0.35)["pass"] is False


def test_tier_a_grad_cap() -> None:
    m = torch.nn.Linear(2, 2)
    r = check_tier_a_engineering(m, max_grad_l2=100.0, max_grad_cap=25.0)
    assert r["grad_under_cap"] is False
    r2 = check_tier_a_engineering(m, max_grad_l2=1.0, max_grad_cap=25.0)
    assert r2["pass"] is True


def test_fresh_batch_generators_differ_per_epoch() -> None:
    """Тот же контракт, что в protocol_train: разные сиды эпох → разные батчи."""
    from protocol_train import DEVICE, VOCAB_SIZE, BATCH_SIZE, SEQ_LEN, _make_batch_list  # noqa: E402

    g0 = torch.Generator(device=DEVICE)
    g0.manual_seed(42 + 0 * 100_003)
    g1 = torch.Generator(device=DEVICE)
    g1.manual_seed(42 + 1 * 100_003)
    a = _make_batch_list(2, g0)
    b = _make_batch_list(2, g1)
    assert not torch.equal(a[0], b[0])

