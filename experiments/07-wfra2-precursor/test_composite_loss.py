"""Формы и градиенты композитного лосса (pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "07-wfra2-precursor"))

from composite_loss import (  # noqa: E402
    loss_energy_homeostat,
    loss_phase_alignment,
    loss_resonance_coherence,
    wfra2_composite_loss,
)


def test_resonance_in_0_1_range() -> None:
    ph = torch.randn(2, 16, 8) * 0.1
    lr = loss_resonance_coherence(ph)
    assert lr.ndim == 0
    assert 0.0 <= float(lr) <= 2.0


def test_wfra2_backward() -> None:
    B, T, V, P = 2, 12, 32, 16
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    phases = torch.randn(B, T, P, requires_grad=True)
    spk = torch.sigmoid(torch.randn((), requires_grad=True))
    total, _ = wfra2_composite_loss(logits, targets, phases, spk)
    total.backward()
    assert logits.grad is not None
    assert phases.grad is not None


def test_energy_homeostat() -> None:
    r = torch.tensor(0.15, requires_grad=True)
    l = loss_energy_homeostat(r, 0.10)
    assert abs(l.detach().item() - 0.05) < 1e-5
