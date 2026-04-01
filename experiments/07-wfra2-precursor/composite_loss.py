"""
Композитный лосс в духе WFRA-2.0 / RFP v1 (прекурсор): явные L_res, L_energy, L_phase.

ВАЖНО по теории текущего кода:
- В WFRNetwork одна тензорная сетка фаз ``phases`` [B,T,P] общая для всех резонансных слоёв.
  Поэтому L_res из плана «по слоям l» здесь сводится к одному порядковому параметру по группам фаз
  (аналог «усреднить по l» при L идентичных слагаемых).
- L_energy в плане: |r̄ − r*|. В WFRLM ``state.energy`` — средняя доля спайков по standing wave;
  используем её как r̄ (как в эксп. 06), не смешивая с другими определениями energy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def loss_task_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-token CE: logits [B,T,V], targets [B,T] — сдвиг как в Exp 06."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        targets[:, 1:].reshape(-1),
    )


def loss_resonance_coherence(phases: torch.Tensor) -> torch.Tensor:
    """
    L_res ≈ 1 − |⟨e^{iφ}⟩| — слабые/размазанные фазы штрафуются.
    phases: [B, T, P] — усреднение по батчу и времени, модуль по комплексной плоскости, затем среднее по P.
    """
    z = torch.exp(1j * phases.to(torch.float32))
    # порядковый параметр на каждую фазовую группу
    order = z.mean(dim=(0, 1))
    mag = torch.abs(order).mean()
    return (1.0 - mag).to(phases.dtype)


def loss_energy_homeostat(
    mean_spike_rate: torch.Tensor,
    target_rate: float = 0.10,
) -> torch.Tensor:
    """L_energy = |r̄ − r*|."""
    return (mean_spike_rate - target_rate).abs()


def loss_phase_alignment(phases: torch.Tensor) -> torch.Tensor:
    """
    Согласованность фаз внутри вектора фаз: отклонение от средней фазы по группам.
    L_phase = E[|sin(φ − φ̄)|] — φ̄ по оси num_phases (detach как «медленная» цель внутри шага).
    """
    phi_bar = phases.mean(dim=-1, keepdim=True).detach()
    return torch.abs(torch.sin(phases - phi_bar)).mean()


def wfra2_composite_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    phases: Optional[torch.Tensor],
    mean_spike_rate: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.05,
    spike_target: float = 0.10,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    L_total = L_task + α L_res + β L_energy + γ L_phase
    Если phases нет — падаем обратно на CE-only по контракту вызывающего (не внутри здесь).
    """
    l_task = loss_task_ce(logits, targets)
    if phases is None:
        raise ValueError("wfra2_composite_loss requires phases (enable phases in WFRLM forward)")
    l_res = loss_resonance_coherence(phases)
    l_en = loss_energy_homeostat(mean_spike_rate, spike_target)
    l_ph = loss_phase_alignment(phases)
    total = l_task + alpha * l_res + beta * l_en + gamma * l_ph
    parts = {
        "l_task": l_task.detach(),
        "l_res": l_res.detach(),
        "l_energy": l_en.detach(),
        "l_phase": l_ph.detach(),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }
    return total, parts
