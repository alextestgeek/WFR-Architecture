"""
Единая целевая функция обучения WFR-LM (теория §6 / Exp 05–08).

L = α · L_task + β · (1 − RC) + γ · E, где L_task — next-token CE,
E — средняя доля спайков по standing wave (как ``WFRLM`` задаёт ``state.energy``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# Значения зафиксированы в Phase 1; менять только осознанно (и в docs).
ALPHA = 1.0
BETA = 0.15
GAMMA = 0.1


def next_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-token CE: logits [B,T,V], targets [B,T] — сдвиг на один шаг."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        targets[:, 1:].reshape(-1),
    )


def _scalarize_rc(rc: torch.Tensor) -> torch.Tensor:
    return rc if rc.dim() == 0 else rc.mean()


def _scalarize_energy(energy: torch.Tensor) -> torch.Tensor:
    return energy if energy.dim() == 0 else energy.mean()


def composite_training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rc: torch.Tensor,
    energy: torch.Tensor,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    task_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Возвращает (total, ce, parts):
    - ``total`` — для backward;
    - ``ce`` — тот же граф, что в L_task (для RFP: ce.detach() и т.д.);
    - ``parts`` — откладываемые скаляры для логов.
    """
    ce = next_token_cross_entropy(logits, targets)
    if task_only:
        total = alpha * ce
        parts = {
            "ce": ce.detach(),
            "rc_term": torch.zeros((), device=ce.device, dtype=ce.dtype),
            "energy": torch.zeros((), device=ce.device, dtype=ce.dtype),
        }
        return total, ce, parts
    rc_term = 1.0 - _scalarize_rc(rc)
    eng = _scalarize_energy(energy)
    total = alpha * ce + beta * rc_term + gamma * eng
    parts = {
        "ce": ce.detach(),
        "rc_term": rc_term.detach(),
        "energy": eng.detach(),
    }
    return total, ce, parts
