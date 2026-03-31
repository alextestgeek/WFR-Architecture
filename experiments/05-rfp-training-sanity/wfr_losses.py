"""Целевая функция Phase 1 (теория §6) — один модуль для обучения и проверок."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def task_loss_ce(logits: torch.Tensor, batch_tokens: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        batch_tokens[:, 1:].reshape(-1),
    )


def energy_cost(out: dict) -> torch.Tensor:
    spikes = out["layer_spikes"]
    rates = torch.stack([s.mean() for s in spikes])
    return rates.mean()


def rc_penalty(out: dict) -> torch.Tensor:
    rc = out["resonance_confidence"]
    if rc.dim() == 0:
        return 1.0 - rc
    return (1.0 - rc).mean()


def compute_loss(
    logits: torch.Tensor,
    batch_tokens: torch.Tensor,
    out: dict,
    alpha: float,
    beta: float,
    gamma: float,
    task_only: bool,
) -> tuple[torch.Tensor, dict]:
    ce = task_loss_ce(logits, batch_tokens)
    if task_only:
        total = alpha * ce
        parts = {
            "ce": ce.detach(),
            "rc_term": torch.zeros((), device=ce.device),
            "energy": torch.zeros((), device=ce.device),
        }
        return total, parts
    rc_term = rc_penalty(out)
    energy = energy_cost(out)
    total = alpha * ce + beta * rc_term + gamma * energy
    parts = {"ce": ce.detach(), "rc_term": rc_term.detach(), "energy": energy.detach()}
    return total, parts
