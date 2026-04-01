"""Tier A/B/C проверки для протокола fresh-train (без изменения ядра)."""

from __future__ import annotations

import math
import statistics
from typing import Any

import torch


def check_tier_a_engineering(
    model: torch.nn.Module,
    max_grad_l2: float,
    *,
    max_grad_cap: float = 25.0,
) -> dict[str, Any]:
    """
    Tier A: стабильность оптимизации — градиенты в разумном коридоре, веса finite.
    Порог по градиенту намеренно мягкий (инженерный sanity), не «научный» критерий качества.
    """
    finite_ok = True
    bad_names: list[str] = []
    for name, p in model.named_parameters():
        if p.data.numel() and not torch.isfinite(p.data).all():
            finite_ok = False
            bad_names.append(name)
    grad_ok = max_grad_l2 <= max_grad_cap and math.isfinite(max_grad_l2)
    return {
        "tier": "A",
        "pass": bool(finite_ok and grad_ok),
        "finite_weights": finite_ok,
        "nonfinite_param_names": bad_names,
        "max_grad_l2": max_grad_l2,
        "max_grad_cap": max_grad_cap,
        "grad_under_cap": grad_ok,
    }


def check_tier_b_field(
    final_val_rc: float,
    mean_spike_rate: float,
    *,
    min_rc: float = 0.35,
) -> dict[str, Any]:
    """
    Tier B: «физика поля» — RC и spike не NaN; RC не коллапсирует в ноль (мягкий порог).
    Spike-диапазон не навязываем как PASS/FAIL (на toy он часто 0).
    """
    ok = (
        math.isfinite(final_val_rc)
        and math.isfinite(mean_spike_rate)
        and final_val_rc >= min_rc
    )
    return {
        "tier": "B",
        "pass": ok,
        "final_val_rc": final_val_rc,
        "mean_spike_rate": mean_spike_rate,
        "min_rc_threshold": min_rc,
    }


def check_tier_c_task_signal(
    history_val_ce: list[float],
    *,
    min_std: float = 1e-8,
    min_range: float = 1e-7,
) -> dict[str, Any]:
    """
    Tier C: на фиксированном val динамика val CE не должна быть математически нулевой
    при достаточной длине — иначе нет смысла коррелировать RFP с «ошибкой».

    Мягкие пороги: если серия слишком короткая, только report без fail.
    """
    n = len(history_val_ce)
    if n < 3:
        return {
            "tier": "C",
            "pass": None,
            "skipped": True,
            "reason": "need_at_least_3_epochs",
            "history_len": n,
        }
    std = float(statistics.pstdev(history_val_ce))
    rng = float(max(history_val_ce) - min(history_val_ce))
    signal_ok = std >= min_std or rng >= min_range
    return {
        "tier": "C",
        "pass": signal_ok,
        "val_ce_std": std,
        "val_ce_range": rng,
        "min_std": min_std,
        "min_range": min_range,
        "history_len": n,
    }
