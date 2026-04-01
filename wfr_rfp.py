"""
Resonant Field Plasticity (RFP) v0 — шаги поверх Adam + composite loss.

Параметры ядра: experiments/00-smoke-test/wfr_core.py
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

if TYPE_CHECKING:
    from wfr_lm import WFRState


def get_param_by_name(network: torch.nn.Module, name: str) -> torch.nn.Parameter:
    """Имя вида layer_0.frequency | layer_0.freq | layer_0.phase_bias | layer_0.decay | layer_0.spike_threshold."""
    m = re.match(r"layer_(\d+)\.(.+)", name.strip())
    if not m:
        raise KeyError(f"Bad RFP param name: {name}")
    idx = int(m.group(1))
    attr = m.group(2)
    if attr == "freq":
        attr = "frequency"
    layer = network.resonance_layers[idx]
    p = getattr(layer, attr, None)
    if p is None or not isinstance(p, torch.nn.Parameter):
        raise KeyError(f"No Parameter {attr} on layer {idx}")
    return p


def rfp_step(
    network: torch.nn.Module,
    state: "WFRState",
    ce_loss: torch.Tensor,
    mode: str = "offline",
) -> Dict[str, torch.Tensor]:
    """
    Правило v0: Δf ∝ rc·(1−err), Δθ ∝ rc·err, Δdecay ∝ (r−r*).
    mode зарезервирован для online/offline (логика шага снаружи).
    """
    del mode  # шаг частоты задаётся вызывающим кодом (каждые N итераций)
    deltas: Dict[str, torch.Tensor] = {}
    eta_f, eta_theta, eta_alpha = 5e-5, 1e-4, 2e-4

    err = 1.0 - torch.exp(-ce_loss.detach().clamp(max=10.0))
    rc = state.rc.detach()
    if rc.dim() > 0:
        rc = rc.mean()
    real_rate = state.spikes.detach().mean()

    for i, layer in enumerate(network.resonance_layers):
        target_rate = layer.spike_rate_target if hasattr(layer, "spike_rate_target") else 0.25

        deltas[f"layer_{i}.frequency"] = eta_f * rc * (1.0 - err)
        deltas[f"layer_{i}.phase_bias"] = eta_theta * rc * err
        deltas[f"layer_{i}.decay"] = eta_alpha * (real_rate - torch.as_tensor(target_rate, device=real_rate.device, dtype=real_rate.dtype))

    return deltas


def rfp_step_v01(
    network: torch.nn.Module,
    state: "WFRState",
    ce_loss: torch.Tensor,
    layer_resonance_means: Optional[torch.Tensor] = None,
    mode: str = "offline",
) -> Dict[str, torch.Tensor]:
    """
    v0.1: к обновлению phase_bias добавляется cos(Δ) между средними |u| соседних слоёв.
    layer_resonance_means: [L] — средние |resonance| по слоям (детачнутые скаляры).
    """
    base = rfp_step(network, state, ce_loss, mode=mode)
    if layer_resonance_means is None or layer_resonance_means.numel() < 2:
        return base

    eta_cos = 5e-5
    rc = state.rc.detach()
    if rc.dim() > 0:
        rc = rc.mean()
    err = 1.0 - torch.exp(-ce_loss.detach().clamp(max=10.0))
    L = layer_resonance_means.numel()
    device = layer_resonance_means.device
    out = dict(base)
    for i in range(L):
        prev = layer_resonance_means[i - 1] if i > 0 else layer_resonance_means[i]
        nxt = layer_resonance_means[i + 1] if i < L - 1 else layer_resonance_means[i]
        dphi = prev - nxt
        cos_term = torch.cos(dphi).to(device=device, dtype=torch.float32)
        extra = eta_cos * rc * err * cos_term
        key = f"layer_{i}.phase_bias"
        out[key] = out[key] + extra
    return out


def apply_rfp_deltas(
    network: torch.nn.Module,
    deltas: Dict[str, Any],
    clip: float = 0.01,
) -> None:
    for name, delta in deltas.items():
        param = get_param_by_name(network, name)
        d = delta
        if not isinstance(d, torch.Tensor):
            d = torch.as_tensor(d, device=param.data.device, dtype=param.data.dtype)
        else:
            d = d.to(device=param.data.device, dtype=param.data.dtype)
        param.data.add_(torch.clamp(d, -clip, clip))
