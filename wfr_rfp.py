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


def rfp_step_v02(
    network: torch.nn.Module,
    state: "WFRState",
    ce_loss: torch.Tensor,
    mode: str = "offline",
    *,
    spike_target: float = 0.25,
    eta_f: float = 8e-5,
    eta_theta: float = 2e-4,
    eta_alpha: float = 3e-5,
    min_spike_bonus: float = 0.12,
) -> Dict[str, torch.Tensor]:
    r"""
    RFP v0.2: усиление частоты при низком спайке, \(\cos\) между слоями для \(\Delta\theta\),
    мягкий homeostatic по decay.

    Обозначения (KaTeX):
    - \(\varepsilon = 1 - e^{-\min(L_{\mathrm{CE}},\,8)}\) — нормированная ошибка задачи.
    - \(\mathrm{RC}\) — скалярный resonance confidence.
    - \(r\) — средняя доля спайков по standing wave; \(r^\*=\) ``spike_target``.
    - Бонус к частоте: \(b = \max(0,\, r_{\min} - r)\) с \(r_{\min}=\) ``min_spike_bonus``.
    - \(\Delta f_l \propto \eta_f\,\mathrm{RC}(1-\varepsilon) + 0.8\,\eta_f\,b\).
    - \(\Delta\theta_l \propto \eta_\theta\big(\mathrm{RC}\,\varepsilon + 0.1\cos(\phi_{l-1}-\phi_{l+1})\big)\) (средние |u| как прокси фаз).
    - \(\Delta\alpha_l \propto \eta_\alpha\,(r-r^\*)\cdot \kappa(r)\), где \(\kappa=1\) если \(r>0.05\) иначе \(0.3\).
    """
    del mode
    deltas: Dict[str, torch.Tensor] = {}
    err = 1.0 - torch.exp(-ce_loss.detach().clamp(max=8.0))
    rc = state.rc.detach()
    if rc.dim() > 0:
        rc = rc.mean()

    spikes = state.spikes.detach()
    real_spike_rate = spikes.mean()
    device, dtype = real_spike_rate.device, real_spike_rate.dtype

    means = getattr(state, "layer_resonance_means", None)
    if means is not None:
        means = means.detach()

    for i, layer in enumerate(network.resonance_layers):
        # Бонус частоты при «загасших» спайках
        spike_bonus = torch.relu(
            torch.as_tensor(min_spike_bonus, device=device, dtype=dtype) - real_spike_rate
        )
        # \(\Delta f\)
        d_freq = (
            eta_f * rc * (1.0 - err)
            + eta_f * 0.8 * spike_bonus
        )
        deltas[f"layer_{i}.frequency"] = d_freq

        # phase_lock_term: \(\cos(\bar u_{l-1} - \bar u_{l+1})\) по средним |резонанс|
        if means is not None and means.numel() >= 2:
            Ln = means.numel()
            prev = means[i - 1] if i > 0 else means[i]
            nxt = means[i + 1] if i < Ln - 1 else means[i]
            phase_lock_term = torch.cos(prev - nxt)
        else:
            phase_lock_term = torch.zeros((), device=device, dtype=dtype)

        deltas[f"layer_{i}.phase_bias"] = eta_theta * (rc * err + 0.1 * phase_lock_term)

        homeo_error = real_spike_rate - torch.as_tensor(
            spike_target, device=device, dtype=dtype
        )
        scale = torch.where(
            real_spike_rate > 0.05,
            torch.ones((), device=device, dtype=dtype),
            torch.as_tensor(0.3, device=device, dtype=dtype),
        )
        deltas[f"layer_{i}.decay"] = eta_alpha * homeo_error * scale

    return deltas


def rfp_step_v03(
    network: torch.nn.Module,
    state: "WFRState",
    ce_loss: torch.Tensor,
    mode: str = "offline",
    *,
    spike_target: float = 0.25,
    eta_f: float = 1.2e-4,
    eta_theta: float = 3e-4,
    eta_alpha: float = 8e-6,
    rescue_threshold: float = 0.12,
    rescue_factor: float = 0.45,
) -> Dict[str, torch.Tensor]:
    r"""
    RFP v0.3: per-layer пластичность по \(r_l\), \(\mathrm{rc\_share}_l\), rescue-терм, мягкий homeostatic.

    Обозначения (KaTeX):
    - \(\varepsilon = 1 - e^{-\min(L_{\mathrm{CE}},\,8)}\).
    - \(r_l\) — средняя доля спайков по слою (``layer_spike_means[l]``).
    - \(\rho_l = \mathrm{rc\_share}_l\) — доля энергии резонанса слоя (``layer_rc_share[l]``).
    - \(\mathrm{rescue}_l = \lambda \cdot \max(0,\, r_{\mathrm{res}} - r_l)\); \(\lambda=\) ``rescue_factor``, \(r_{\mathrm{res}}=\) ``rescue_threshold``.
    - \(\Delta f_l = \eta_f \rho_l (1-\varepsilon) + \eta_f \cdot \mathrm{rescue}_l\).
    - \(\Delta\theta_l = \eta_\theta\big(\rho_l \varepsilon + 0.25 \cos(\bar u_l - \bar u_{l+1})\big)\) (циклический сосед).
    - \(\Delta\alpha_l = \eta_\alpha (r_l - r^*) \kappa_l\), \(\kappa_l=1\) если \(r_l > r_{\mathrm{res}}\), иначе \(0.25\); \(r^*=\) ``spike_target``.
    """
    del mode
    deltas: Dict[str, torch.Tensor] = {}
    err = 1.0 - torch.exp(-ce_loss.detach().clamp(max=8.0))

    lsm = getattr(state, "layer_spike_means", None)
    lrc = getattr(state, "layer_rc_share", None)
    means = getattr(state, "layer_resonance_means", None)
    if means is not None:
        means = means.detach()

    L = len(network.resonance_layers)
    if lsm is None or lrc is None or lsm.numel() < L or lrc.numel() < L:
        # Fallback: одна скалярная доля спайка на все слои
        r_global = state.spikes.detach().mean()
        device, dtype = r_global.device, r_global.dtype
        lsm = r_global.expand(L).contiguous()
        lrc = torch.ones(L, device=device, dtype=dtype) / float(L)

    device = lsm.device
    dtype = lsm.dtype
    spike_t = torch.as_tensor(spike_target, device=device, dtype=dtype)
    r_res = torch.as_tensor(rescue_threshold, device=device, dtype=dtype)

    for i, _layer in enumerate(network.resonance_layers):
        r_l = lsm[i]
        rc_share_l = lrc[i]
        rescue = rescue_factor * torch.relu(r_res - r_l)
        d_freq = eta_f * rc_share_l * (1.0 - err) + eta_f * rescue
        deltas[f"layer_{i}.frequency"] = d_freq

        if means is not None and means.numel() == L:
            nxt = (i + 1) % L
            phase_lock = torch.cos(means[i] - means[nxt])
        else:
            phase_lock = torch.zeros((), device=device, dtype=dtype)

        deltas[f"layer_{i}.phase_bias"] = eta_theta * (rc_share_l * err + 0.25 * phase_lock)

        homeo_error = r_l - spike_t
        kappa = torch.where(r_l > r_res, torch.ones((), device=device, dtype=dtype), torch.as_tensor(0.25, device=device, dtype=dtype))
        deltas[f"layer_{i}.decay"] = eta_alpha * homeo_error * kappa

    return deltas


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
