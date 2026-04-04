"""
Официальный wrapper WFRLM: WFR core + контентный канал + readout.

Ядро: пакет :mod:`wfr.core` (репозиторий в PYTHONPATH или cwd при запуске из корня).

Трактовка токена как фазового сдвига и readout: docs/03-theory.md §10.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wfr.core import WFRNetwork

WFRState = namedtuple(
    "WFRState",
    [
        "resonance",
        "spikes",
        "rc",
        "energy",
        "phases",
        "logits",
        "layer_resonance_means",
        "layer_spike_means",
        "layer_rc_share",
    ],
    defaults=(None, None, None),
)


class WFRLM(nn.Module):
    def __init__(
        self,
        core_network: WFRNetwork,
        vocab_size: int,
        num_phases: int,
        readout_feat_dim: int = 3,
        readout_mlp_hidden: Optional[int] = None,
        content_neighbor_mix: bool = False,
        readout_wave_kernel: int = 1,
    ):
        super().__init__()
        self.core = core_network
        self.vocab_size = vocab_size
        self.num_phases = num_phases
        if readout_feat_dim < 3:
            raise ValueError("readout_feat_dim must be >= 3 (base: wave, rc, energy)")
        self.readout_feat_dim = readout_feat_dim
        self.readout_mlp_hidden = readout_mlp_hidden
        if readout_mlp_hidden is not None and readout_mlp_hidden < 1:
            raise ValueError("readout_mlp_hidden must be >= 1 when set")
        if readout_wave_kernel < 1:
            raise ValueError("readout_wave_kernel must be >= 1 (1 = без локальной смеси волны)")
        self.readout_wave_kernel = readout_wave_kernel
        self._readout_wave_dw: Optional[nn.Conv1d] = None
        if readout_wave_kernel > 1:
            self._readout_wave_dw = nn.Conv1d(
                1, 1, kernel_size=readout_wave_kernel, padding=0, bias=False
            )
            nn.init.zeros_(self._readout_wave_dw.weight)
        self.content_neighbor_mix = bool(content_neighbor_mix)
        # Causal: фазовый вклад позиции t зависит от токена t и (масштабированно) от t−1.
        # Ядро WFRNetwork не меняется; это локальная связь до резонансного стека.
        self._content_neighbor_scale = (
            nn.Parameter(torch.zeros(1)) if self.content_neighbor_mix else None
        )
        self.token_phase_offset = nn.Embedding(vocab_size, num_phases)
        extra = readout_feat_dim - 3
        self._wave_extra = nn.Linear(1, extra) if extra > 0 else None
        if readout_mlp_hidden:
            self.head = nn.Sequential(
                nn.Linear(readout_feat_dim, readout_mlp_hidden),
                nn.GELU(),
                nn.Linear(readout_mlp_hidden, vocab_size),
            )
        else:
            self.head = nn.Linear(readout_feat_dim, vocab_size)

    def forward(
        self,
        positions: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        *,
        content_scale: float = 1.0,
    ):
        if tokens is None:
            content_delta = None
        else:
            base = self.token_phase_offset(tokens)
            if self.content_neighbor_mix and self._content_neighbor_scale is not None:
                left = torch.zeros_like(base)
                left[:, 1:, :] = base[:, :-1, :]
                content_delta = base + self._content_neighbor_scale * left
            else:
                content_delta = base
            if content_scale != 1.0:
                content_delta = content_delta * float(content_scale)
        out = self.core(positions, content_delta=content_delta)

        resonance_out = out["standing_wave"]
        batch = resonance_out.shape[0]

        thr = self.core.threshold
        spikes = (torch.abs(resonance_out) > thr).float()
        energy = spikes.mean()

        rc = out["resonance_confidence"]
        if self.core.rc is not None:
            rc = self.core.rc

        wave_for_logits = resonance_out
        if self._readout_wave_dw is not None:
            k = self.readout_wave_kernel
            x = resonance_out.unsqueeze(1)
            x = F.pad(x, (k - 1, 0))
            x = self._readout_wave_dw(x)
            wave_for_logits = resonance_out + x.squeeze(1)

        logits = self.readout(wave_for_logits, rc, energy, batch)

        layer_resonance_means = torch.stack(
            [r.detach().abs().mean() for r in out["layer_resonances"]]
        )
        layer_spike_means = torch.stack(
            [s.detach().float().mean() for s in out["layer_spikes"]]
        )
        tot_r = layer_resonance_means.sum().clamp(min=1e-8)
        layer_rc_share = layer_resonance_means / tot_r

        return WFRState(
            resonance=resonance_out,
            spikes=spikes,
            rc=rc,
            energy=energy,
            phases=out["phases"],
            logits=logits,
            layer_resonance_means=layer_resonance_means,
            layer_spike_means=layer_spike_means,
            layer_rc_share=layer_rc_share,
        )

    def readout(
        self,
        wave: torch.Tensor,
        rc: torch.Tensor,
        energy: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """[B,T] standing wave → [B,T,readout_feat_dim] (база: wave, rc, energy; остальное — от wave) → logits."""
        if wave.dim() != 2:
            raise ValueError("Expected standing_wave [batch, seq]")
        _b, seq = wave.shape
        wave_feat = wave.unsqueeze(-1)
        if rc.dim() == 0:
            rc_feat = rc.view(1, 1, 1).expand(batch, seq, 1)
        else:
            rc_feat = rc.view(batch, 1, 1).expand(-1, seq, 1)
        if energy.dim() == 0:
            energy_b = energy.view(1, 1, 1).expand(batch, seq, 1)
        else:
            energy_b = energy.view(batch, 1, 1).expand(-1, seq, 1)

        parts = [wave_feat, rc_feat, energy_b]
        if self._wave_extra is not None:
            parts.append(self._wave_extra(wave_feat))
        features = torch.cat(parts, dim=-1)
        return self.head(features)  # Linear или MLP (feat → hidden → vocab)
