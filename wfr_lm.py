"""
Официальный wrapper WFRLM: WFR core + контентный канал + readout.

Ядро: experiments/00-smoke-test/wfr_core.py (импорт через smoke-test в sys.path).
"""

from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
_smoke = Path(__file__).resolve().parent / "experiments" / "00-smoke-test"
if str(_smoke) not in sys.path:
    sys.path.insert(0, str(_smoke))

from wfr_core import WFRNetwork  # noqa: E402

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
    ],
    defaults=(None,),
)


class WFRLM(nn.Module):
    def __init__(
        self,
        core_network: WFRNetwork,
        vocab_size: int,
        num_phases: int,
    ):
        super().__init__()
        self.core = core_network
        self.vocab_size = vocab_size
        self.num_phases = num_phases
        self.token_phase_offset = nn.Embedding(vocab_size, num_phases)
        feat_dim = 3
        self.head = nn.Linear(feat_dim, vocab_size)

    def forward(
        self,
        positions: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ):
        content_delta = self.token_phase_offset(tokens) if tokens is not None else None
        out = self.core(positions, content_delta=content_delta)

        resonance_out = out["standing_wave"]
        batch = resonance_out.shape[0]

        thr = self.core.threshold
        spikes = (torch.abs(resonance_out) > thr).float()
        energy = spikes.mean()

        rc = out["resonance_confidence"]
        if self.core.rc is not None:
            rc = self.core.rc

        logits = self.readout(resonance_out, rc, energy, batch)

        layer_resonance_means = torch.stack(
            [r.detach().abs().mean() for r in out["layer_resonances"]]
        )

        return WFRState(
            resonance=resonance_out,
            spikes=spikes,
            rc=rc,
            energy=energy,
            phases=None,
            logits=logits,
            layer_resonance_means=layer_resonance_means,
        )

    def readout(
        self,
        wave: torch.Tensor,
        rc: torch.Tensor,
        energy: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """[B,T] standing wave → признаки [B,T,3] → logits [B,T,V] (next-token как в Exp 05)."""
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

        features = torch.cat([wave_feat, rc_feat, energy_b], dim=-1)
        return self.head(features)
