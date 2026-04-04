"""
Подбор TinyCausalTransformer по числу параметров WFRLM (этап A2 roadmap).

WFRLM с узким readout (feat_dim=3) и тем же V даёт ~5k trainable params; дефолтный
Transformer (d=128, L=2) — на порядок больше. Грид здесь подбирает (d_model, nhead,
nlayers, dim_feedforward) с минимальным |log(n_tf / n_wfr)| при устойчивости слоя PyTorch.
"""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class MatchedTransformerConfig:
    d_model: int
    nhead: int
    nlayers: int
    dim_feedforward: int
    num_params: int
    wfr_num_params: int

    @property
    def param_ratio_tf_over_wfr(self) -> float:
        return self.num_params / max(1, self.wfr_num_params)


def count_wfr_lm_trainable(
    vocab_size: int,
    num_resonance_layers: int,
    spike_rate_target: float = 0.25,
    readout_feat_dim: int = 3,
    readout_mlp_hidden: Optional[int] = None,
    content_neighbor_mix: bool = False,
    phase_causal_kernel: int = 1,
    readout_wave_kernel: int = 1,
) -> int:
    """То же построение, что train_wikitext_run, без обучения (stdout слоёв подавляется)."""
    from pathlib import Path
    import sys

    root = Path(__file__).resolve().parents[2]
    exp08 = root / "experiments" / "08-wikitext-rfp"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(exp08))

    from run_wikitext_train import PHASE0_FREQ_BALANCED, _resonance_freqs_thresholds  # noqa: E402
    from wfr.core import WFRNetwork  # noqa: E402
    from wfr_lm import WFRLM  # noqa: E402

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        layer_freqs, layer_thresh = _resonance_freqs_thresholds(num_resonance_layers)
        cfg = PHASE0_FREQ_BALANCED
        core = WFRNetwork(
            num_phases=16,
            num_fractal_levels=6,
            num_resonance_layers=num_resonance_layers,
            layer_frequencies=layer_freqs,
            layer_thresholds=layer_thresh,
            homeostatic_enabled=True,
            spike_rate_target=spike_rate_target,
            homeostatic_eta=cfg["homeostatic_eta"],
            homeostatic_always_on=True,
            phase_causal_kernel=phase_causal_kernel,
        )
        core.target_mode = cfg["target_mode"]
        model = WFRLM(
            core,
            vocab_size=vocab_size,
            num_phases=16,
            readout_feat_dim=readout_feat_dim,
            readout_mlp_hidden=readout_mlp_hidden,
            content_neighbor_mix=content_neighbor_mix,
            readout_wave_kernel=readout_wave_kernel,
        )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_transformer_trainable(
    vocab_size: int,
    seq_len: int,
    d_model: int,
    nhead: int,
    nlayers: int,
    dim_feedforward: int,
    max_len_pad: int = 32,
) -> int:
    from run_transformer_char_baseline import TinyCausalTransformer

    m = TinyCausalTransformer(
        vocab_size,
        d_model=d_model,
        nhead=nhead,
        nlayers=nlayers,
        dim_feedforward=dim_feedforward,
        max_len=seq_len + max_len_pad,
    )
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_transformer_to_wfr_params(
    wfr_num_params: int,
    vocab_size: int,
    seq_len: int,
    *,
    min_d_model: int = 8,
    max_layers: int = 6,
    max_d_model: int = 128,
) -> MatchedTransformerConfig:
    """
    Перебор небольшого грида: минимизируем |log(n_tf / n_wfr)|.
    Часть комбинаций отбрасывается конструктором / PyTorch (узкий d_model).

    ``min_d_model`` (по умолчанию 8) отсекает крайне узкие модели вроде d=4, где CE
    часто упирается в оптимизацию, а не в архитектурный зазор.
    """
    if wfr_num_params <= 0:
        raise ValueError("wfr_num_params must be positive")

    dim_ff_choices = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
    best_score = float("inf")
    best: tuple[int, int, int, int, int] | None = None

    lo = max(4, min_d_model)
    if lo % 4 != 0:
        lo = ((lo + 3) // 4) * 4
    for d_model in range(lo, max_d_model + 1, 4):
        for nhead in (1, 2, 4, 8):
            if d_model % nhead != 0:
                continue
            for nlayers in range(1, max_layers + 1):
                for dim_ff in dim_ff_choices:
                    if dim_ff < d_model:
                        continue
                    try:
                        n = count_transformer_trainable(
                            vocab_size, seq_len, d_model, nhead, nlayers, dim_ff
                        )
                    except (RuntimeError, ValueError):
                        continue
                    ratio = n / float(wfr_num_params)
                    score = abs(math.log(max(ratio, 1e-9)))
                    if score < best_score:
                        best_score = score
                        best = (d_model, nhead, nlayers, dim_ff, n)

    if best is None:
        raise RuntimeError(
            "Could not find any valid TinyCausalTransformer config; "
            "try increasing max_d_model or check PyTorch limits."
        )

    d_model, nhead, nlayers, dim_ff, n = best
    return MatchedTransformerConfig(
        d_model=d_model,
        nhead=nhead,
        nlayers=nlayers,
        dim_feedforward=dim_ff,
        num_params=n,
        wfr_num_params=wfr_num_params,
    )
