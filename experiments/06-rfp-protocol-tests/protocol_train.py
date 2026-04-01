"""
Обучение с протоколом «свежие train-батчи каждую эпоху + фиксированный val holdout».

Не изменяет ядро (wfr_core / wfr_lm / wfr_rfp): только схема сэмплирования данных и вызов тех же шагов, что в run_rfp_training.train_run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import torch
import torch.nn.functional as F

# Импорты из Exp 06 (без дублирования evaluate / метрик)
import sys

ROOT = Path(__file__).resolve().parents[2]
_EXP06 = ROOT / "experiments" / "06-rfp-v0"
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(_EXP06))

from phase0_best_config import PHASE0_FREQ_BALANCED  # noqa: E402
from wfr_core import WFRNetwork  # noqa: E402
from wfr_lm import WFRLM  # noqa: E402
from wfr_rfp import apply_rfp_deltas, rfp_step, rfp_step_v01, rfp_step_v02, rfp_step_v03  # noqa: E402

from run_rfp_training import (  # noqa: E402
    ALPHA,
    BATCH_SIZE,
    BETA,
    DEVICE,
    GAMMA,
    RunMetrics,
    SEQ_LEN,
    VOCAB_SIZE,
    _phase_bias_vector,
    evaluate,
    evaluate_detailed,
    pearson_r,
    plot_training_curves,
    total_grad_l2_norm,
)


@dataclass
class FreshRunResult:
    """Результат прогона с протоколом fresh-train."""

    metrics: RunMetrics
    history_val_ce: list[float]
    history_val_rc: list[float]
    history_spike: list[float]
    model: WFRLM


def _make_batch_list(
    n_batches: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    return [
        torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=generator, device=DEVICE)
        for _ in range(n_batches)
    ]


def train_run_fresh_epochs(
    epochs: int,
    use_rfp: bool,
    rfp_interval: int,
    rfp_version: str,
    online_rfp: bool,
    homeostatic_always_on: bool,
    spike_rate_target: float,
    *,
    train_seed: int = 42,
    val_seed: int = 4242,
    num_train_batches: int = 8,
    num_val_batches: int = 4,
    lr: float = 0.05,
    log_path: Optional[Path] = None,
    log_every_epochs: int = 4,
    eta_alpha_v02: float = 3e-5,
    eta_f_v02: float = 8e-5,
    eta_theta_v02: float = 2e-4,
    min_spike_bonus_v02: float = 0.12,
    eta_f_v03: float = 1.2e-4,
    eta_theta_v03: float = 3e-4,
    eta_alpha_v03: float = 8e-6,
    rescue_threshold_v03: float = 0.12,
    rescue_factor_v03: float = 0.45,
    save_png: bool = True,
    plot_path: Optional[Path] = None,
    plot_title_prefix: str = "Protocol 06 — fresh train",
) -> FreshRunResult:
    """
    Каждую эпоху: новые train-батчи (детерминированно от ``train_seed + ep``).
    Val: фиксированный holdout из ``val_seed`` (не меняется между эпохами).
    """
    if rfp_version == "v02" and use_rfp:
        log_path = log_path or (_THIS / "outputs" / "rfp_v02_log_protocol.json")
    elif rfp_version == "v03" and use_rfp:
        log_path = log_path or (_THIS / "outputs" / "rfp_v03_log_protocol.json")

    val_gen = torch.Generator(device=DEVICE)
    val_gen.manual_seed(val_seed)
    val_batches = _make_batch_list(num_val_batches, val_gen)

    cfg = PHASE0_FREQ_BALANCED
    core = WFRNetwork(
        num_phases=16,
        num_fractal_levels=6,
        num_resonance_layers=4,
        layer_frequencies=cfg["frequencies"][:4],
        layer_thresholds=cfg["thresholds"][:4],
        homeostatic_enabled=True,
        spike_rate_target=spike_rate_target,
        homeostatic_eta=cfg["homeostatic_eta"],
        homeostatic_always_on=homeostatic_always_on,
    )
    core.target_mode = cfg["target_mode"]

    model = WFRLM(core, vocab_size=VOCAB_SIZE, num_phases=16).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    mode = "baseline"
    if use_rfp:
        mode = "rfp_online" if online_rfp else f"rfp_every_{rfp_interval}"
        mode += f"_{rfp_version}"
    mode += "_fresh_train"

    max_g = 0.0
    best_val_ce = float("inf")
    final_rc = 0.0
    mean_sp = 0.0
    global_step = 0

    log_entries: list[dict[str, Any]] = []
    prev_log_val_ce: Optional[float] = None
    pb_at_last_log = _phase_bias_vector(model).detach().cpu().clone()
    d_pb_series: list[float] = []
    d_ce_series: list[float] = []

    use_detailed_rfp_log = rfp_version in ("v02", "v03") and use_rfp and log_path is not None
    rescue_hits = 0
    total_rfp_steps = 0

    history_val_ce: list[float] = []
    history_val_rc: list[float] = []
    history_spike: list[float] = []

    for ep in range(epochs):
        train_gen = torch.Generator(device=DEVICE)
        train_gen.manual_seed(train_seed + ep * 100_003)
        train_batches = _make_batch_list(num_train_batches, train_gen)

        model.train()
        for batch in train_batches:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            state = model(batch, batch)
            ce_loss = F.cross_entropy(
                state.logits[:, :-1].reshape(-1, state.logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            rc_term = 1.0 - state.rc if state.rc.dim() == 0 else (1.0 - state.rc).mean()
            total = ALPHA * ce_loss + BETA * rc_term + GAMMA * state.energy
            total.backward()
            g2 = total_grad_l2_norm(model)
            max_g = max(max_g, g2)
            opt.step()

            do_rfp = use_rfp and (online_rfp or (global_step % rfp_interval == 0))
            if do_rfp:
                ce_t = ce_loss.detach()
                if rfp_version == "v01":
                    deltas = rfp_step_v01(
                        model.core,
                        state,
                        ce_t,
                        layer_resonance_means=state.layer_resonance_means,
                    )
                elif rfp_version == "v02":
                    deltas = rfp_step_v02(
                        model.core,
                        state,
                        ce_t,
                        spike_target=spike_rate_target,
                        eta_f=eta_f_v02,
                        eta_theta=eta_theta_v02,
                        eta_alpha=eta_alpha_v02,
                        min_spike_bonus=min_spike_bonus_v02,
                    )
                elif rfp_version == "v03":
                    total_rfp_steps += 1
                    lsm = getattr(state, "layer_spike_means", None)
                    if lsm is not None:
                        r_res = torch.as_tensor(rescue_threshold_v03, device=lsm.device, dtype=lsm.dtype)
                        rescue_vec = rescue_factor_v03 * torch.relu(r_res - lsm.detach())
                        if bool((rescue_vec > 1e-8).any().item()):
                            rescue_hits += 1
                    deltas = rfp_step_v03(
                        model.core,
                        state,
                        ce_t,
                        spike_target=spike_rate_target,
                        eta_f=eta_f_v03,
                        eta_theta=eta_theta_v03,
                        eta_alpha=eta_alpha_v03,
                        rescue_threshold=rescue_threshold_v03,
                        rescue_factor=rescue_factor_v03,
                    )
                else:
                    deltas = rfp_step(model.core, state, ce_t)
                apply_rfp_deltas(model.core, deltas)

        ev = evaluate(model, val_batches)
        history_val_ce.append(ev["val_ce"])
        history_val_rc.append(ev["val_rc"])
        history_spike.append(ev["spike_rate"])
        best_val_ce = min(best_val_ce, ev["val_ce"])
        final_rc = ev["val_rc"]
        mean_sp = ev["spike_rate"]

        should_log = use_detailed_rfp_log and ((ep + 1) % log_every_epochs == 0 or (ep + 1) == epochs)
        if should_log:
            det = evaluate_detailed(model, val_batches)
            pb_now = _phase_bias_vector(model).detach().cpu()
            sum_abs_delta_pb = float((pb_now - pb_at_last_log).abs().sum().item())
            pb_at_last_log = pb_now.clone()
            val_ce = det["val_ce"]
            delta_ce = val_ce - prev_log_val_ce if prev_log_val_ce is not None else 0.0
            if prev_log_val_ce is not None:
                d_pb_series.append(sum_abs_delta_pb)
                d_ce_series.append(delta_ce)
            prev_log_val_ce = val_ce

            entry = {
                "epoch": ep + 1,
                "val_ce": det["val_ce"],
                "val_rc": det["val_rc"],
                "mean_spike_rate": det["mean_spike_rate"],
                "per_layer_spike_rate": det["per_layer_spike_rate"],
                "per_layer_rc_share": det["per_layer_rc_share"],
                "min_max_frequency": det["min_max_frequency"],
                "min_max_decay": det["min_max_decay"],
                "delta_ce_since_last_log": delta_ce,
                "sum_abs_delta_phase_bias_interval": sum_abs_delta_pb,
            }
            log_entries.append(entry)

    corr: Optional[float] = None
    if len(d_pb_series) >= 2 and len(d_ce_series) >= 2:
        corr = pearson_r(d_pb_series, d_ce_series)

    if use_detailed_rfp_log and log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "protocol": "fresh_train_each_epoch",
            "train_seed": train_seed,
            "val_seed": val_seed,
            "num_train_batches": num_train_batches,
            "num_val_batches": num_val_batches,
            "rfp_version": rfp_version,
            "epochs": epochs,
            "rfp_interval": rfp_interval,
            "online_rfp": online_rfp,
            "spike_rate_target": spike_rate_target,
            "correlation_delta_phase_bias_vs_delta_ce": corr,
            "series_delta_pb": d_pb_series,
            "series_delta_ce": d_ce_series,
            "log_every_epochs": log_every_epochs,
            "entries": log_entries,
        }
        if rfp_version == "v02":
            payload.update(
                {
                    "eta_alpha_v02": eta_alpha_v02,
                    "eta_f_v02": eta_f_v02,
                    "eta_theta_v02": eta_theta_v02,
                    "min_spike_bonus_v02": min_spike_bonus_v02,
                }
            )
        if rfp_version == "v03":
            payload.update(
                {
                    "eta_f_v03": eta_f_v03,
                    "eta_theta_v03": eta_theta_v03,
                    "eta_alpha_v03": eta_alpha_v03,
                    "rescue_threshold_v03": rescue_threshold_v03,
                    "rescue_factor_v03": rescue_factor_v03,
                    "rescue_step_fraction": (
                        rescue_hits / total_rfp_steps if total_rfp_steps > 0 else None
                    ),
                }
            )
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

    if save_png and history_val_ce:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = "".join(c if c.isalnum() or c in "-_" else "_" for c in mode)[:40]
        png_out = plot_path or (_THIS / "outputs" / f"protocol06_fresh_{safe_mode}_{ts}.png")
        plot_training_curves(
            history_val_ce,
            history_val_rc,
            history_spike,
            VOCAB_SIZE,
            png_out,
            title_suffix=f" ({plot_title_prefix}, {mode})",
        )

    rescue_frac: Optional[float] = (
        rescue_hits / total_rfp_steps if total_rfp_steps > 0 and rfp_version == "v03" else None
    )

    m = RunMetrics(
        best_val_ce=best_val_ce,
        final_val_rc=final_rc,
        mean_spike_rate=mean_sp,
        max_grad_l2=max_g,
        epochs=epochs,
        mode=mode,
        corr_delta_pb_delta_ce=corr,
        rescue_step_fraction=rescue_frac,
    )
    return FreshRunResult(
        metrics=m,
        history_val_ce=history_val_ce,
        history_val_rc=history_val_rc,
        history_spike=history_spike,
        model=model,
    )


def assert_all_parameters_finite(model: torch.nn.Module) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)
    return len(bad) == 0, bad
