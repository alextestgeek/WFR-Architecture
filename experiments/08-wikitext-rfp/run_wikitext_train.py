"""
Experiment 08 — обучение WFRLM + RFP на char-level WikiText-2 (raw).

Протокол: **каждая эпоха — новые train-окна** из train split; **val — фиксированный holdout**
из validation split (как Exp 06 protocol fresh-train).

Запуск из корня репозитория:
  python experiments/08-wikitext-rfp/run_wikitext_train.py --quick
  python experiments/08-wikitext-rfp/run_wikitext_train.py --epochs 24 --rfp-version v03
  python experiments/08-wikitext-rfp/run_wikitext_train.py --wiki-push --rfp-version v03

Удалённый GPU (nohup, см. RULES.md §6) — важно **unbuffered** stdout в лог:
  cd ~/Desktop/WFR-Memory-Test && mkdir -p experiments/08-wikitext-rfp/outputs && \\
    nohup .venv/bin/python -u experiments/08-wikitext-rfp/run_wikitext_train.py \\
      --wiki-push --rfp-version v03 \\
      > experiments/08-wikitext-rfp/outputs/wikipush_a100.log 2>&1 &

Ожидание завершения на сервере: ``bash experiments/08-wikitext-rfp/wait_wikitext_python.sh``
  (не использовать ``pgrep -f ...run_wikitext_train.py`` в своём bash-цикле — см. скрипт).

Поведение для длинных прогонов:
  - таблица эпох: ``print(..., flush=True)`` + line-buffering stdout при редиректе;
  - JSON по эпохам (``--wiki-push`` задаёт путь) **перезаписывается после каждой эпохи**;
  - CSV пишется **в конце** (для live-метрик смотреть epoch-json или лог).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import torch

_THIS = Path(__file__).resolve()


def _repo_root() -> Path:
    if _THIS.parent.name == "08-wikitext-rfp" and _THIS.parent.parent.name == "experiments":
        return _THIS.parents[2]
    return _THIS.parent


ROOT = _repo_root()


def _configure_stdio_for_remote_logs() -> None:
    """При редиректе в файл (nohup) включаем построчную буферизацию; дополняет flush в print."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except (AttributeError, OSError, ValueError, TypeError):
            pass


_EXP06 = ROOT / "experiments" / "06-rfp-v0"
if not (_EXP06 / "run_rfp_training.py").is_file():
    _EXP06 = _THIS.parent
_EXP_DIR = _THIS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(_EXP06))

from phase0_best_config import PHASE0_FREQ_BALANCED  # noqa: E402
from wfr.losses import composite_training_loss, next_token_cross_entropy  # noqa: E402
from wfr.core import WFRNetwork  # noqa: E402
from wfr_lm import WFRLM  # noqa: E402
from wfr_rfp import apply_rfp_deltas, rfp_step, rfp_step_v01, rfp_step_v02, rfp_step_v03  # noqa: E402

from run_rfp_training import (  # noqa: E402
    ALPHA,
    BETA,
    DEVICE,
    GAMMA,
    RunMetrics,
    _phase_bias_vector,
    pearson_r,
    plot_training_curves,
    total_grad_l2_norm,
)

from wikitext_loader import WikiTextCharCorpus  # noqa: E402
from wikitext_integrity import (  # noqa: E402
    assert_train_windows_change_across_epochs,
    assert_val_holdout_reproducible,
    tensor_batches_fingerprint,
)

# Реальный текст: чуть длиннее окно, чем toy (48)
DEFAULT_SEQ_LEN = 64
DEFAULT_BATCH = 12
DEFAULT_LR = 0.02

# Длинный прогон (~минуты на GPU): больше работы на эпоху и явная таблица метрик
LONG_RUN_EPOCHS = 72
LONG_NUM_TRAIN_BATCHES = 20
LONG_NUM_VAL_BATCHES = 8
LONG_SEQ_LEN = 96

# «Дожать» WikiText на большой GPU (A100 80GB): больше слоёв, длинный контекст, тяжёлая эпоха
WIKI_PUSH_EPOCHS = 200
WIKI_PUSH_SEQ_LEN = 256
WIKI_PUSH_BATCH = 64
WIKI_PUSH_NUM_TRAIN_BATCHES = 64
WIKI_PUSH_NUM_VAL_BATCHES = 24
WIKI_PUSH_NUM_RESONANCE_LAYERS = 8


def _resonance_freqs_thresholds(num_layers: int) -> tuple[list[float], list[float]]:
    """Частоты/порога резонансных слоёв: Phase0 для n≤4, дальше геометрическое продолжение."""
    cfg = PHASE0_FREQ_BALANCED
    base_f = list(cfg["frequencies"])
    base_t = list(cfg["thresholds"])
    if num_layers <= len(base_f):
        return base_f[:num_layers], base_t[:num_layers]
    freqs = base_f[:]
    thr = base_t[:]
    ratio = freqs[-1] / max(freqs[-2], 1e-6)
    while len(freqs) < num_layers:
        freqs.append(min(120.0, freqs[-1] * ratio))
        thr.append(min(0.75, thr[-1] + 0.05))
    return freqs[:num_layers], thr[:num_layers]


def make_positions(tokens: torch.Tensor, *, pos_mode: str, vocab_size: int) -> torch.Tensor:
    """
    Варианты входа позиции (wave-phase encoder):
    - absolute: позиции 0..T-1 (рекомендуемо для LM)
    - token_as_pos: legacy/диагностика — позиции = token ids (может искажать задачу)
    """
    if tokens.dim() != 2:
        raise ValueError("tokens must be [B,T]")
    b, t = tokens.shape
    if pos_mode == "absolute":
        base = torch.arange(t, device=tokens.device, dtype=tokens.dtype).unsqueeze(0).expand(b, -1)
        return base
    if pos_mode == "absolute_modV":
        base = torch.arange(t, device=tokens.device, dtype=tokens.dtype) % int(vocab_size)
        return base.unsqueeze(0).expand(b, -1)
    if pos_mode == "token_as_pos":
        return tokens
    raise ValueError(f"Unknown pos_mode: {pos_mode}")


@torch.no_grad()
def eval_batches(
    model: WFRLM,
    batches: list[torch.Tensor],
    *,
    pos_mode: str,
    vocab_size: int,
    content_mode: str,
    content_scale: float,
) -> dict[str, float]:
    model.eval()
    ce_sum = 0.0
    acc_sum = 0.0
    rc_sum = 0.0
    sp_sum = 0.0
    n = 0
    for tok in batches:
        pos = make_positions(tok, pos_mode=pos_mode, vocab_size=vocab_size)
        if content_mode == "off":
            state = model(pos, None)
        else:
            state = model(pos, tok, content_scale=content_scale)
        ce = next_token_cross_entropy(state.logits, tok)
        pred = state.logits[:, :-1].argmax(dim=-1)
        acc = (pred == tok[:, 1:]).float().mean()
        rc = state.rc
        if rc.dim() > 0:
            rc = rc.mean()
        sp = state.spikes.mean()
        ce_sum += float(ce.item())
        acc_sum += float(acc.item())
        rc_sum += float(rc.item())
        sp_sum += float(sp.item())
        n += 1
    inv = 1.0 / max(1, n)
    return {
        "val_ce": ce_sum * inv,
        "val_acc1": acc_sum * inv,
        "val_rc": rc_sum * inv,
        "spike_rate": sp_sum * inv,
    }


@torch.no_grad()
def eval_batches_detailed(
    model: WFRLM,
    batches: list[torch.Tensor],
    *,
    pos_mode: str,
    vocab_size: int,
    content_mode: str,
    content_scale: float,
) -> dict[str, Any]:
    """Val-метрики + per-layer spike/share, min/max freq/decay."""
    model.eval()
    ce_sum = 0.0
    rc_sum = 0.0
    sp_sum = 0.0
    n = 0
    layer_sp_acc: Optional[torch.Tensor] = None
    layer_share_acc: Optional[torch.Tensor] = None
    for tok in batches:
        pos = make_positions(tok, pos_mode=pos_mode, vocab_size=vocab_size)
        if content_mode == "off":
            state = model(pos, None)
        else:
            state = model(pos, tok, content_scale=content_scale)
        ce = next_token_cross_entropy(state.logits, tok)
        rc = state.rc
        if rc.dim() > 0:
            rc = rc.mean()
        sp = state.spikes.mean()
        ce_sum += float(ce.item())
        rc_sum += float(rc.item())
        sp_sum += float(sp.item())
        n += 1
        if state.layer_spike_means is not None:
            v = state.layer_spike_means.detach().float().cpu()
            layer_sp_acc = v if layer_sp_acc is None else layer_sp_acc + v
        if state.layer_rc_share is not None:
            u = state.layer_rc_share.detach().float().cpu()
            layer_share_acc = u if layer_share_acc is None else layer_share_acc + u
    inv = 1.0 / max(1, n)
    out: dict[str, Any] = {
        "val_ce": ce_sum * inv,
        "val_rc": rc_sum * inv,
        "mean_spike_rate": sp_sum * inv,
    }
    if layer_sp_acc is not None:
        out["per_layer_spike_rate"] = [float(x) for x in (layer_sp_acc * inv).tolist()]
    else:
        out["per_layer_spike_rate"] = []
    if layer_share_acc is not None:
        out["per_layer_rc_share"] = [float(x) for x in (layer_share_acc * inv).tolist()]
    else:
        out["per_layer_rc_share"] = []
    # min/max params
    freqs = [float(layer.frequency.data.item()) for layer in model.core.resonance_layers]
    decs = [float(layer.decay.data.item()) for layer in model.core.resonance_layers]
    out["min_max_frequency"] = {"min": min(freqs), "max": max(freqs)}
    out["min_max_decay"] = {"min": min(decs), "max": max(decs)}
    return out


@torch.no_grad()
def mean_train_state_metrics(model: torch.nn.Module, batches: list[torch.Tensor]) -> dict[str, float]:
    """Средние CE, RC и energy на train-батчах после эпохи (eval) — согласовано с лоссом и теорией."""
    model.eval()
    ce_sum = rc_sum = e_sum = 0.0
    n = 0
    for batch in batches:
        # legacy helper kept for backward compatibility; main path uses explicit positions in train loop
        state = model(batch, batch)
        ce = next_token_cross_entropy(state.logits, batch)
        rc = state.rc if state.rc.dim() == 0 else state.rc.mean()
        eng = state.energy if state.energy.dim() == 0 else state.energy.mean()
        ce_sum += float(ce.item())
        rc_sum += float(rc.item())
        e_sum += float(eng.item())
        n += 1
    if n == 0:
        return {"train_ce_mean": float("nan"), "train_rc_mean": float("nan"), "train_energy_mean": float("nan")}
    inv = 1.0 / n
    return {
        "train_ce_mean": ce_sum * inv,
        "train_rc_mean": rc_sum * inv,
        "train_energy_mean": e_sum * inv,
    }


@dataclass
class WikiRunResult:
    metrics: RunMetrics
    history_val_ce: list[float]
    history_val_rc: list[float]
    history_spike: list[float]
    history_train_ce: list[float]
    history_train_rc: list[float]
    history_train_energy: list[float]
    epoch_log: list[dict[str, Any]]
    val_fingerprint: str
    manifest: dict[str, Any]


def train_wikitext_run(
    corpus: WikiTextCharCorpus,
    *,
    epochs: int,
    seq_len: int,
    batch_size: int,
    num_train_batches: int,
    num_val_batches: int,
    use_rfp: bool,
    rfp_interval: int,
    rfp_version: str,
    online_rfp: bool,
    homeostatic_always_on: bool,
    spike_rate_target: float,
    train_seed: int = 42,
    val_seed: int = 4242,
    init_seed: int = 42,
    lr: float = DEFAULT_LR,
    optimizer: str = "adam",
    grad_clip_norm: Optional[float] = None,
    log_path: Optional[Path] = None,
    log_every_epochs: int = 2,
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
    epoch_json_path: Optional[Path] = None,
    verbose_epochs: bool = False,
    integrity_checks: bool = True,
    manifest_path: Optional[Path] = None,
    epoch_csv_path: Optional[Path] = None,
    pos_mode: str = "absolute",
    content_mode: str = "normal",
    content_scale: float = 1.0,
    num_resonance_layers: int = 4,
    loss_mode: str = "full",
    readout_feat_dim: int = 3,
    readout_mlp_hidden: Optional[int] = None,
    content_neighbor_mix: bool = False,
    phase_causal_kernel: int = 1,
    readout_wave_kernel: int = 1,
) -> WikiRunResult:
    torch.manual_seed(init_seed)
    vocab_size = corpus.vocab_size
    layer_freqs, layer_thresh = _resonance_freqs_thresholds(num_resonance_layers)
    if rfp_version == "v02" and use_rfp:
        log_path = log_path or (_EXP_DIR / "outputs" / "wikitext_rfp_v02_log.json")
    elif rfp_version == "v03" and use_rfp:
        log_path = log_path or (_EXP_DIR / "outputs" / "wikitext_rfp_v03_log.json")

    val_batches = corpus.make_val_batches(num_val_batches, batch_size, seq_len, val_seed, DEVICE)
    val_fp = tensor_batches_fingerprint(val_batches)
    if integrity_checks:
        fp2 = assert_val_holdout_reproducible(
            corpus,
            num_val_batches=num_val_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            val_seed=val_seed,
            device=DEVICE,
        )
        if fp2 != val_fp:
            raise AssertionError("val fingerprint mismatch after reproducibility check")
        assert_train_windows_change_across_epochs(
            corpus,
            num_train_batches=num_train_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            train_seed=train_seed,
            device=DEVICE,
        )

    if loss_mode not in ("full", "ce_only"):
        raise ValueError("loss_mode must be 'full' or 'ce_only'")
    if optimizer not in ("adam", "adamw"):
        raise ValueError("optimizer must be 'adam' or 'adamw'")
    if readout_feat_dim < 3:
        raise ValueError("readout_feat_dim must be >= 3")
    if readout_mlp_hidden is not None and readout_mlp_hidden < 1:
        raise ValueError("readout_mlp_hidden must be >= 1 when set")
    if phase_causal_kernel < 1:
        raise ValueError("phase_causal_kernel must be >= 1")
    if readout_wave_kernel < 1:
        raise ValueError("readout_wave_kernel must be >= 1")

    cfg = PHASE0_FREQ_BALANCED
    loss_formula = f"{ALPHA}*CE + {BETA}*(1-RC_mean) + {GAMMA}*energy_mean"
    manifest: dict[str, Any] = {
        "protocol": "wikitext_char_fresh_train_fixed_val",
        "loss_mode": loss_mode,
        "loss": loss_formula if loss_mode == "full" else "CE_only (next-token cross-entropy; RC/energy not in objective)",
        "loss_full_L_reference": loss_formula,
        "ALPHA": ALPHA,
        "BETA": BETA,
        "GAMMA": GAMMA,
        "val_rc_note": (
            "RC on val is a forward diagnostic; under full L it is also partially shaped by the objective."
            if loss_mode == "full"
            else "RC on val is not optimized under CE-only training; use for diagnostics only vs full-L runs."
        ),
        "init_seed": init_seed,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "train_window_rule": "train_seed + epoch * 100_003 (Generator)",
        "val_holdout_rule": "single fixed set from val split via val_seed",
        "vocab_size": vocab_size,
        "ln_vocab": math.log(vocab_size),
        "val_batches_sha256": val_fp,
        "corpus_meta": corpus.meta,
        "num_train_batches_per_epoch": num_train_batches,
        "num_val_batches": num_val_batches,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "use_rfp": use_rfp,
        "rfp_version": rfp_version,
        "rfp_interval": rfp_interval,
        "online_rfp": online_rfp,
        "lr": lr,
        "optimizer": optimizer,
        "grad_clip_norm": grad_clip_norm,
        "integrity_checks_ran": integrity_checks,
        "pos_mode": pos_mode,
        "content_mode": content_mode,
        "content_scale": content_scale,
        "num_resonance_layers": num_resonance_layers,
        "layer_frequencies": layer_freqs,
        "layer_thresholds": layer_thresh,
        "readout_feat_dim": readout_feat_dim,
        "readout_mlp_hidden": readout_mlp_hidden,
        "content_neighbor_mix": content_neighbor_mix,
        "phase_causal_kernel": phase_causal_kernel,
        "readout_wave_kernel": readout_wave_kernel,
    }

    core = WFRNetwork(
        num_phases=16,
        num_fractal_levels=6,
        num_resonance_layers=num_resonance_layers,
        layer_frequencies=layer_freqs,
        layer_thresholds=layer_thresh,
        homeostatic_enabled=True,
        spike_rate_target=spike_rate_target,
        homeostatic_eta=cfg["homeostatic_eta"],
        homeostatic_always_on=homeostatic_always_on,
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
    ).to(DEVICE)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )

    mode = "baseline"
    if use_rfp:
        mode = "rfp_online" if online_rfp else f"rfp_every_{rfp_interval}"
        mode += f"_{rfp_version}"
    mode += "_wikitext"
    if loss_mode == "ce_only":
        mode += "_ce_only_train"

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
    history_train_ce: list[float] = []
    history_train_rc: list[float] = []
    history_train_energy: list[float] = []
    epoch_log: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    if verbose_epochs:
        hdr = (
            f"{'ep':>4} | {'tr_CE':>8} | {'tr_RC':>7} | {'tr_E':>7} | "
            f"{'val_CE':>8} | {'val_RC':>7} | {'spike':>7} | {'best_v':>8} | step"
        )
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)

    for ep in range(epochs):
        train_batches = corpus.make_train_batches_for_epoch(
            num_train_batches, batch_size, seq_len, ep, train_seed, DEVICE
        )

        model.train()
        for batch in train_batches:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            pos = make_positions(batch, pos_mode=pos_mode, vocab_size=vocab_size)
            if content_mode == "off":
                state = model(pos, None)
            else:
                state = model(pos, batch, content_scale=content_scale)
            if loss_mode == "ce_only":
                ce_loss = next_token_cross_entropy(state.logits, batch)
                ce_loss.backward()
            else:
                total, ce_loss, _ = composite_training_loss(
                    state.logits,
                    batch,
                    state.rc,
                    state.energy,
                    alpha=ALPHA,
                    beta=BETA,
                    gamma=GAMMA,
                )
                total.backward()
            g2 = total_grad_l2_norm(model)
            max_g = max(max_g, g2)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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

        # train metrics with the same pos_mode
        tm = {"train_ce_mean": 0.0, "train_rc_mean": 0.0, "train_energy_mean": 0.0}
        model.eval()
        for tb in train_batches:
            pos = make_positions(tb, pos_mode=pos_mode, vocab_size=vocab_size)
            if content_mode == "off":
                st = model(pos, None)
            else:
                st = model(pos, tb, content_scale=content_scale)
            ce = next_token_cross_entropy(st.logits, tb)
            rc = st.rc if st.rc.dim() == 0 else st.rc.mean()
            eng = st.energy if st.energy.dim() == 0 else st.energy.mean()
            tm["train_ce_mean"] += float(ce.item())
            tm["train_rc_mean"] += float(rc.item())
            tm["train_energy_mean"] += float(eng.item())
        inv = 1.0 / max(1, len(train_batches))
        tr_ce = tm["train_ce_mean"] * inv
        tr_rc = tm["train_rc_mean"] * inv
        tr_e = tm["train_energy_mean"] * inv

        ev = eval_batches(
            model,
            val_batches,
            pos_mode=pos_mode,
            vocab_size=vocab_size,
            content_mode=content_mode,
            content_scale=content_scale,
        )
        history_train_ce.append(tr_ce)
        history_train_rc.append(tr_rc)
        history_train_energy.append(tr_e)
        history_val_ce.append(ev["val_ce"])
        history_val_rc.append(ev["val_rc"])
        history_spike.append(ev["spike_rate"])
        best_val_ce = min(best_val_ce, ev["val_ce"])
        final_rc = ev["val_rc"]
        mean_sp = ev["spike_rate"]

        row = {
            "epoch": ep + 1,
            "train_ce_mean": tr_ce,
            "train_rc_mean": tr_rc,
            "train_energy_mean": tr_e,
            "val_ce": ev["val_ce"],
            "val_acc1": ev["val_acc1"],
            "val_rc": ev["val_rc"],
            "spike_rate": ev["spike_rate"],
            "best_val_ce_so_far": best_val_ce,
            "global_step": global_step,
        }
        epoch_log.append(dict(row))
        csv_rows.append(row)
        if verbose_epochs:
            print(
                f"{ep + 1:4d} | {tr_ce:8.4f} | {tr_rc:7.4f} | {tr_e:7.4f} | "
                f"{ev['val_ce']:8.4f} | {ev['val_rc']:7.4f} | {ev['spike_rate']:7.4f} | "
                f"{best_val_ce:8.4f} | {global_step}",
                flush=True,
            )
        if epoch_json_path is not None:
            epoch_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(epoch_json_path, "w", encoding="utf-8") as f:
                json.dump(epoch_log, f, indent=2, ensure_ascii=False)

        should_log = use_detailed_rfp_log and ((ep + 1) % log_every_epochs == 0 or (ep + 1) == epochs)
        if should_log:
            det = eval_batches_detailed(
                model,
                val_batches,
                pos_mode=pos_mode,
                vocab_size=vocab_size,
                content_mode=content_mode,
                content_scale=content_scale,
            )
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
            "protocol": "wikitext_char_fresh_train",
            "manifest_ref": str(manifest_path) if manifest_path else None,
            "vocab_size": vocab_size,
            "train_seed": train_seed,
            "val_seed": val_seed,
            "init_seed": init_seed,
            "corpus_meta": corpus.meta,
            "num_train_batches": num_train_batches,
            "num_val_batches": num_val_batches,
            "seq_len": seq_len,
            "batch_size": batch_size,
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
            "epoch_steps": epoch_log,
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

    if epoch_csv_path is not None and csv_rows:
        epoch_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_rows[0].keys())
        with open(epoch_csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)

    if save_png and history_val_ce:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = "".join(c if c.isalnum() or c in "-_" else "_" for c in mode)[:48]
        png_out = plot_path or (_EXP_DIR / "outputs" / f"wikitext08_curves_{safe_mode}_{ts}.png")
        plot_training_curves(
            history_val_ce,
            history_val_rc,
            history_spike,
            png_out,
            title_suffix=f" (WikiText-2 char, {mode})",
            vocab_size=vocab_size,
            history_train_ce=history_train_ce,
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
    manifest["mode"] = mode
    manifest["best_val_ce"] = best_val_ce
    manifest["final_val_rc"] = final_rc
    manifest["mean_spike_rate_end"] = mean_sp
    manifest["max_grad_l2"] = max_g
    manifest["correlation_delta_pb_delta_ce"] = corr
    manifest["rescue_step_fraction"] = rescue_frac
    manifest["num_trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return WikiRunResult(
        metrics=m,
        history_val_ce=history_val_ce,
        history_val_rc=history_val_rc,
        history_spike=history_spike,
        history_train_ce=history_train_ce,
        history_train_rc=history_train_rc,
        history_train_energy=history_train_energy,
        epoch_log=epoch_log,
        val_fingerprint=val_fp,
        manifest=manifest,
    )


def main() -> None:
    _configure_stdio_for_remote_logs()
    p = argparse.ArgumentParser(description="Experiment 08 — WikiText char LM + RFP")
    p.add_argument("--dataset-dir", type=Path, default=None, help="path to wikitext-2-raw-v1 (save_to_disk)")
    p.add_argument("--max-vocab", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="короткий прогон (мало батчей и эпох)")
    p.add_argument(
        "--long-run",
        action="store_true",
        help=f"длинный прогон: {LONG_RUN_EPOCHS} ep, seq={LONG_SEQ_LEN}, "
        f"{LONG_NUM_TRAIN_BATCHES} train batches/ep, таблица эпох + CSV + манифест",
    )
    p.add_argument(
        "--wiki-push",
        action="store_true",
        help=f"дожим WikiText на большой GPU: {WIKI_PUSH_EPOCHS} ep, seq={WIKI_PUSH_SEQ_LEN}, "
        f"batch={WIKI_PUSH_BATCH}, {WIKI_PUSH_NUM_TRAIN_BATCHES}/{WIKI_PUSH_NUM_VAL_BATCHES} батчей, "
        f"{WIKI_PUSH_NUM_RESONANCE_LAYERS} резонансных слоёв (частоты — продолжение Phase0)",
    )
    p.add_argument(
        "--num-resonance-layers",
        type=int,
        default=None,
        metavar="N",
        help="число резонансных слоёв WFRNetwork (2–32). По умолчанию: 4; с --wiki-push — 8, если не задано явно",
    )
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--num-train-batches", type=int, default=6)
    p.add_argument("--num-val-batches", type=int, default=4)
    p.add_argument("--no-rfp", action="store_true")
    p.add_argument("--rfp-interval", type=int, default=8)
    p.add_argument("--online-rfp", action="store_true")
    p.add_argument(
        "--rfp-version",
        type=str,
        default="v03",
        choices=("v0", "v01", "v02", "v03"),
    )
    p.add_argument("--no-homeostatic-always-on", action="store_true")
    p.add_argument("--spike-rate-target", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=("adam", "adamw"),
        help="Adam — как раньше (Exp 08); AdamW+betas/WD — как TinyCausalTransformer в Exp 09",
    )
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        metavar="MAX",
        help="если задано — clip_grad_norm_ перед шагом (Exp 09 baseline: 1.0)",
    )
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=4242)
    p.add_argument("--init-seed", type=int, default=42, help="инициализация весов WFRLM/ядра (воспроизводимость)")
    p.add_argument("--log-json", type=Path, default=None)
    p.add_argument("--log-every-epochs", type=int, default=2)
    p.add_argument("--epoch-json", type=Path, default=None, help="пошаговый лог по эпохам (train/val CE)")
    p.add_argument("--epoch-csv", type=Path, default=None, help="CSV с метриками по эпохам")
    p.add_argument("--manifest-json", type=Path, default=None, help="манифест протокола и гиперпараметров")
    p.add_argument("--verbose-epochs", action="store_true", help="печать таблицы по эпохам в stdout")
    p.add_argument("--quiet-epochs", action="store_true", help="не печатать таблицу (по умолчанию для long-run — печать)")
    p.add_argument("--no-integrity", action="store_true", help="отключить assert val/train (не для отчётов)")
    p.add_argument(
        "--pos-mode",
        type=str,
        default="absolute",
        choices=("absolute", "absolute_modV", "token_as_pos"),
        help="как формировать positions для WPE (по умолчанию: absolute 0..T-1)",
    )
    p.add_argument(
        "--content-mode",
        type=str,
        default="normal",
        choices=("normal", "off"),
        help="normal: content_delta из token embedding; off: отключить content_delta (контроль)",
    )
    p.add_argument("--content-scale", type=float, default=1.0, help="масштаб content_delta (normal mode)")
    p.add_argument(
        "--loss-mode",
        type=str,
        default="full",
        choices=("full", "ce_only"),
        help="full: L из Phase 1 (CE+RC+energy); ce_only: только next-token CE для backward (честный baseline)",
    )
    p.add_argument(
        "--readout-feat-dim",
        type=int,
        default=3,
        metavar="D",
        help="ширина вектора признаков перед logits (>=3; дорожная карта B1, по умолчанию 3 как раньше)",
    )
    p.add_argument(
        "--readout-mlp-hidden",
        type=int,
        default=None,
        metavar="H",
        help="если задано — MLP readout D→H→GELU→vocab вместо одного Linear (дорожная карта B1)",
    )
    p.add_argument(
        "--content-neighbor-mix",
        action="store_true",
        help="WFRLM: причинно смести content_delta[t] += α·δ[t−1] (α обучаемый; ядро не трогаем)",
    )
    p.add_argument(
        "--phase-causal-kernel",
        type=int,
        default=1,
        metavar="K",
        help="WFRNetwork: каузальная depthwise Conv1d по времени на фазах после WPE+content; K=1 выкл.",
    )
    p.add_argument(
        "--readout-wave-kernel",
        type=int,
        default=1,
        metavar="K",
        help="WFRLM: каузальная Conv1d(1,1,K) на standing_wave перед признаками readout; K=1 выкл.",
    )
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args()

    corpus = WikiTextCharCorpus.from_hf_disk(args.dataset_dir, max_vocab=args.max_vocab)
    out_dir = _EXP_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.save_vocab_json(out_dir / "wikitext08_vocab_meta.json")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    num_layers = args.num_resonance_layers
    if num_layers is None:
        num_layers = WIKI_PUSH_NUM_RESONANCE_LAYERS if args.wiki_push else 4
    if num_layers < 2 or num_layers > 32:
        raise SystemExit("--num-resonance-layers must be between 2 and 32")

    wiki_plot: Optional[Path] = None
    if args.wiki_push:
        epochs = WIKI_PUSH_EPOCHS
        n_train = WIKI_PUSH_NUM_TRAIN_BATCHES
        n_val = WIKI_PUSH_NUM_VAL_BATCHES
        seq_len = WIKI_PUSH_SEQ_LEN
        args.batch_size = WIKI_PUSH_BATCH
        log_every = 2
        verbose_epochs = not args.quiet_epochs
        epoch_csv = args.epoch_csv or (out_dir / f"wikitext08_wikipush_epochs_{ts}.csv")
        manifest_path = args.manifest_json or (out_dir / f"wikitext08_wikipush_manifest_{ts}.json")
        wiki_plot = out_dir / f"wikitext08_wikipush_curves_{ts}.png"
    elif args.long_run:
        epochs = LONG_RUN_EPOCHS
        n_train = LONG_NUM_TRAIN_BATCHES
        n_val = LONG_NUM_VAL_BATCHES
        seq_len = LONG_SEQ_LEN
        log_every = 1
        verbose_epochs = not args.quiet_epochs
        epoch_csv = args.epoch_csv or (out_dir / f"wikitext08_epochs_{ts}.csv")
        manifest_path = args.manifest_json or (out_dir / f"wikitext08_manifest_{ts}.json")
    elif args.quick:
        epochs = min(args.epochs, 8)
        n_train = min(args.num_train_batches, 4)
        n_val = min(args.num_val_batches, 2)
        seq_len = args.seq_len
        log_every = args.log_every_epochs
        verbose_epochs = args.verbose_epochs
        epoch_csv = args.epoch_csv
        manifest_path = args.manifest_json
    else:
        epochs = args.epochs
        n_train = args.num_train_batches
        n_val = args.num_val_batches
        seq_len = args.seq_len
        log_every = args.log_every_epochs
        verbose_epochs = args.verbose_epochs
        epoch_csv = args.epoch_csv
        manifest_path = args.manifest_json

    epoch_json = args.epoch_json or (
        (out_dir / f"wikitext08_wikipush_epoch_{ts}.json") if args.wiki_push else (out_dir / "wikitext08_epoch_log.json")
    )

    r = train_wikitext_run(
        corpus,
        epochs=epochs,
        seq_len=seq_len,
        batch_size=args.batch_size,
        num_train_batches=n_train,
        num_val_batches=n_val,
        use_rfp=not args.no_rfp,
        rfp_interval=args.rfp_interval,
        rfp_version=args.rfp_version,
        online_rfp=args.online_rfp,
        homeostatic_always_on=not args.no_homeostatic_always_on,
        spike_rate_target=args.spike_rate_target,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
        init_seed=args.init_seed,
        lr=args.lr,
        optimizer=args.optimizer,
        grad_clip_norm=args.grad_clip_norm,
        log_path=args.log_json,
        log_every_epochs=log_every,
        save_png=not args.no_png,
        epoch_json_path=epoch_json,
        verbose_epochs=verbose_epochs,
        integrity_checks=not args.no_integrity,
        manifest_path=manifest_path,
        epoch_csv_path=epoch_csv,
        pos_mode=args.pos_mode,
        content_mode=args.content_mode,
        content_scale=args.content_scale,
        num_resonance_layers=num_layers,
        plot_path=wiki_plot,
        loss_mode=args.loss_mode,
        readout_feat_dim=args.readout_feat_dim,
        readout_mlp_hidden=args.readout_mlp_hidden,
        content_neighbor_mix=args.content_neighbor_mix,
        phase_causal_kernel=args.phase_causal_kernel,
        readout_wave_kernel=args.readout_wave_kernel,
    )

    summary = {
        "metrics": asdict(r.metrics),
        "vocab_size": corpus.vocab_size,
        "ln_vocab": math.log(corpus.vocab_size),
        "final_train_ce": r.history_train_ce[-1] if r.history_train_ce else None,
        "final_train_rc": r.history_train_rc[-1] if r.history_train_rc else None,
        "final_train_energy": r.history_train_energy[-1] if r.history_train_energy else None,
        "val_batches_sha256": r.val_fingerprint,
        "artifacts": {
            "epoch_log": str(epoch_json),
            "epoch_csv": str(epoch_csv) if epoch_csv else None,
            "manifest_json": str(manifest_path) if manifest_path else None,
            "vocab_meta": str(out_dir / "wikitext08_vocab_meta.json"),
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    with open(out_dir / "wikitext08_last_run.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
