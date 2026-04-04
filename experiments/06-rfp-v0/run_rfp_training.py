"""
Experiment 06 — RFP v0 / v0.2: Adam + composite loss + RFP-шаг.

Запуск из корня репозитория или с добавлением ROOT в PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))

from phase0_best_config import PHASE0_FREQ_BALANCED  # noqa: E402
from wfr.losses import ALPHA, BETA, GAMMA, composite_training_loss, next_token_cross_entropy  # noqa: E402
from wfr.core import WFRNetwork  # noqa: E402
from wfr_lm import WFRLM  # noqa: E402
from wfr_rfp import apply_rfp_deltas, rfp_step, rfp_step_v01, rfp_step_v02, rfp_step_v03  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 32
SEQ_LEN = 48
BATCH_SIZE = 16
SEED = 42


def pearson_r(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


@dataclass
class RunMetrics:
    best_val_ce: float
    final_val_rc: float
    mean_spike_rate: float
    max_grad_l2: float
    epochs: int
    mode: str
    corr_delta_pb_delta_ce: Optional[float] = None
    rescue_step_fraction: Optional[float] = None


def total_grad_l2_norm(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(s) if s > 0 else 0.0


def _param_min_max(network: WFRNetwork) -> tuple[dict[str, float], dict[str, float]]:
    freqs = [float(layer.frequency.data.item()) for layer in network.resonance_layers]
    decs = [float(layer.decay.data.item()) for layer in network.resonance_layers]
    return (
        {"min": min(freqs), "max": max(freqs)},
        {"min": min(decs), "max": max(decs)},
    )


@torch.no_grad()
def evaluate(
    model: WFRLM,
    batches: list[torch.Tensor],
) -> dict:
    model.eval()
    ce_sum = 0.0
    rc_sum = 0.0
    sp_sum = 0.0
    n = 0
    for batch in batches:
        state = model(batch, batch)
        ce = next_token_cross_entropy(state.logits, batch)
        rc = state.rc
        if rc.dim() > 0:
            rc = rc.mean()
        sp = state.spikes.mean()
        ce_sum += float(ce.item())
        rc_sum += float(rc.item())
        sp_sum += float(sp.item())
        n += 1
    return {
        "val_ce": ce_sum / n,
        "val_rc": rc_sum / n,
        "spike_rate": sp_sum / n,
    }


@torch.no_grad()
def evaluate_detailed(
    model: WFRLM,
    batches: list[torch.Tensor],
) -> dict[str, Any]:
    """Val-метрики + per-layer spike / доля энергии резонанса, min/max freq & decay."""
    model.eval()
    ce_sum = 0.0
    rc_sum = 0.0
    sp_sum = 0.0
    n = 0
    layer_sp_acc: Optional[torch.Tensor] = None
    layer_share_acc: Optional[torch.Tensor] = None

    for batch in batches:
        state = model(batch, batch)
        ce = next_token_cross_entropy(state.logits, batch)
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

    out: dict[str, Any] = {
        "val_ce": ce_sum / n,
        "val_rc": rc_sum / n,
        "mean_spike_rate": sp_sum / n,
    }
    if layer_sp_acc is not None:
        layer_sp_acc = layer_sp_acc / n
        out["per_layer_spike_rate"] = [float(x) for x in layer_sp_acc.tolist()]
    else:
        out["per_layer_spike_rate"] = []
    if layer_share_acc is not None:
        layer_share_acc = layer_share_acc / n
        out["per_layer_rc_share"] = [float(x) for x in layer_share_acc.tolist()]
    else:
        out["per_layer_rc_share"] = []

    fmm, dmm = _param_min_max(model.core)
    out["min_max_frequency"] = fmm
    out["min_max_decay"] = dmm
    return out


def _phase_bias_vector(model: WFRLM) -> torch.Tensor:
    return torch.cat(
        [model.core.resonance_layers[i].phase_bias.data.flatten() for i in range(len(model.core.resonance_layers))]
    )


def plot_training_curves(
    history_val_ce: list[float],
    history_val_rc: list[float],
    history_spike: list[float],
    path_png: Path,
    title_suffix: str = "",
    *,
    vocab_size: Optional[int] = None,
    history_train_ce: Optional[list[float]] = None,
) -> None:
    """Val CE / RC / spike rate по эпохам + линия ln(V) для CE (как в Exp 05).

    ``history_train_ce`` (опционально): средний train CE по эпохам — рисуется вместе с val на первой панели.
    """
    if not history_val_ce:
        return
    path_png.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history_val_ce) + 1))
    v = vocab_size if vocab_size is not None else VOCAB_SIZE
    ln_v = math.log(v)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    if history_train_ce is not None and len(history_train_ce) == len(history_val_ce):
        axes[0].plot(epochs, history_train_ce, color="#9333ea", label="train CE (epoch mean)", alpha=0.85)
    axes[0].plot(epochs, history_val_ce, color="#16a34a", label="val CE")
    axes[0].axhline(ln_v, color="gray", linestyle=":", label=f"ln V = {ln_v:.3f}")
    axes[0].set_ylabel("CE")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Experiment 06 — validation" + title_suffix)

    axes[1].plot(epochs, history_val_rc, color="#2563eb", label="val RC")
    axes[1].set_ylabel("RC")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history_spike, color="#ea580c", label="mean spike rate (standing wave)")
    axes[2].axhline(0.25, color="purple", linestyle="--", alpha=0.5, label="target ~0.25")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("spike rate")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def plot_ab_modes_bar(
    rows: list[dict[str, Any]],
    path_png: Path,
    title: str = "Experiment 06 — A/B best val CE",
    *,
    vocab_size: Optional[int] = None,
) -> None:
    """Столбчатая диаграмма best val CE по режимам."""
    if not rows:
        return
    path_png.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["mode"] for r in rows]
    ces = [r["best_val_ce"] for r in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5))
    bars = ax.bar(x, ces, color=["#64748b" if "only" in lb else "#0d9488" for lb in labels])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("best val CE")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    v = vocab_size if vocab_size is not None else VOCAB_SIZE
    ln_v = math.log(v)
    ax.axhline(ln_v, color="gray", linestyle=":", label=f"ln V = {ln_v:.3f}")
    ax.legend(loc="upper right", fontsize=8)
    for i, (b, v) in enumerate(zip(bars, ces)):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def train_run(
    epochs: int,
    use_rfp: bool,
    rfp_interval: int,
    rfp_version: str,
    online_rfp: bool,
    homeostatic_always_on: bool,
    spike_rate_target: float,
    lr: float = 0.05,
    seed: int = SEED,
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
) -> RunMetrics:
    if rfp_version == "v02" and use_rfp:
        log_path = log_path or (_EXP_DIR / "outputs" / "rfp_v02_log.json")
    elif rfp_version == "v03" and use_rfp:
        log_path = log_path or (_EXP_DIR / "outputs" / "rfp_v03_log.json")

    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

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

    def make_batches(num: int) -> list[torch.Tensor]:
        return [
            torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=gen, device=DEVICE)
            for _ in range(num)
        ]

    train_batches = make_batches(8)
    val_batches = make_batches(3)

    max_g = 0.0
    best_val_ce = float("inf")
    final_rc = 0.0
    mean_sp = 0.0
    global_step = 0

    mode = "baseline"
    if use_rfp:
        mode = "rfp_online" if online_rfp else f"rfp_every_{rfp_interval}"
        mode += f"_{rfp_version}"

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
        model.train()
        for batch in train_batches:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            state = model(batch, batch)
            total, ce_loss, _ = composite_training_loss(
                state.logits, batch, state.rc, state.energy,
                alpha=ALPHA, beta=BETA, gamma=GAMMA,
            )
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
        png_out = plot_path or (_EXP_DIR / "outputs" / f"rfp06_curves_{safe_mode}_{ts}.png")
        plot_training_curves(
            history_val_ce,
            history_val_rc,
            history_spike,
            png_out,
            title_suffix=f" ({mode})",
            vocab_size=VOCAB_SIZE,
        )

    rescue_frac: Optional[float] = (
        rescue_hits / total_rfp_steps if total_rfp_steps > 0 and rfp_version == "v03" else None
    )

    return RunMetrics(
        best_val_ce=best_val_ce,
        final_val_rc=final_rc,
        mean_spike_rate=mean_sp,
        max_grad_l2=max_g,
        epochs=epochs,
        mode=mode,
        corr_delta_pb_delta_ce=corr,
        rescue_step_fraction=rescue_frac,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="RFP v0 / v0.2 training (Experiment 06)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--no-rfp", action="store_true", help="Adam only (baseline)")
    p.add_argument("--rfp-interval", type=int, default=8)
    p.add_argument("--online-rfp", action="store_true")
    p.add_argument(
        "--rfp-version",
        type=str,
        default="v0",
        choices=("v0", "v01", "v02", "v03"),
        help="v0 | v01 | v02 | v03 (RFP v0.3 per-layer + rescue)",
    )
    p.add_argument(
        "--no-homeostatic-always-on",
        action="store_true",
        help="отключить homeostatic в train (по умолчанию включён для RFP v0)",
    )
    p.add_argument("--spike-rate-target", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--log-json",
        type=Path,
        default=None,
        help="полный лог v0.2/v0.3 (по умолчанию rfp_v02_log.json / rfp_v03_log.json)",
    )
    p.add_argument("--log-every-epochs", type=int, default=4)
    p.add_argument("--eta-alpha-v02", type=float, default=3e-5)
    p.add_argument("--eta-f-v02", type=float, default=8e-5)
    p.add_argument("--eta-theta-v02", type=float, default=2e-4)
    p.add_argument("--min-spike-bonus-v02", type=float, default=0.12)
    p.add_argument("--eta-f-v03", type=float, default=1.2e-4)
    p.add_argument("--eta-theta-v03", type=float, default=3e-4)
    p.add_argument("--eta-alpha-v03", type=float, default=8e-6)
    p.add_argument("--rescue-threshold-v03", type=float, default=0.12)
    p.add_argument("--rescue-factor-v03", type=float, default=0.45)
    p.add_argument("--no-png", action="store_true", help="не сохранять PNG кривых в outputs/")
    args = p.parse_args()

    _out = Path(__file__).parent / "outputs"
    default_log_v02 = _out / "rfp_v02_log.json"
    default_log_v03 = _out / "rfp_v03_log.json"
    log_path = args.log_json
    if args.rfp_version == "v02" and not args.no_rfp:
        log_path = log_path or default_log_v02
    elif args.rfp_version == "v03" and not args.no_rfp:
        log_path = log_path or default_log_v03

    m = train_run(
        epochs=args.epochs,
        use_rfp=not args.no_rfp,
        rfp_interval=args.rfp_interval,
        rfp_version=args.rfp_version,
        online_rfp=args.online_rfp,
        homeostatic_always_on=not args.no_homeostatic_always_on,
        spike_rate_target=args.spike_rate_target,
        lr=args.lr,
        seed=args.seed,
        log_path=log_path,
        log_every_epochs=args.log_every_epochs,
        eta_alpha_v02=args.eta_alpha_v02,
        eta_f_v02=args.eta_f_v02,
        eta_theta_v02=args.eta_theta_v02,
        min_spike_bonus_v02=args.min_spike_bonus_v02,
        eta_f_v03=args.eta_f_v03,
        eta_theta_v03=args.eta_theta_v03,
        eta_alpha_v03=args.eta_alpha_v03,
        rescue_threshold_v03=args.rescue_threshold_v03,
        rescue_factor_v03=args.rescue_factor_v03,
        save_png=not args.no_png,
    )
    d = asdict(m)
    print(json.dumps(d, indent=2))
    out = args.out_json or (Path(__file__).parent / "outputs" / "last_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


if __name__ == "__main__":
    main()
