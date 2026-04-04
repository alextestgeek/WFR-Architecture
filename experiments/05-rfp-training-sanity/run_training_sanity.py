"""
Experiment 05 — RFP Training Sanity (Phase 1)
============================================

Режимы:
  (по умолчанию) Расширенный: полная целевая функция L из теории (§6),
    фиксированные train/val батчи, метрики компонент.
  --quick         Быстрый sanity: один батч, только CE (как в v0).

См. docs/10-phase-1-plan.md и experiments/05-rfp-training-sanity/README.md.
Полноценный двухпрогонный протокол (полная L vs CE-only): `run_full_training.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_root = Path(__file__).resolve().parent
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_root))
_smoke = _root.parent / "00-smoke-test"
sys.path.insert(0, str(_smoke))
from phase0_best_config import PHASE0_FREQ_BALANCED
from wfr.core import WFRNetwork

from wfr_losses import compute_loss, energy_cost, rc_penalty, task_loss_ce

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Параметры данных ---
VOCAB_SIZE = 32
SEQ_LEN = 48
BATCH_SIZE = 16
NUM_TRAIN_BATCHES = 8
NUM_VAL_BATCHES = 3
# Длинное обучение: проверка сходимости loss по эпохам + PNG-кривые
EPOCHS = 120
LR = 0.05
SEED = 42

# Коэффициенты L = alpha * L_task + beta * (1 - RC) + gamma * L_energy (теория §6)
ALPHA = 1.0
BETA = 0.15
GAMMA = 0.1

# Быстрый режим — длиннее, чтобы кривая на PNG была информативной
QUICK_STEPS = 280

# Единый конфиг Phase 0 — см. experiments/00-smoke-test/phase0_best_config.py
BEST_CONFIG = PHASE0_FREQ_BALANCED


class WFRNextTokenProbe(nn.Module):
    """token_ids подаются в WPE как индексы; голова предсказывает следующий токен."""

    def __init__(self, vocab_size: int, num_phases: int = 16, num_fractal_levels: int = 6, num_layers: int = 4):
        super().__init__()
        self.wfr = WFRNetwork(
            num_phases=num_phases,
            num_fractal_levels=num_fractal_levels,
            num_resonance_layers=num_layers,
            layer_frequencies=BEST_CONFIG["frequencies"][:num_layers],
            layer_thresholds=BEST_CONFIG["thresholds"][:num_layers],
            homeostatic_enabled=False,
            spike_rate_target=0.10,
            homeostatic_eta=0.01,
        )
        self.wfr.target_mode = BEST_CONFIG["target_mode"]
        self.head = nn.Linear(num_phases, vocab_size)

    def forward(self, token_ids: torch.Tensor):
        out = self.wfr(token_ids)
        logits = self.head(out["phases"])
        return logits, out


def total_grad_l2_norm(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(s) if s > 0 else 0.0


def max_grad_abs(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total = max(total, p.grad.detach().abs().max().item())
    return total


@torch.no_grad()
def evaluate_batches(
    model: WFRNextTokenProbe,
    batches: list[torch.Tensor],
    alpha: float,
    beta: float,
    gamma: float,
    task_only: bool,
) -> dict:
    model.eval()
    sums = {"total": 0.0, "ce": 0.0, "rc_term": 0.0, "energy": 0.0}
    for batch in batches:
        logits, out = model(batch)
        ce = task_loss_ce(logits, batch)
        if task_only:
            t = float((alpha * ce).item())
            sums["total"] += t
            sums["ce"] += float(ce.item())
        else:
            rc_term = rc_penalty(out)
            energy = energy_cost(out)
            total = alpha * ce + beta * rc_term + gamma * energy
            sums["total"] += float(total.item())
            sums["ce"] += float(ce.item())
            sums["rc_term"] += float(rc_term.item())
            sums["energy"] += float(energy.item())
    n = len(batches)
    return {k: v / n for k, v in sums.items()}


def train_step(
    model: WFRNextTokenProbe,
    batch_tokens: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    beta: float,
    gamma: float,
    task_only: bool,
) -> tuple[float, dict, float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, out = model(batch_tokens)
    loss, _parts = compute_loss(logits, batch_tokens, out, alpha, beta, gamma, task_only)
    loss.backward()
    g_abs = max_grad_abs(model)
    g_l2 = total_grad_l2_norm(model)
    optimizer.step()
    ce = task_loss_ce(logits, batch_tokens)
    with torch.no_grad():
        det = {
            "ce": float(ce.item()),
            "total": float(loss.item()),
        }
        if not task_only:
            det["rc_term"] = float(rc_penalty(out).item())
            det["energy"] = float(energy_cost(out).item())
    return float(loss.item()), det, g_abs, g_l2


def build_fixed_batches(num_batches: int, gen: torch.Generator) -> list[torch.Tensor]:
    return [
        torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=gen, device=DEVICE) for _ in range(num_batches)
    ]


def plot_enhanced_curves(
    history_train_total: list[float],
    history_val: list[dict],
    ln_v: float,
    path_png: Path,
    title_suffix: str = "",
) -> None:
    epochs = np.arange(1, len(history_train_total) + 1)
    val_total = [v["total"] for v in history_val]
    val_ce = [v["ce"] for v in history_val]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history_train_total, label="train L (mean / epoch)", color="#2563eb")
    ax.plot(epochs, val_total, label="val total L", color="#ea580c")
    ax.plot(epochs, val_ce, label="val CE", color="#16a34a", linestyle="--")
    ax.axhline(ln_v, color="gray", linestyle=":", label=f"ln V = {ln_v:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Experiment 05 — enhanced: train vs validation" + title_suffix)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def plot_enhanced_components(
    history_val: list[dict],
    path_png: Path,
    title: str,
) -> None:
    """Val: CE, (1−RC), energy — теория §6, телеметрия полной L на val."""
    epochs = np.arange(1, len(history_val) + 1)
    val_ce = [v["ce"] for v in history_val]
    val_rc = [v["rc_term"] for v in history_val]
    val_en = [v["energy"] for v in history_val]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epochs, val_ce, color="#16a34a")
    axes[0].set_ylabel("val CE")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, val_rc, color="#c2410c")
    axes[1].set_ylabel("val (1−RC̄)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, val_en, color="#7c3aed")
    axes[2].set_ylabel("val energy (ρ̄ spike)")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def plot_quick_losses(losses: list[float], ln_v: float, path_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#2563eb", linewidth=0.8)
    ax.axhline(ln_v, color="gray", linestyle=":", label=f"ln V = {ln_v:.3f}")
    ax.set_xlabel("step")
    ax.set_ylabel("CE (fixed batch)")
    ax.set_title("Experiment 05 — quick: cross-entropy vs step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def run_quick(skip_precheck: bool = False) -> int:
    """Один батч, только CE — минимальная проверка как в первой версии скрипта."""
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model = WFRNextTokenProbe(VOCAB_SIZE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED)
    fixed_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=g, device=DEVICE)

    if not skip_precheck:
        from phase1_checks import precheck_must_pass, print_precheck, run_all_prechecks

        pre = run_all_prechecks(model, fixed_tokens, ALPHA, BETA, GAMMA, DEVICE)
        print_precheck(pre)
        if not precheck_must_pass(pre):
            print("PRECHECK FAILED — обучение не запускается.")
            return 2

    losses = []
    grad_ok = False
    for step in range(QUICK_STEPS):
        loss_val, _, g_abs, _ = train_step(model, fixed_tokens, opt, ALPHA, BETA, GAMMA, task_only=True)
        losses.append(loss_val)
        if not math.isfinite(loss_val):
            break
        if step == 0:
            grad_ok = g_abs > 1e-12

    initial = losses[0] if losses else float("nan")
    final = losses[-1] if losses else float("nan")
    random_guess_ce = math.log(VOCAB_SIZE)
    pass_finite = all(math.isfinite(x) for x in losses)
    pass_grad = grad_ok
    pass_learn = (final < initial) or (final < random_guess_ce - 0.05)
    passed = pass_finite and pass_grad and pass_learn

    result = {
        "mode": "quick",
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "steps": QUICK_STEPS,
            "lr": LR,
            "seed": SEED,
            "loss": "CE_only",
        },
        "loss_initial": round(initial, 6),
        "loss_final": round(final, 6),
        "loss_min": round(min(losses), 6) if losses else None,
        "loss_curve": [round(x, 6) for x in losses],
        "random_baseline_ce_ln_v": round(random_guess_ce, 6),
        "grad_nonzero_after_step0": bool(grad_ok),
        "criteria": {
            "finite_losses": pass_finite,
            "nonzero_grad": pass_grad,
            "learning": pass_learn,
        },
        "pass": bool(passed),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = OUTPUT_DIR / f"training_sanity_quick_{ts}.json"
    png_path = OUTPUT_DIR / f"training_sanity_quick_curve_{ts}.png"
    plot_quick_losses(losses, random_guess_ce, png_path)

    result["artifacts"] = {"json": str(json_path.name), "png_curve": str(png_path.name)}
    _write_json(json_path, result)
    _print_summary("quick", passed, initial, final, random_guess_ce, grad_ok, str(json_path))
    print(f"PNG: {png_path}")
    return 0 if passed else 1


def execute_enhanced_training(
    num_epochs: int | None = None,
    skip_precheck: bool = False,
    strict_learning: bool = False,
    task_only: bool = False,
    train_batches: list | None = None,
    val_batches: list | None = None,
) -> tuple[dict, int]:
    """
    Одна сессия enhanced: полная L на train или только CE (task_only), val всегда логируется
    с полной разложимостью (CE, RC, energy) для сравнения с теорией §6.

    Возвращает (result_dict, exit_code): 0 = PASS по критериям, 1 = FAIL обучения, 2 = precheck FAIL.
    Файлы не пишет — см. run_enhanced / run_full_training_protocol.
    """
    n_ep = num_epochs if num_epochs is not None else EPOCHS

    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    if train_batches is None or val_batches is None:
        g = torch.Generator(device=DEVICE)
        g.manual_seed(SEED)
        train_batches = build_fixed_batches(NUM_TRAIN_BATCHES, g)
        g2 = torch.Generator(device=DEVICE)
        g2.manual_seed(SEED + 99991)
        val_batches = build_fixed_batches(NUM_VAL_BATCHES, g2)

    model = WFRNextTokenProbe(VOCAB_SIZE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    precheck_report: dict | None = None
    if not skip_precheck:
        from phase1_checks import precheck_must_pass, print_precheck, run_all_prechecks

        precheck_report = run_all_prechecks(model, train_batches[0], ALPHA, BETA, GAMMA, DEVICE)
        print_precheck(precheck_report)
        if not precheck_must_pass(precheck_report):
            print("PRECHECK FAILED — обучение не запускается.")
            return (
                {
                    "mode": "enhanced_task_only" if task_only else "enhanced",
                    "precheck_failed": True,
                    "pass": False,
                },
                2,
            )

    # Val-метрики всегда с полным разложением L (даже при обучении только по CE)
    val_start = evaluate_batches(model, val_batches, ALPHA, BETA, GAMMA, task_only=False)

    history_train_total: list[float] = []
    history_train_rc: list[float] = []
    history_train_en: list[float] = []
    history_val: list[dict] = []
    grad_ok = False
    grad_l2_step0 = 0.0

    for epoch in range(n_ep):
        epoch_losses = []
        for bi, batch in enumerate(train_batches):
            loss_val, det, g_abs, g_l2 = train_step(
                model, batch, opt, ALPHA, BETA, GAMMA, task_only=task_only
            )
            epoch_losses.append(det)
            if epoch == 0 and bi == 0:
                grad_ok = g_abs > 1e-12
                grad_l2_step0 = g_l2

        avg_train = sum(d["total"] for d in epoch_losses) / len(epoch_losses)
        history_train_total.append(avg_train)
        if task_only:
            history_train_rc.append(0.0)
            history_train_en.append(0.0)
        else:
            history_train_rc.append(sum(d.get("rc_term", 0.0) for d in epoch_losses) / len(epoch_losses))
            history_train_en.append(sum(d.get("energy", 0.0) for d in epoch_losses) / len(epoch_losses))

        val_metrics = evaluate_batches(model, val_batches, ALPHA, BETA, GAMMA, task_only=False)
        history_val.append(val_metrics)

    val_end = history_val[-1]
    random_guess_ce = math.log(VOCAB_SIZE)

    val_ce_list = [v["ce"] for v in history_val]
    val_total_list = [v["total"] for v in history_val]
    best_val_ce = min(val_ce_list)
    best_epoch_ce = int(np.argmin(val_ce_list)) + 1
    best_val_total = min(val_total_list)
    best_epoch_total = int(np.argmin(val_total_list)) + 1

    pass_finite = all(math.isfinite(x) for x in history_train_total) and all(
        math.isfinite(v["total"]) for v in history_val
    )
    pass_grad = grad_ok
    pass_learn_val_total = val_end["total"] < val_start["total"] - 1e-4
    pass_learn_val_ce = val_end["ce"] < val_start["ce"] - 1e-4
    pass_learn_ce_vs_random_end = val_end["ce"] < random_guess_ce - 0.02
    pass_learn_ce_vs_random_best = best_val_ce < random_guess_ce - 0.02
    pass_learn = (
        pass_learn_val_total
        or pass_learn_val_ce
        or pass_learn_ce_vs_random_end
        or pass_learn_ce_vs_random_best
    )

    strict_ok = (not strict_learning) or pass_learn_ce_vs_random_best
    passed = pass_finite and pass_grad and pass_learn and strict_ok

    mode = "enhanced_task_only" if task_only else "enhanced"
    loss_desc = "CE_only_train_val_full_decomposition" if task_only else "alpha*CE + beta*(1-RC) + gamma*energy"

    result = {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "num_train_batches": NUM_TRAIN_BATCHES,
            "num_val_batches": NUM_VAL_BATCHES,
            "epochs": n_ep,
            "lr": LR,
            "seed": SEED,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "loss_train": loss_desc,
            "task_only_train": task_only,
            "strict_learning": strict_learning,
            "phase0_config_module": "experiments/00-smoke-test/phase0_best_config.py",
        },
        "val_start": {k: round(v, 6) for k, v in val_start.items()},
        "val_end": {k: round(v, 6) for k, v in val_end.items()},
        "best_val_ce": round(best_val_ce, 6),
        "best_epoch_ce": best_epoch_ce,
        "best_val_total": round(best_val_total, 6),
        "best_epoch_total": best_epoch_total,
        "train_total_first_epoch": round(history_train_total[0], 6),
        "train_total_last_epoch": round(history_train_total[-1], 6),
        "history": {
            "train_total_per_epoch": [round(x, 6) for x in history_train_total],
            "train_rc_term_per_epoch": [round(x, 6) for x in history_train_rc],
            "train_energy_per_epoch": [round(x, 6) for x in history_train_en],
            "val_total_per_epoch": [round(v["total"], 6) for v in history_val],
            "val_ce_per_epoch": [round(v["ce"], 6) for v in history_val],
            "val_rc_term_per_epoch": [round(v["rc_term"], 6) for v in history_val],
            "val_energy_per_epoch": [round(v["energy"], 6) for v in history_val],
        },
        "random_baseline_ce_ln_v": round(random_guess_ce, 6),
        "grad_nonzero_after_step0": bool(grad_ok),
        "grad_l2_norm_step0": round(grad_l2_step0, 8),
        "criteria": {
            "finite_losses": pass_finite,
            "nonzero_grad": pass_grad,
            "learning": pass_learn,
            "strict_learning_best_ce_below_random": strict_ok if strict_learning else None,
            "learning_breakdown": {
                "val_total_decreased": pass_learn_val_total,
                "val_ce_decreased": pass_learn_val_ce,
                "val_ce_below_random_at_end": pass_learn_ce_vs_random_end,
                "val_ce_below_random_best_epoch": pass_learn_ce_vs_random_best,
            },
        },
        "pass": bool(passed),
    }
    if precheck_report is not None:
        result["precheck"] = {k: {"ok": v[0], "msg": v[1]} for k, v in precheck_report.items()}

    code = 0 if passed else 1
    return result, code


def run_enhanced(
    num_epochs: int | None = None,
    skip_precheck: bool = False,
    strict_learning: bool = False,
    task_only: bool = False,
    artifact_prefix: str = "training_sanity_enhanced",
    write_artifacts: bool = True,
) -> int:
    """Полная L (или task_only), train/val; история loss и компонент §6; PNG основной + компоненты val."""
    n_ep = num_epochs if num_epochs is not None else EPOCHS
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = OUTPUT_DIR / f"{artifact_prefix}_{ts}.json"
    png_path = OUTPUT_DIR / f"{artifact_prefix}_curves_{ts}.png"
    png_comp_path = OUTPUT_DIR / f"{artifact_prefix}_val_components_{ts}.png"

    result, code = execute_enhanced_training(
        num_epochs=n_ep,
        skip_precheck=skip_precheck,
        strict_learning=strict_learning,
        task_only=task_only,
    )
    if code == 2:
        return 2

    random_guess_ce = float(result["random_baseline_ce_ln_v"])
    history_train = result["history"]["train_total_per_epoch"]
    # Восстановить history_val из result — нет в result напрямую, только per-epoch списки
    val_tot = result["history"]["val_total_per_epoch"]
    val_ce = result["history"]["val_ce_per_epoch"]
    val_rc = result["history"]["val_rc_term_per_epoch"]
    val_en = result["history"]["val_energy_per_epoch"]
    history_val = [
        {"total": val_tot[i], "ce": val_ce[i], "rc_term": val_rc[i], "energy": val_en[i]}
        for i in range(len(val_tot))
    ]

    suffix = " [CE-only train]" if task_only else ""
    plot_enhanced_curves(history_train, history_val, random_guess_ce, png_path, title_suffix=suffix)
    plot_enhanced_components(
        history_val,
        png_comp_path,
        title=f"Experiment 05 — val components (§6){suffix}",
    )

    result["artifacts"] = {
        "json": str(json_path.name),
        "png_curves": str(png_path.name),
        "png_val_components": str(png_comp_path.name),
    }

    if write_artifacts:
        _write_json(json_path, result)

    val_start = {k: result["val_start"][k] for k in result["val_start"]}
    val_end = {k: result["val_end"][k] for k in result["val_end"]}
    print("Experiment 05 — RFP Training Sanity [enhanced]" + (" [task-only train]" if task_only else ""))
    print(f"Device: {DEVICE} | epochs={n_ep} | task_only_train={task_only}")
    print(
        f"Val total: {val_start['total']:.4f} -> {val_end['total']:.4f} | best {result['best_val_total']:.4f} @ ep {result['best_epoch_total']}"
    )
    print(
        f"Val CE:    {val_start['ce']:.4f} -> {val_end['ce']:.4f} | best {result['best_val_ce']:.4f} @ ep {result['best_epoch_ce']} (ln V ~ {random_guess_ce:.4f})"
    )
    print(f"Grad OK: {result['grad_nonzero_after_step0']} | grad L2 (step0): {result['grad_l2_norm_step0']:.6f}")
    if strict_learning:
        se = result["criteria"].get("strict_learning_best_ce_below_random")
        print(f"Strict (--strict): best val CE < ln(V)-0.02 -> {'OK' if se else 'FAIL'}")
    print(f"PASS: {result['pass']} | JSON: {json_path}")
    print(f"PNG:  {png_path} | {png_comp_path}")
    return code


def _write_json(path: Path, result: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def _print_summary(mode: str, passed: bool, initial: float, final: float, baseline: float, grad_ok: bool, path):
    print(f"Experiment 05 — RFP Training Sanity [{mode}]")
    print(f"Loss: {initial:.4f} -> {final:.4f} (ln V ~ {baseline:.4f})")
    print(f"Grad OK: {grad_ok} | PASS: {passed}")
    if path:
        print(f"JSON: {path}")


def main():
    p = argparse.ArgumentParser(description="Experiment 05 — RFP training sanity")
    p.add_argument("--quick", action="store_true", help="Один батч, только CE (быстро)")
    p.add_argument("--epochs", type=int, default=None, help=f"Число эпох enhanced (по умолчанию {EPOCHS})")
    p.add_argument(
        "--skip-precheck",
        action="store_true",
        help="Пропустить проверки Phase 0 / знак homeostatic / формула L (не рекомендуется)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Жёстко: best val CE < ln(V)-0.02 (иначе FAIL даже при снижении loss от старта)",
    )
    p.add_argument(
        "--task-only",
        action="store_true",
        help="Обучение только по CE; val по-прежнему с полным разложением L (§6) для телеметрии",
    )
    args = p.parse_args()
    if args.quick:
        return run_quick(skip_precheck=args.skip_precheck)
    return run_enhanced(
        num_epochs=args.epochs,
        skip_precheck=args.skip_precheck,
        strict_learning=args.strict,
        task_only=args.task_only,
    )


if __name__ == "__main__":
    sys.exit(main())
