"""
Полноценный протокол обучения Phase 1 (Experiment 05).

Соответствие документам:
  — docs/03-theory.md §6: L = α·Task + β·(1−AvgRC) + γ·EnergyCost;
  — docs/10-phase-1-plan.md §3.2 п.6: явное сравнение «только TaskLoss» vs полное L.

Шаги:
  1) Один раз precheck (Phase 0, знак homeostatic, согласованность compute_loss).
  2) Прогон A: обучение с полной L на train (как enhanced по умолчанию).
  3) Прогон B: обучение только по CE на тех же батчах; на val логируется полное разложение L
     (телеметрия поля при CE-only оптимизации).
  4) Сводный JSON + PNG сравнения кривых val (total L, CE).

Выход: 0 — прогон A PASS; 1 — A FAIL; 2 — precheck FAIL.
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
import torch

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
_smoke = _root.parent / "00-smoke-test"
sys.path.insert(0, str(_smoke))

from run_training_sanity import (  # noqa: E402
    DEVICE,
    EPOCHS,
    NUM_TRAIN_BATCHES,
    NUM_VAL_BATCHES,
    OUTPUT_DIR,
    SEED,
    build_fixed_batches,
    execute_enhanced_training,
)

OUTPUT_DIR.mkdir(exist_ok=True)


def plot_full_protocol_compare(
    history_a_total: list[float],
    history_a_ce: list[float],
    history_b_total: list[float],
    history_b_ce: list[float],
    ln_v: float,
    path_png: Path,
) -> None:
    epochs = list(range(1, len(history_a_total) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history_a_total, label="full L train", color="#2563eb", linewidth=1.2)
    ax1.plot(epochs, history_b_total, label="CE-only train (val total L)", color="#ea580c", linewidth=1.2)
    ax1.axhline(ln_v, color="gray", linestyle=":", alpha=0.7)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("val total L (§6)")
    ax1.set_title("Val: полная L — сравнение прогонов")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history_a_ce, label="full L train", color="#2563eb", linewidth=1.2)
    ax2.plot(epochs, history_b_ce, label="CE-only train", color="#ea580c", linewidth=1.2)
    ax2.axhline(ln_v, color="gray", linestyle=":", label=f"ln V = {ln_v:.3f}", alpha=0.7)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val CE")
    ax2.set_title("Val CE")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def run_protocol(
    num_epochs: int | None,
    skip_precheck: bool,
    strict_learning: bool,
    skip_ce_baseline: bool,
) -> int:
    n_ep = num_epochs if num_epochs is not None else EPOCHS

    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED)
    train_batches = build_fixed_batches(NUM_TRAIN_BATCHES, g)
    g2 = torch.Generator(device=DEVICE)
    g2.manual_seed(SEED + 99991)
    val_batches = build_fixed_batches(NUM_VAL_BATCHES, g2)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    merged_path = OUTPUT_DIR / f"training_full_protocol_{ts}.json"
    png_compare = OUTPUT_DIR / f"training_full_protocol_compare_{ts}.png"

    print("=== Experiment 05 — полноценный протокол Phase 1 ===")
    print(f"Эпох: {n_ep} | strict (прогон A): {strict_learning} | прогон B (CE-only): {not skip_ce_baseline}\n")

    r_a, code_a = execute_enhanced_training(
        num_epochs=n_ep,
        skip_precheck=skip_precheck,
        strict_learning=strict_learning,
        task_only=False,
        train_batches=train_batches,
        val_batches=val_batches,
    )
    if code_a == 2:
        print("PRECHECK FAILED — протокол прерван.")
        return 2

    r_b = None
    code_b = None
    if not skip_ce_baseline:
        r_b, code_b = execute_enhanced_training(
            num_epochs=n_ep,
            skip_precheck=True,
            strict_learning=False,
            task_only=True,
            train_batches=train_batches,
            val_batches=val_batches,
        )

    ln_v = float(r_a["random_baseline_ce_ln_v"])
    h_a = r_a["history"]
    h_b = r_b["history"] if r_b else None

    comparison = {
        "ln_vocab": ln_v,
        "run_A_full_L": {
            "pass": r_a.get("pass"),
            "best_val_ce": r_a.get("best_val_ce"),
            "best_val_total": r_a.get("best_val_total"),
            "strict_ok": (r_a.get("criteria") or {}).get("strict_learning_best_ce_below_random"),
        },
    }
    if r_b is not None:
        comparison["run_B_ce_only_train"] = {
            "pass": r_b.get("pass"),
            "best_val_ce": r_b.get("best_val_ce"),
            "best_val_total": r_b.get("best_val_total"),
        }
        comparison["delta_best_val_ce_B_minus_A"] = round(float(r_b["best_val_ce"]) - float(r_a["best_val_ce"]), 6)

    merged = {
        "protocol": "full_phase1_training",
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "config": {
            "epochs": n_ep,
            "strict_learning_run_A": strict_learning,
            "skip_ce_baseline": skip_ce_baseline,
            "theory_ref": "docs/03-theory.md §6, docs/10-phase-1-plan.md §3.2",
        },
        "run_A_full_L": r_a,
        "run_B_ce_only_train": r_b,
        "comparison": comparison,
    }

    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    if h_b is not None:
        plot_full_protocol_compare(
            h_a["val_total_per_epoch"],
            h_a["val_ce_per_epoch"],
            h_b["val_total_per_epoch"],
            h_b["val_ce_per_epoch"],
            ln_v,
            png_compare,
        )
        print(f"Сравнение PNG: {png_compare}")

    print(f"\nСводный JSON: {merged_path}")
    print(f"Прогон A (полная L) PASS: {r_a.get('pass')} | код: {code_a}")
    if code_b is not None:
        print(f"Прогон B (CE-only)   PASS: {r_b.get('pass')} | код: {code_b}")

    return code_a


def main() -> int:
    p = argparse.ArgumentParser(description="Experiment 05 — полноценный протокол Phase 1 (полная L vs CE-only)")
    p.add_argument("--epochs", type=int, default=None, help=f"Эпох (по умолчанию {EPOCHS})")
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Не требовать best val CE < ln(V)-0.02 для прогона A (полная L)",
    )
    p.add_argument(
        "--skip-precheck",
        action="store_true",
        help="Пропустить precheck (не рекомендуется)",
    )
    p.add_argument(
        "--skip-ce-baseline",
        action="store_true",
        help="Только прогон A (полная L), без сравнения с CE-only",
    )
    args = p.parse_args()
    return run_protocol(
        num_epochs=args.epochs,
        skip_precheck=args.skip_precheck,
        strict_learning=not args.no_strict,
        skip_ce_baseline=args.skip_ce_baseline,
    )


if __name__ == "__main__":
    sys.exit(main())
