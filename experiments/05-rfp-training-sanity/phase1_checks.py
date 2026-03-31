"""
Phase 1 — проверки согласованности теории, кода Phase 0 и обучения.

- Параметры Phase 0 реально лежат в слоях (частоты, пороги, target_mode).
- Homeostatic: отрицательная обратная связь (баг v2.0 был с инвертированным знаком).
- Скаляр L = alpha*CE + beta*(1-RC) + gamma*energy — согласованность compute_loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
_smoke = _root.parent / "00-smoke-test"
sys.path.insert(0, str(_smoke))

from phase0_best_config import assert_wfr_matches_phase0
from wfr_core import TheoreticalResonanceLayer

from wfr_losses import compute_loss, energy_cost, rc_penalty, task_loss_ce


def check_homeostatic_negative_feedback(device: torch.device) -> tuple[bool, str]:
    """
    При r_real < r_target порог должен снижаться: delta = eta * (r_real - r_target) < 0.
    Инверсия (r_target - r_real) давала бы рост порога при молчании — как в баге Phase 0.
    """
    layer = TheoreticalResonanceLayer(
        num_phases=16,
        frequency=1.0,
        threshold=2.0,
        layer_idx=0,
        homeostatic_enabled=True,
        spike_rate_target=0.10,
        homeostatic_eta=0.01,
    ).to(device)
    layer.eval()
    layer.spike_threshold.data.fill_(2.0)
    thr_before = float(layer.spike_threshold.item())
    phases = torch.randn(2, 64, 16, device=device, dtype=torch.float32) * 0.3
    with torch.no_grad():
        _ = layer(phases, target_mode="frequency")
    thr_after = float(layer.spike_threshold.item())
    if thr_after >= thr_before - 1e-9:
        return (
            False,
            "homeostatic: ожидалось снижение порога при недостаточной активности спайков; "
            f"threshold {thr_before:.6f} -> {thr_after:.6f}. Проверьте знак (r_real - r_target) в wfr_core.",
        )
    return True, "homeostatic sign OK (порог снизился при r_real < r_target)"


def check_loss_formula_matches(
    model,
    batch: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[bool, str]:
    """compute_loss согласован с разложением на CE, (1-RC), energy."""
    model.eval()
    logits, out = model(batch)
    total, _ = compute_loss(logits, batch, out, alpha, beta, gamma, task_only=False)
    ce = task_loss_ce(logits, batch)
    rc_t = rc_penalty(out)
    en = energy_cost(out)
    manual = alpha * ce + beta * rc_t + gamma * en
    diff = float((total - manual).abs().item())
    if diff > 1e-4:
        return False, f"расхождение L: |total - manual| = {diff}"
    return True, f"L formula OK (diff {diff:.2e})"


def run_all_prechecks(
    probe,
    sample_batch: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    device: torch.device,
) -> dict[str, tuple[bool, str]]:
    """Возвращает {имя: (ok, msg)}."""
    results: dict[str, tuple[bool, str]] = {}

    try:
        assert_wfr_matches_phase0(probe.wfr, num_layers=len(probe.wfr.resonance_layers))
        results["phase0_params_in_wfr"] = (
            True,
            "частоты/пороги/target_mode совпадают с phase0_best_config",
        )
    except AssertionError as e:
        results["phase0_params_in_wfr"] = (False, str(e))

    results["homeostatic_sign"] = check_homeostatic_negative_feedback(device)
    results["loss_formula"] = check_loss_formula_matches(probe, sample_batch, alpha, beta, gamma)

    return results


def precheck_must_pass(results: dict[str, tuple[bool, str]]) -> bool:
    return all(v[0] for v in results.values())


def print_precheck(results: dict[str, tuple[bool, str]]) -> None:
    print("--- Phase 1 precheck (теория / Phase 0 / знак homeostatic) ---")
    for name, (ok, msg) in results.items():
        tag = "OK" if ok else "FAIL"
        print(f"  [{tag}] {name}: {msg}")


if __name__ == "__main__":
    # Быстрый автономный прогон проверок (без полного обучения)
    from run_training_sanity import ALPHA, BETA, GAMMA, BATCH_SIZE, SEQ_LEN, SEED, VOCAB_SIZE, WFRNextTokenProbe

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=g, device=dev)
    probe = WFRNextTokenProbe(VOCAB_SIZE).to(dev)
    r = run_all_prechecks(probe, batch, ALPHA, BETA, GAMMA, dev)
    print_precheck(r)
    sys.exit(0 if precheck_must_pass(r) else 2)
