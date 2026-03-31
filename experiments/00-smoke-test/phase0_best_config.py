"""
Единый источник правды для конфигурации Phase 0 (Freq-Balanced и др.).

Использование: импорт в Experiment 05, smoke tests, чтобы частоты/пороги
не расходились с документацией и между фазами.
"""

from __future__ import annotations

# Согласовано с docs / Experiment 04 / Memory test (Freq-Balanced)
PHASE0_FREQ_BALANCED = {
    "name": "Freq-Balanced",
    "target_mode": "frequency",
    "frequencies": [1.0, 1.8, 3.0, 5.0],
    "thresholds": [0.20, 0.25, 0.30, 0.40],
    "spike_rate_target": 0.10,
    "homeostatic_eta": 0.01,
}


def assert_wfr_matches_phase0(wfr, num_layers: int = 4) -> None:
    """Проверяет, что слои WFRNetwork несут те же частоты/пороги, что и PHASE0_FREQ_BALANCED."""
    cfg = PHASE0_FREQ_BALANCED
    assert len(wfr.resonance_layers) == num_layers
    for i, layer in enumerate(wfr.resonance_layers):
        f_exp = cfg["frequencies"][i]
        t_exp = cfg["thresholds"][i]
        f_got = float(layer.frequency.data.item())
        t_got = float(layer.spike_threshold.data.item())
        if abs(f_got - f_exp) > 1e-5:
            raise AssertionError(
                f"Phase 0 freq mismatch layer {i}: expected {f_exp}, got {f_got}"
            )
        if abs(t_got - t_exp) > 1e-5:
            raise AssertionError(
                f"Phase 0 threshold mismatch layer {i}: expected {t_exp}, got {t_got}"
            )
    if getattr(wfr, "target_mode", None) != cfg["target_mode"]:
        raise AssertionError(
            f"target_mode: expected {cfg['target_mode']}, got {wfr.target_mode}"
        )
